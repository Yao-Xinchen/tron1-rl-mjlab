# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import MLP
from rsl_rl.modules.distribution import Distribution

from .mlp_model import MLPModel


class EncoderModel(MLPModel):
    """Actor with a history encoder, privileged-obs decoder, and velocity estimator head.

    - Encoder MLP : history obs → latent z
    - Actor MLP   : [actor_obs, z] → action  (inherited from MLPModel)
    - Decoder MLP : z → privileged obs        (training only; reconstruction loss)
    - Velocity head: z → velocity estimate    (supervised; shared buffer for env feedback)
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        encoder_hidden_dims: tuple[int, ...] = (256, 128),
        encoder_latent_dim: int = 32,
        decoder_hidden_dims: tuple[int, ...] = (128, 256),
        actor_obs_key: str = "actor",
        history_obs_key: str = "history",
        privileged_obs_key: str = "critic",
        velocity_obs_key: str = "velocity",
        decoder_loss_coef: float = 1.0,
        velocity_loss_coef: float = 1.0,
    ) -> None:
        # Set encoder attributes before super().__init__ so overridden
        # _get_obs_dim / _get_latent_dim see the correct values.
        self.actor_obs_key = actor_obs_key
        self.history_obs_key = history_obs_key
        self.privileged_obs_key = privileged_obs_key
        self.velocity_obs_key = velocity_obs_key
        self.encoder_latent_dim = encoder_latent_dim
        self.decoder_loss_coef = decoder_loss_coef
        self.velocity_loss_coef = velocity_loss_coef

        super().__init__(
            obs, obs_groups, obs_set, output_dim,
            hidden_dims, activation, obs_normalization, distribution_cfg,
        )

        history_dim = obs[history_obs_key].shape[-1]
        privileged_dim = obs[privileged_obs_key].shape[-1]
        vel_dim = obs[velocity_obs_key].shape[-1]
        num_envs = obs[actor_obs_key].shape[0]

        self.encoder_mlp = MLP(history_dim, encoder_latent_dim, encoder_hidden_dims, activation)
        self.decoder_mlp = MLP(encoder_latent_dim, privileged_dim, decoder_hidden_dims, activation)
        self.vel_head = MLP(encoder_latent_dim, vel_dim, decoder_hidden_dims, activation)

        # Sizes stored for export helpers.
        self._history_size = history_dim
        self._vel_dim = vel_dim

        # Non-persistent buffer shared with the command manager (moves with model, written in-place).
        # Shape is (num_envs, vel_dim) — only updated during rollout, not minibatch updates.
        # persistent=False excludes it from state_dict so num_envs doesn't leak into checkpoints.
        self.register_buffer("_vel_est", torch.zeros(num_envs, vel_dim), persistent=False)
        self._decoder_out: torch.Tensor | None = None
        # Current-forward-pass output (may be minibatch-sized); read by PPO loss.
        self._vel_est_current: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # MLPModel overrides
    # ------------------------------------------------------------------

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict, obs_set: str) -> tuple[list[str], int]:
        actor_dim = obs[self.actor_obs_key].shape[-1]
        return [self.actor_obs_key], actor_dim

    def _get_latent_dim(self) -> int:
        return self.obs_dim + self.encoder_latent_dim

    def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
        z = self.encoder_mlp(obs[self.history_obs_key])
        vel_out = self.vel_head(z)
        self._decoder_out = self.decoder_mlp(z)
        # Update the shared buffer only during rollout (full num_envs batch).
        # During PPO minibatch updates the batch size differs — skip the copy.
        if vel_out.shape[0] == self._vel_est.shape[0]:  # type: ignore[union-attr]
            self._vel_est.copy_(vel_out)  # type: ignore[union-attr]
        self._vel_est_current = vel_out
        actor_obs = self.obs_normalizer(obs[self.actor_obs_key])  # type: ignore[operator]
        return torch.cat([actor_obs, z], dim=-1)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # _vel_est is a non-persistent runtime buffer — strip it from old checkpoints
        # that saved it as a persistent buffer before this was changed.
        state_dict.pop("_vel_est", None)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    # ------------------------------------------------------------------
    # Encoder-specific accessors (called by PPO after each forward pass)
    # ------------------------------------------------------------------

    def get_decoder_out(self) -> torch.Tensor:
        """Privileged obs reconstruction from the most recent forward pass."""
        if self._decoder_out is None:
            raise RuntimeError("Call forward() before get_decoder_out()")
        return self._decoder_out

    def get_vel_est(self) -> torch.Tensor:
        """Velocity estimate from the most recent forward pass, used by PPO for the velocity loss."""
        if self._vel_est_current is None:
            raise RuntimeError("Call forward() before get_vel_est()")
        return self._vel_est_current

    # ------------------------------------------------------------------
    # Export (decoder excluded — not needed at deployment)
    # ------------------------------------------------------------------

    def as_jit(self) -> nn.Module:
        return _TorchEncoderModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _OnnxEncoderModel(self, verbose)


class _TorchEncoderModel(nn.Module):
    """JIT-exportable encoder model (encoder + vel head + actor; no decoder)."""

    def __init__(self, model: EncoderModel) -> None:
        super().__init__()
        self.encoder_mlp = copy.deepcopy(model.encoder_mlp)
        self.vel_head = copy.deepcopy(model.vel_head)
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        dist: Distribution | None = model.distribution
        if dist is not None:
            self.deterministic_output = dist.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, actor_obs: torch.Tensor, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder_mlp(history)
        vel_est = self.vel_head(z)
        actor_input = torch.cat([self.obs_normalizer(actor_obs), z], dim=-1)  # type: ignore[operator]
        return self.deterministic_output(self.mlp(actor_input)), vel_est

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxEncoderModel(nn.Module):
    """ONNX-exportable encoder model (encoder + vel head + actor; no decoder)."""

    is_recurrent: bool = False

    def __init__(self, model: EncoderModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.encoder_mlp = copy.deepcopy(model.encoder_mlp)
        self.vel_head = copy.deepcopy(model.vel_head)
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        dist: Distribution | None = model.distribution
        if dist is not None:
            self.deterministic_output = dist.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.actor_obs_size = model.obs_dim
        self.history_size = model._history_size

    def forward(self, actor_obs: torch.Tensor, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder_mlp(history)
        vel_est = self.vel_head(z)
        actor_input = torch.cat([self.obs_normalizer(actor_obs), z], dim=-1)  # type: ignore[operator]
        return self.deterministic_output(self.mlp(actor_input)), vel_est

    def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (torch.zeros(1, self.actor_obs_size), torch.zeros(1, self.history_size))

    @property
    def input_names(self) -> list[str]:
        return ["actor_obs", "history"]

    @property
    def output_names(self) -> list[str]:
        return ["actions", "vel_est"]
