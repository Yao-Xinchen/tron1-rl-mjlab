# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Actor model: 2D CNN (current depth frame) + MLP (proprio) → GRU → latent."""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP, RNN, HiddenState
from rsl_rl.utils import resolve_nn_activation, unpad_trajectories


class EncoderRNNActorModel(MLPModel):
    """Actor with dual encoders feeding a GRU, plus two sets of decoder heads.

    Forward path::

        actor obs  [B, D_actor]   → MLP proprio encoder  → [B, D_proprio]
        depth obs  [B, H, W]      → 2D CNN               → [B, D_vision]
        cat([proprio, vision])    → GRU                  → [B, D_gru]
        cat([actor_obs, gru_out]) → actor head MLP        → actions

    Decoder heads and their gradient targets::

        CNN features → depth_decoder       → reconstructed depth image  (trains CNN)
        GRU latent   → privileged_decoder  → privileged obs             (trains GRU)
        GRU latent   → height_map_decoder  → height map                 (trains GRU)

    The two decoder updates are kept separate in PPOWithDecoder:
    - CNN autoencoder runs on buffered raw images after PPO (no rollout storage needed).
    - GRU decoders run on stored CNN features before PPO (rollout storage still live).

    Args:
        obs: Sample observation TensorDict from the environment.
        obs_groups: Mapping from obs-set names to lists of obs-group names.
        obs_set: Which obs-set this model reads (e.g. ``"actor"``).
        output_dim: Number of action dimensions.
        hidden_dims: Hidden dims of the actor head MLP.
        activation: Activation function name.
        obs_normalization: Whether to apply empirical normalisation to 1D obs.
        distribution_cfg: Config dict for the output distribution.
        actor_obs_group: Name of the 1D proprioceptive obs group.
        proprio_encoder_hidden_dims: Hidden dims of the proprioceptive MLP encoder.
        proprio_encoder_output_dim: Output dim of the proprioceptive encoder.
        depth_obs_group: Name of the depth-image obs group (shape [B, H, W]).
        cnn_output_channels: Output channel counts for each Conv2d layer.
        cnn_kernel_size: Kernel size shared across all Conv2d layers.
        cnn_strides: Stride for each Conv2d layer.
        cnn_output_dim: Output dim of the 2D CNN (after linear projection).
        rnn_hidden_dim: GRU hidden state dimension.
        rnn_num_layers: Number of GRU layers.
        privileged_decoder_obs_group: Obs group used to infer privileged target dim.
        height_map_decoder_obs_group: Obs group used to infer height-map target dim.
        decoder_hidden_dims: Hidden dims for the GRU-based decoder MLPs.
        depth_decoder_hidden_dims: Hidden dims for the CNN depth-autoencoder MLP.
    """

    is_recurrent: bool = True

    def __init__(
            self,
            obs: TensorDict,
            obs_groups: dict[str, list[str]],
            obs_set: str,
            output_dim: int,
            hidden_dims: tuple[int, ...] | list[int] = (256, 128),
            activation: str = "elu",
            obs_normalization: bool = False,
            distribution_cfg: dict | None = None,
            # Proprio encoder
            actor_obs_group: str = "actor",
            proprio_encoder_hidden_dims: tuple[int, ...] = (256, 128),
            proprio_encoder_output_dim: int = 64,
            # 2D CNN
            depth_obs_group: str = "depth_camera",
            cnn_output_channels: tuple[int, ...] = (32, 64, 64),
            cnn_kernel_size: int = 3,
            cnn_strides: tuple[int, ...] = (2, 2, 2),
            cnn_output_dim: int = 128,
            # GRU
            rnn_hidden_dim: int = 256,
            rnn_num_layers: int = 1,
            # GRU decoder heads
            privileged_decoder_obs_group: str = "critic",
            height_map_decoder_obs_group: str = "height_map",
            decoder_hidden_dims: tuple[int, ...] = (512, 256),
            # CNN depth autoencoder decoder
            depth_decoder_hidden_dims: tuple[int, ...] = (256, 512),
    ) -> None:
        # Store before super().__init__ which calls _get_obs_dim / _get_latent_dim
        self._actor_obs_group = actor_obs_group
        self._depth_obs_group = depth_obs_group
        self._rnn_hidden_dim = rnn_hidden_dim
        self.cnn_output_dim = cnn_output_dim

        super().__init__(
            obs, obs_groups, obs_set, output_dim,
            hidden_dims, activation, obs_normalization, distribution_cfg,
        )

        act_fn = resolve_nn_activation(activation)

        # --- Proprio encoder ---------------------------------------------------
        self.proprio_encoder = MLP(
            self.obs_dim, proprio_encoder_output_dim,
            proprio_encoder_hidden_dims, activation,
        )

        # --- 2D CNN ------------------------------------------------------------
        cnn_layers: list[nn.Module] = []
        in_ch = 1
        for out_ch, stride in zip(cnn_output_channels, cnn_strides):
            pad = cnn_kernel_size // 2
            cnn_layers += [
                nn.Conv2d(in_ch, out_ch, kernel_size=cnn_kernel_size,
                          stride=stride, padding=pad),
                act_fn,
            ]
            in_ch = out_ch
        cnn_layers += [
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, cnn_output_dim),
            act_fn,
        ]
        self.cnn2d = nn.Sequential(*cnn_layers)

        # --- GRU ---------------------------------------------------------------
        gru_input_dim = proprio_encoder_output_dim + cnn_output_dim
        self.rnn = RNN(gru_input_dim, rnn_hidden_dim, rnn_num_layers, "gru")

        # --- GRU decoder heads (GRU latent → privileged obs / height map) -----
        privileged_dim = int(obs[privileged_decoder_obs_group].shape[-1])
        height_map_dim = int(obs[height_map_decoder_obs_group].shape[-1])
        self.privileged_decoder = MLP(rnn_hidden_dim, privileged_dim, decoder_hidden_dims, activation)
        self.height_map_decoder = MLP(rnn_hidden_dim, height_map_dim, decoder_hidden_dims, activation)

        # --- CNN depth autoencoder (CNN features → reconstructed depth image) -
        depth_h = int(obs[depth_obs_group].shape[-2])
        depth_w = int(obs[depth_obs_group].shape[-1])
        self.depth_decoder = MLP(cnn_output_dim, depth_h * depth_w, depth_decoder_hidden_dims, activation)

        self._last_gru_latent: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # MLPModel overrides
    # ------------------------------------------------------------------

    def _get_obs_dim(
            self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str
    ) -> tuple[list[str], int]:
        """Classify each active obs group as 1D (MLP) or image (CNN); return 1D groups."""
        active_groups = obs_groups[obs_set]
        obs_groups_1d: list[str] = []
        obs_dim_1d = 0
        for group in active_groups:
            shape = obs[group].shape
            if len(shape) > 2:
                if group != self._depth_obs_group:
                    raise ValueError(
                        f"Unexpected multi-dimensional obs group '{group}'. "
                        f"Only '{self._depth_obs_group}' is handled by the 2D CNN."
                    )
                continue
            obs_groups_1d.append(group)
            obs_dim_1d += shape[-1]
        return obs_groups_1d, obs_dim_1d

    def _get_latent_dim(self) -> int:
        return self.obs_dim + self._rnn_hidden_dim

    def encode_depth(self, depth: torch.Tensor) -> torch.Tensor:
        """Run CNN on a depth image batch. ``[B, H, W]`` → ``[B, cnn_output_dim]``."""
        return self.cnn2d(depth.unsqueeze(1))

    def get_latent(
            self,
            obs: TensorDict,
            masks: torch.Tensor | None = None,
            hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        """Encode obs and return ``cat(actor_obs_normed, gru_latent)``.

        Accepts either raw depth images ``[B, H, W]`` or pre-computed CNN
        features ``[B, D_cnn]`` (stored by PPOWithDecoder during rollout).
        """
        batch_mode = masks is not None

        obs_1d = torch.cat([obs[g] for g in self.obs_groups], dim=-1)
        obs_1d_normed = self.obs_normalizer(obs_1d)

        depth = obs[self._depth_obs_group]

        # Distinguish raw images from pre-computed features by ndim:
        #   rollout mode : images [B, H, W] ndim=3 ; features [B, D] ndim=2
        #   batch mode   : images [T, B, H, W] ndim=4 ; features [T, B, D] ndim=3
        is_precomputed = depth.ndim < (4 if batch_mode else 3)

        if batch_mode:
            prefix = depth.shape[:2]
            proprio_enc = self.proprio_encoder(obs_1d_normed)

            if is_precomputed:
                vision_enc = depth
            else:
                depth_flat = depth.flatten(0, 1).unsqueeze(1)
                vision_enc = self.cnn2d(depth_flat).view(*prefix, -1)

            gru_input = torch.cat([proprio_enc, vision_enc], dim=-1)
            gru_latent = self.rnn(gru_input, masks, hidden_state)
            self._last_gru_latent = gru_latent
            actor_obs = unpad_trajectories(obs_1d_normed, masks)
        else:
            proprio_enc = self.proprio_encoder(obs_1d_normed)

            if is_precomputed:
                vision_enc = depth
            else:
                vision_enc = self.cnn2d(depth.unsqueeze(1))

            gru_input = torch.cat([proprio_enc, vision_enc], dim=-1)
            gru_latent = self.rnn(gru_input, None, None).squeeze(0)
            self._last_gru_latent = gru_latent
            actor_obs = obs_1d_normed

        return torch.cat([actor_obs, gru_latent], dim=-1)

    # ------------------------------------------------------------------
    # Recurrent interface
    # ------------------------------------------------------------------

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        self.rnn.reset(dones, hidden_state)

    def get_hidden_state(self) -> HiddenState:
        return self.rnn.hidden_state

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        self.rnn.detach_hidden_state(dones)
