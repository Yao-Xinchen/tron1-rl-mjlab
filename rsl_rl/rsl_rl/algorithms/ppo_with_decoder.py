# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO with GRU reconstruction loss (joint) and CNN autoencoder (separate)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import unpad_trajectories


class PPOWithDecoder(PPO):
    """PPO with two decoder training strategies per rollout.

    GRU reconstruction (joint with PPO)::

        Inside every PPO mini-batch, after the actor forward pass, the cached
        GRU latent is passed through privileged_decoder and height_map_decoder.
        The MSE reconstruction loss is added to the PPO loss and optimised in
        the same backward/step call.  This shares ``num_mini_batches`` and
        ``num_learning_epochs`` with PPO — no extra parameters.

    CNN autoencoder (separate, before PPO)::

        Raw depth images buffered during rollout are passed through the CNN and
        depth_decoder to reconstruct the original image.  Only CNN and
        depth_decoder receive gradients.  ``cnn_image_fraction`` controls what
        fraction of the rollout images are used; ``cnn_mini_batch_size`` caps
        the CNN activation memory per step.

    Update order per iteration::

        1. CNN autoencoder  (raw images → CNN → depth_decoder → MSE)
        2. PPO loop         (stored features → GRU → actor + GRU recon decoders)
           └─ _compute_auxiliary_loss called inside every mini-batch
        3. storage.clear() + empty_cache()

    Args:
        privileged_obs_group: Key in rollout observations for privileged target.
        height_map_obs_group: Key in rollout observations for height-map target.
        depth_obs_group: Key in observations containing raw depth images.
        recon_loss_coef: Weight applied to all reconstruction losses.
        cnn_image_fraction: Fraction of rollout images used per CNN autoencoder
            update (e.g. 0.25 → 25 % of T×B images).
        cnn_mini_batch_size: Images per CNN gradient step; controls peak CNN
            activation memory (roughly proportional to this value).
    """

    def __init__(
            self,
            actor,
            critic,
            storage: RolloutStorage,
            privileged_obs_group: str = "critic",
            height_map_obs_group: str = "height_map",
            depth_obs_group: str = "depth_camera",
            recon_loss_coef: float = 1.0,
            cnn_image_fraction: float = 0.25,
            cnn_mini_batch_size: int = 768,
            **ppo_kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, **ppo_kwargs)
        self.privileged_obs_group = privileged_obs_group
        self.height_map_obs_group = height_map_obs_group
        self.depth_obs_group = depth_obs_group
        self.recon_loss_coef = recon_loss_coef
        self.cnn_image_fraction = cnn_image_fraction
        self.cnn_mini_batch_size = cnn_mini_batch_size
        self._depth_image_buffer: torch.Tensor | None = None

        # Replace [T, B, H, W] depth storage with [T, B, D_cnn] feature storage.
        self._resize_depth_storage()

    # ------------------------------------------------------------------
    # Storage helpers
    # ------------------------------------------------------------------

    def _resize_depth_storage(self) -> None:
        obs_td = self.storage.observations
        if self.depth_obs_group not in obs_td.keys():
            return
        T, B = obs_td.batch_size
        D: int = self.actor.cnn_output_dim  # type: ignore[assignment]
        obs_td[self.depth_obs_group] = torch.zeros(T, B, D, device=self.storage.device)

    # ------------------------------------------------------------------
    # Rollout: precompute CNN features and buffer raw images
    # ------------------------------------------------------------------

    def act(self, obs: TensorDict) -> torch.Tensor:
        depth_images = obs[self.depth_obs_group]

        if self._depth_image_buffer is None:
            T = self.storage.num_transitions_per_env
            self._depth_image_buffer = torch.zeros(
                T, *depth_images.shape, device=self.device, dtype=depth_images.dtype
            )

        self._depth_image_buffer[self.storage.step].copy_(depth_images)

        with torch.no_grad():
            features = self.actor.encode_depth(depth_images)  # type: ignore[attr-defined]

        obs = obs.clone(recurse=False)
        obs[self.depth_obs_group] = features
        return super().act(obs)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self) -> dict[str, float]:
        # CNN autoencoder first — raw images used here, then dormant
        cnn_logs = self._update_cnn_autoencoder()
        # PPO loop includes GRU reconstruction via _compute_auxiliary_loss
        loss_dict = super().update()
        loss_dict.update(cnn_logs)
        return loss_dict

    # ------------------------------------------------------------------
    # GRU reconstruction: auxiliary loss hook inside PPO mini-batch loop
    # ------------------------------------------------------------------

    def _compute_auxiliary_loss(
            self, batch: RolloutStorage.Batch, original_batch_size: int
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """MSE reconstruction loss from GRU latent; called inside every PPO mini-batch."""
        gru_latent = self.actor._last_gru_latent  # type: ignore[attr-defined]
        if gru_latent is None:
            return torch.zeros(1, device=self.device).squeeze(), {}

        priv_pred = self.actor.privileged_decoder(gru_latent)  # type: ignore[attr-defined]
        hmap_pred = self.actor.height_map_decoder(gru_latent)  # type: ignore[attr-defined]

        if batch.masks is not None:
            priv_target = unpad_trajectories(
                batch.observations[self.privileged_obs_group], batch.masks  # type: ignore[index]
            ).detach()
            hmap_target = unpad_trajectories(
                batch.observations[self.height_map_obs_group], batch.masks  # type: ignore[index]
            ).detach()
        else:
            priv_target = batch.observations[self.privileged_obs_group].detach()  # type: ignore[index]
            hmap_target = batch.observations[self.height_map_obs_group].detach()  # type: ignore[index]

        priv_loss = F.mse_loss(priv_pred, priv_target)
        hmap_loss = F.mse_loss(hmap_pred, hmap_target)
        aux_loss = self.recon_loss_coef * (priv_loss + hmap_loss)

        return aux_loss, {
            "recon_privileged": priv_loss.item(),
            "recon_height_map": hmap_loss.item(),
        }

    # ------------------------------------------------------------------
    # CNN autoencoder: separate update using buffered raw images
    # ------------------------------------------------------------------

    def _update_cnn_autoencoder(self) -> dict[str, float]:
        """Train CNN + depth_decoder via depth image reconstruction.

        Uses ``cnn_image_fraction`` of the rollout images, processed in steps
        of ``cnn_mini_batch_size``.  Only CNN and depth_decoder are updated.
        """
        if self._depth_image_buffer is None:
            return {}

        actor = self.actor
        T = self.storage.num_transitions_per_env
        N = T * self.storage.num_envs

        n_use = max(self.cnn_mini_batch_size, int(N * self.cnn_image_fraction))
        n_use = min(n_use, N)
        n_steps = n_use // self.cnn_mini_batch_size

        images = self._depth_image_buffer[:T].flatten(0, 1)  # [N, H, W]
        idx = torch.randperm(N, device=self.device)[:n_steps * self.cnn_mini_batch_size]

        total_loss = 0.0
        for i in range(n_steps):
            batch_idx = idx[i * self.cnn_mini_batch_size: (i + 1) * self.cnn_mini_batch_size]
            imgs = images[batch_idx]
            features = actor.encode_depth(imgs)  # type: ignore[attr-defined]
            depth_recon = actor.depth_decoder(features)  # type: ignore[attr-defined]
            loss = F.mse_loss(depth_recon, imgs.flatten(1))

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item()

        return {"recon_depth": total_loss / n_steps}
