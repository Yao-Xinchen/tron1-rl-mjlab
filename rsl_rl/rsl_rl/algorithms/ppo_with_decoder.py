# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO extended with supervised reconstruction losses from GRU latent and CNN features."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import unpad_trajectories


class PPOWithDecoder(PPO):
    """PPO + MSE reconstruction losses from two independent decoder paths.

    Option B architecture: two separate decoder sets train two separate encoders::

        gru_latent  → privileged_decoder / height_map_decoder
                      gradient: GRU + proprio_encoder + these decoders

        cnn_feat    → cnn_privileged_decoder / cnn_height_map_decoder
                      gradient: CNN + these decoders (CNN re-run with grad)

    Total loss::

        L = L_ppo
          + recon_loss_coef * [MSE(decode_priv(gru), obs["critic"])
                             + MSE(decode_hmap(gru), obs["height_map"])
                             + MSE(cnn_decode_priv(cnn), obs["critic"])
                             + MSE(cnn_decode_hmap(cnn), obs["height_map"])]

    The actor model is expected to expose ``get_last_decoder_outputs(masks)``
    returning a dict with keys ``"privileged"``, ``"height_map"``,
    ``"cnn_privileged"``, ``"cnn_height_map"``.

    Args:
        privileged_obs_group: Key in ``batch.observations`` for the privileged target.
        height_map_obs_group: Key in ``batch.observations`` for the height-map target.
        recon_loss_coef: Weight applied to the total reconstruction loss.
    """

    def __init__(
        self,
        actor,
        critic,
        storage: RolloutStorage,
        privileged_obs_group: str = "critic",
        height_map_obs_group: str = "height_map",
        recon_loss_coef: float = 1.0,
        **ppo_kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, **ppo_kwargs)
        self.privileged_obs_group = privileged_obs_group
        self.height_map_obs_group = height_map_obs_group
        self.recon_loss_coef = recon_loss_coef

    def _compute_auxiliary_loss(
        self, batch: RolloutStorage.Batch, original_batch_size: int
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute MSE reconstruction losses from GRU and CNN decoder heads."""
        decoder_outputs = self.actor.get_last_decoder_outputs()  # type: ignore[attr-defined]
        if decoder_outputs is None:
            return torch.zeros(1, device=self.device).squeeze(), {}

        priv_pred = decoder_outputs["privileged"]
        hmap_pred = decoder_outputs["height_map"]

        # Unpad observation targets when in recurrent (padded trajectory) mode
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
