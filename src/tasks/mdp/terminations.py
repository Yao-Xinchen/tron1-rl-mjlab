"""Termination functions for the task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def bad_orientation_stochastic(
        env: ManagerBasedRlEnv,
        limit_angle: float,
        probability: float,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    bad_orientation = torch.acos(-asset.data.projected_gravity_b[:, 2]).abs() > limit_angle
    random_values = torch.rand(bad_orientation.shape, device=bad_orientation.device)
    return bad_orientation & (random_values < probability)


def bad_height_stochastic(
        env: ManagerBasedRlEnv,
        limit_height: float,
        probability: float,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    foot_position = asset.data.body_link_pos_w[:, env._wheels_link_ids, :]
    height = asset.data.root_link_pos_w[:, 2] - foot_position[:, :, 2].mean(dim=-1) + env._foot_radius
    bad_height = height < limit_height
    random_values = torch.rand(bad_height.shape, device=bad_height.device)
    return bad_height & (random_values < probability)
