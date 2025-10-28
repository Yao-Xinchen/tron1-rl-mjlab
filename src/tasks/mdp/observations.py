"""Observation functions for the robot."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_env import ManagerBasedEnv
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def joint_acc(
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    jnt_ids = asset_cfg.joint_ids
    return asset.data.joint_acc[:, jnt_ids]


def actuator_force(
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    if not asset.data.is_actuated:
        raise ValueError(f"Entity '{asset_cfg.name}' is not actuated.")
    return asset.data.actuator_force


def body_lin_vel(
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    return asset.data.body_link_lin_vel_w[:, body_ids].flatten(start_dim=1)


def joint_stiffness(
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    jnt_ids = asset_cfg.joint_ids
    return asset.data.default_joint_stiffness[:, jnt_ids]


def joint_damping(
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    jnt_ids = asset_cfg.joint_ids
    return asset.data.default_joint_damping[:, jnt_ids]


def base_height_error(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
        base_height_target: float = 0.9,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    if not hasattr(env, '_wheels_link_ids') or not hasattr(env, '_foot_radius'):
        return torch.zeros((env.num_envs, 1), device=env.device)

    foot_position = asset.data.body_link_pos_w[:, env._wheels_link_ids, :]
    base_height_w = asset.data.root_link_pos_w[:, 2] - foot_position[:, :, 2].mean(dim=-1) + 0.127

    return (base_height_w - base_height_target).unsqueeze(1)


def foot_rel_position_w(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    if not hasattr(env, '_wheels_link_ids') or not hasattr(env, '_foot_radius'):
        return torch.zeros((env.num_envs, 6), device=env.device)

    foot_position_w = asset.data.body_link_pos_w[:, env._wheels_link_ids, :]
    base_position_w = asset.data.root_link_pos_w

    return (foot_position_w - base_position_w.unsqueeze(1)).view(env.num_envs, -1)


def contact_forces(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
        sensor_name: str = "contact_sensors",
) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]
    contact_force = robot.data.sensor_data[sensor_name]
    return contact_force.flatten(start_dim=1)
