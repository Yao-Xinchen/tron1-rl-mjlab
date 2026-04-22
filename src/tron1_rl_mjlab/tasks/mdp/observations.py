"""Observation functions for the robot."""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
    matrix_from_quat,
    quat_unique,
)

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def joint_acc(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.joint_acc[:, asset_cfg.joint_ids]


def actuator_force(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    if not asset.data.is_actuated:
        raise ValueError(f"Entity '{asset_cfg.name}' is not actuated.")
    return asset.data.actuator_force


def body_lin_vel(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids].flatten(start_dim=1)


def base_height_error(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
        base_height_target: float = 0.9,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    if not hasattr(env, '_wheels_link_ids') or not hasattr(env, '_foot_radius'):
        return torch.zeros((env.num_envs, 1), device=env.device)

    foot_position = asset.data.body_link_pos_w[:, env._wheels_link_ids, :]
    base_height_w = asset.data.root_link_pos_w[:, 2] - foot_position[:, :, 2].mean(dim=-1) + env._foot_radius

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
    sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = sensor.data
    assert sensor_data.force is not None, f"Contact sensor '{sensor_name}' has no force data"
    return sensor_data.force.flatten(start_dim=1)


def base_commands_b(
        env: ManagerBasedRlEnv,
        command_name: str = "base_pose",
) -> torch.Tensor:
    target_pose_b = env.command_manager.get_command(command_name)
    target_pose_xy = target_pose_b[:, :2]
    target_orientation_x = matrix_from_quat(quat_unique(target_pose_b[:, 3:7]))[:, :, 0]
    return torch.cat([target_pose_xy, target_orientation_x[:, :2]], dim=-1)


def fake_base_commands_b(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Return zero base commands (stationary target at current pose)."""
    target_pose_xy = torch.zeros((env.num_envs, 2), device=env.device)
    target_orientation_x = torch.zeros((env.num_envs, 2), device=env.device)
    target_orientation_x[:, 0] = 1.0
    return torch.cat([target_pose_xy, target_orientation_x], dim=-1)


def base_se3_decrease_rate(
        env: ManagerBasedRlEnv,
        command_name: str = "base_pose",
) -> torch.Tensor:
    return env.command_manager.get_term(command_name).se3_decay_rate.unsqueeze(-1)


def base_commands_vel_c(
        env: ManagerBasedRlEnv,
        command_name: str = "base_pose",
) -> torch.Tensor:
    return env.command_manager.get_term(command_name).vel_command_t
