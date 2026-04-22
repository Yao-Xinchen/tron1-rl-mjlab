"""Reward functions for the task."""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def safety_reward_exp(
        env: ManagerBasedRlEnv,
        std: float,
        base_height_target: float,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward safety of base position and orientation using exponential kernel."""
    asset: Entity = env.scene[asset_cfg.name]

    base_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
    base_position = asset.data.root_link_pos_w.unsqueeze(1).expand(-1, 2, -1)

    foot_position = asset.data.body_link_pos_w[:, env._wheels_link_ids, :]
    foot_position_b = quat_apply_inverse(base_quat, foot_position - base_position)
    base_height = asset.data.root_link_pos_w[:, 2] - foot_position[:, :, 2].mean(dim=-1) + env._foot_radius

    foot_pos_error_b = foot_position_b[:, :, :2] - env._nominal_foot_position_b[:, :2]

    # adduction is penalized harder than abduction
    adduction = ((env._nominal_foot_position_b[:, 1] > 0.0) * (foot_pos_error_b[:, :, 1] < 0.0)) | (
            (env._nominal_foot_position_b[:, 1] < 0.0) * (foot_pos_error_b[:, :, 1] > 0.0)
    )
    foot_pos_error_b[:, :, 1] = torch.where(
        adduction, foot_pos_error_b[:, :, 1] / 0.1, foot_pos_error_b[:, :, 1] / 0.2
    )
    foot_pos_error_b[:, :, 0] = foot_pos_error_b[:, :, 0] / 0.2

    foot_pos_error_b = torch.clamp(foot_pos_error_b.abs().sum(dim=-1).sum(dim=-1), max=8.0)

    base_orient_error_roll = torch.abs(asset.data.projected_gravity_b[:, 1]) / 0.1
    base_orient_error_pitch = torch.abs(asset.data.projected_gravity_b[:, 0]) / 0.85
    base_height_error = ((base_height - base_height_target) / 0.1) ** 2

    wheel_vel_error = (torch.sum(torch.abs(asset.data.joint_vel[:, env._wheels_joint_ids]), dim=1) / 3.0).clip(max=4)
    base_lin_vel_error = torch.norm(asset.data.root_link_lin_vel_b, p=2, dim=1) / 0.5
    base_ang_vel_error = torch.norm(asset.data.root_link_ang_vel_b, p=2, dim=1) / 1.2

    normalized_mani_error = (
            (
                    foot_pos_error_b
                    + wheel_vel_error
                    + base_lin_vel_error
                    + base_ang_vel_error
                    + base_height_error * 0.5
                    + base_orient_error_roll * 0.5
                    + base_orient_error_pitch * 0.25
            ) / 8.0
    )

    normalized_loco_error = (
            (
                    foot_pos_error_b / 2.0
                    + base_orient_error_pitch
                    + base_orient_error_roll
                    + base_height_error * 2.0
            ) / 5.0
    )

    mani_safety_scale = torch.exp(-normalized_mani_error / std ** 2)
    loco_safety_scale = torch.exp(-normalized_loco_error / std ** 2)

    env._mani_safety_scale = mani_safety_scale + 0.4
    env._loco_safety_scale = loco_safety_scale + 0.4

    return mani_safety_scale * 0.5 + loco_safety_scale * 0.5


def track_base_position_exp(
        env: ManagerBasedRlEnv,
        std: float,
        command_name: str = "base_pose",
) -> torch.Tensor:
    position_error = env.command_manager.get_term(command_name).metrics["est_position_error"]
    normal = torch.exp(-position_error / std ** 2)
    micro_enhancement = torch.exp(-5 * position_error / std ** 2)
    return (normal + micro_enhancement) * 0.5 * env._loco_safety_scale


def track_base_orientation_exp(
        env: ManagerBasedRlEnv,
        std: float,
        command_name: str = "base_pose",
) -> torch.Tensor:
    # Scale by estimated position error (what the robot observes) but keep orientation on true
    est_position_error = env.command_manager.get_term(command_name).metrics["est_position_error"]
    position_scale = torch.exp(-est_position_error / 0.5)
    orientation_error = env.command_manager.get_term(command_name).metrics["orientation_error"]
    normal = torch.exp(-orientation_error / std ** 2)
    micro_enhancement = torch.exp(-5 * orientation_error / std ** 2)
    return (normal + micro_enhancement) * position_scale * 0.5 * env._loco_safety_scale


def track_base_progress(env: ManagerBasedRlEnv, command_name: str = "base_pose") -> torch.Tensor:
    """Reward improvement over the best-achieved estimated position and true orientation errors."""
    cmd = env.command_manager.get_term(command_name)
    est_position_scale = torch.exp(-cmd.est_min_pos_error / 0.5)
    orient_scale = torch.exp(-cmd.min_orient_error / 0.5)
    return (2 * cmd.est_pos_improvement * est_position_scale + cmd.orient_improvement * orient_scale) * env._loco_safety_scale


def track_base_reference_exp(
        env: ManagerBasedRlEnv,
        std: float,
        delta: float = 0.5,
        command_name: str = "base_pose",
) -> torch.Tensor:
    est_position_error = env.command_manager.get_term(command_name).metrics["est_position_error"]
    orientation_error = env.command_manager.get_term(command_name).metrics["orientation_error"]
    se3_distance_ref = env.command_manager.get_term(command_name).se3_distance_ref_est
    track_error = torch.clamp(
        torch.abs(se3_distance_ref - orientation_error - 2 * est_position_error) - delta, min=0.0
    )
    return torch.exp(-track_error / std ** 2) * 0.5 * env._loco_safety_scale


def weighted_joint_torques_l2(
        env: ManagerBasedRlEnv,
        torque_weight: dict[str, float],
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    if not asset.data.is_actuated:
        return torch.zeros(env.num_envs, device=env.device)

    w = torch.zeros(asset.data.actuator_force.shape[1], device=env.device)
    for joint_name, weight in torque_weight.items():
        ids, _ = asset.find_joints(joint_name)
        w[ids] = weight

    return torch.sum(torch.square(asset.data.actuator_force) * w, dim=1)


def weighted_joint_power_l1(
        env: ManagerBasedRlEnv,
        power_weight: dict[str, float],
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    if not asset.data.is_actuated:
        return torch.zeros(env.num_envs, device=env.device)

    w = torch.zeros(asset.data.actuator_force.shape[1], device=env.device)
    for joint_name, weight in power_weight.items():
        ids, _ = asset.find_joints(joint_name)
        w[ids] = weight

    return torch.sum(torch.abs(asset.data.actuator_force * asset.data.joint_vel) * w, dim=1)
