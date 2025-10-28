"""Reward functions for the task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.third_party.isaaclab.isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
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

    # Prepare variables
    base_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
    base_position = asset.data.root_link_pos_w.unsqueeze(1).expand(-1, 2, -1)

    # Compute the nominal foot error
    foot_position = asset.data.body_link_pos_w[:, env._wheels_link_ids, :]
    foot_position_b = quat_apply_inverse(base_quat, foot_position - base_position)
    base_height = asset.data.root_link_pos_w[:, 2] - foot_position[:, :, 2].mean(dim=-1) + env._foot_radius

    foot_pos_error_b = foot_position_b[:, :, :2] - env._nominal_foot_position_b[:, :2]

    # Abduction
    abduction = ((env._nominal_foot_position_b[:, 1] > 0.0) * (foot_pos_error_b[:, :, 1] < 0.0)) | (
            (env._nominal_foot_position_b[:, 1] < 0.0) * (foot_pos_error_b[:, :, 1] > 0.0)
    )

    foot_pos_error_b[:, :, 1] = torch.where(
        abduction, foot_pos_error_b[:, :, 1] / 0.1, foot_pos_error_b[:, :, 1] / 0.2
    )
    foot_pos_error_b[:, :, 0] = foot_pos_error_b[:, :, 0] / 0.2

    foot_pos_error_b = torch.sum(torch.sum(foot_pos_error_b.abs(), dim=-1), dim=-1)
    foot_pos_error_b = torch.clamp(foot_pos_error_b, max=8.0)

    # Compute base error
    base_orient_error_roll = torch.abs(asset.data.projected_gravity_b[:, 1]) / 0.1
    base_orient_error_pitch = torch.abs(asset.data.projected_gravity_b[:, 0]) / 0.85
    base_height_error = ((base_height - base_height_target) / 0.1) ** 2

    normalized_loco_error = (foot_pos_error_b / 2.0 + base_orient_error_pitch
                             + base_orient_error_roll + base_height_error * 2.0) / 5.0

    loco_safety_scale = torch.exp(-normalized_loco_error / std ** 2)

    env._loco_safety_scale = loco_safety_scale + 0.4

    return loco_safety_scale


def track_base_position_exp(
        env: ManagerBasedRlEnv,
        std: float,
        command_name: str = "base_pose",
) -> torch.Tensor:
    position_error = env.command_manager.get_term(command_name).metrics["position_error"]
    normal = torch.exp(-position_error / std ** 2)
    micro_enhancement = torch.exp(-5 * position_error / std ** 2)
    return (normal + micro_enhancement) * 0.5 * env._loco_safety_scale


def track_base_orientation_exp(
        env: ManagerBasedRlEnv,
        std: float,
        command_name: str = "base_pose",
) -> torch.Tensor:
    base_position_error = env.command_manager.get_term(command_name).metrics["position_error"]
    position_scale = torch.exp(-base_position_error / 0.5)
    base_orientation_error = env.command_manager.get_term(command_name).metrics["orientation_error"]
    normal = torch.exp(-base_orientation_error / std ** 2)
    micro_enhancement = torch.exp(-5 * base_orientation_error / std ** 2)
    return (normal + micro_enhancement) * position_scale * 0.5 * env._loco_safety_scale


def track_base_pb(env: ManagerBasedRlEnv, command_name: str = "base_pose") -> torch.Tensor:
    optim_pos_distance = env.command_manager.get_term(command_name).optim_pos_distance
    position_scale = torch.exp(-optim_pos_distance / 0.5)
    optim_orient_distance = env.command_manager.get_term(command_name).optim_orient_distance
    orient_scale = torch.exp(-optim_orient_distance / 0.5)
    pos_improve = env.command_manager.get_term(command_name).pos_improvement
    orient_improve = env.command_manager.get_term(command_name).orient_improvement
    return (2 * pos_improve * position_scale + orient_improve * orient_scale) * env._loco_safety_scale


def track_base_reference_exp(
        env: ManagerBasedRlEnv,
        std: float,
        delta: float = 0.5,
        command_name: str = "base_pose",
) -> torch.Tensor:
    base_position_error = env.command_manager.get_term(command_name).metrics["position_error"]
    base_orientation_error = env.command_manager.get_term(command_name).metrics["orientation_error"]
    se3_distance_ref = env.command_manager.get_term(command_name).se3_distance_ref
    track_error = torch.abs(se3_distance_ref - base_orientation_error - 2 * base_position_error) - delta
    track_error = torch.clamp(track_error, min=0.0)
    return torch.exp(-track_error / std ** 2) * 0.5 * env._loco_safety_scale


def joint_vel_l2(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def weighted_joint_torques_l2(
        env: ManagerBasedRlEnv,
        torque_weight: dict[str, float],
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    if not asset.data.is_actuated:
        return torch.zeros(env.num_envs, device=env.device)

    weighted_torque = torch.zeros_like(asset.data.actuator_force)

    for joint_name, w in torque_weight.items():
        joint_idx, _ = asset.find_joints(joint_name)
        weighted_torque[:, joint_idx] = torch.square(asset.data.actuator_force[:, joint_idx]) * w

    return torch.sum(weighted_torque, dim=1)


def weighted_joint_power_l1(
        env: ManagerBasedRlEnv,
        power_weight: dict[str, float],
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    if not asset.data.is_actuated:
        return torch.zeros(env.num_envs, device=env.device)

    weighted_power = torch.zeros_like(asset.data.actuator_force)

    for joint_name, w in power_weight.items():
        joint_idx, _ = asset.find_joints(joint_name)
        # power = force * velocity
        weighted_power[:, joint_idx] = (
                torch.abs(asset.data.actuator_force[:, joint_idx] * asset.data.joint_vel[:, joint_idx]) * w
        )

    return torch.sum(weighted_power, dim=1)


class ActionSmoothnessPenaltyWrapper:
    def __init__(self):
        self.prev_prev_action = None
        self.prev_action = None
        self.__name__ = "action_smoothness_penalty"

    def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
        """Penalize large instantaneous changes in the network action output"""
        current_action = env.action_manager.action.clone()

        if self.prev_action is None:
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        if self.prev_prev_action is None:
            self.prev_prev_action = self.prev_action
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        penalty = torch.sum(torch.square(current_action - 2 * self.prev_action + self.prev_prev_action), dim=1)

        # Update actions for next call
        self.prev_prev_action = self.prev_action
        self.prev_action = current_action

        startup_env_musk = env.episode_length_buf < 3
        penalty[startup_env_musk] = 0

        return penalty


action_smoothness_penalty = ActionSmoothnessPenaltyWrapper()
