"""Custom command terms for the task."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.manager_term_config import CommandTermCfg
from mjlab.third_party.isaaclab.isaaclab.utils.math import (
    combine_frame_transforms,
    compute_pose_error,
    quat_apply,
    quat_apply_inverse,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    quat_unique,
    sample_uniform,
)

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


# UniformPoseCommand from IsaacLab

class UniformPoseCommand(CommandTerm):
    cfg: UniformPoseCommandCfg

    def __init__(self, cfg: UniformPoseCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        self.robot: Entity = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        # Create buffers: commands (x, y, z, qw, qx, qy, qz) in root frame
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)

        # Metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "UniformPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        return self.pose_command_b

    def _update_metrics(self) -> None:
        # Transform command from base frame to simulation world frame
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_link_pos_w,
            self.robot.data.root_link_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        # Compute the error
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_link_pos_w[:, self.body_idx],
            self.robot.data.body_link_quat_w[:, self.body_idx],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        # Sample new pose targets
        r = torch.empty(len(env_ids), device=self.device)
        # Position
        self.pose_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        self.pose_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        self.pose_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)
        # Orientation
        euler_angles = torch.zeros_like(self.pose_command_b[env_ids, :3])
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        # Make sure the quaternion has real part as positive if configured
        self.pose_command_b[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat

    def _update_command(self) -> None:
        pass


@dataclass(kw_only=True)
class UniformPoseCommandCfg(CommandTermCfg):
    asset_name: str
    body_name: str
    make_quat_unique: bool = False
    class_type: type[CommandTerm] = UniformPoseCommand

    @dataclass
    class Ranges:
        pos_x: tuple[float, float]
        pos_y: tuple[float, float]
        pos_z: tuple[float, float]
        roll: tuple[float, float]
        pitch: tuple[float, float]
        yaw: tuple[float, float]

    ranges: Ranges


# custom UniformWorldPoseCommand

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        [
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ],
        dim=-1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def compute_rotation_distance(input_quat, target_quat):
    Ee_target_R = quaternion_to_matrix(target_quat)
    Ee_R = quaternion_to_matrix(input_quat)

    R_rel = torch.matmul(torch.transpose(Ee_target_R, 1, 2), Ee_R)
    trace_R_rel = torch.einsum("bii->b", R_rel)

    trace_clamped = torch.clamp((trace_R_rel - 1) / 2, -1.0, 1.0)
    rotation_distance = torch.acos(trace_clamped)
    return rotation_distance


class UniformWorldPoseCommand(UniformPoseCommand):
    cfg: UniformWorldPoseCommandCfg

    def __init__(self, cfg: UniformWorldPoseCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.decrease_vel = torch.zeros(self.num_envs, device=self.device)
        self.se3_distance_ref = torch.ones(self.num_envs, device=self.device)
        self.decrease_vel_range = cfg.se3_decrease_vel_range
        self.resampling_time_scale = cfg.resampling_time_scale
        self.resample_time_range = cfg.resampling_time_range

        self.optim_pos_distance = torch.zeros(self.num_envs, device=self.device)
        self.optim_orient_distance = torch.zeros(self.num_envs, device=self.device)
        self.pos_improvement = torch.zeros(self.num_envs, device=self.device)
        self.orient_improvement = torch.zeros(self.num_envs, device=self.device)

        # Velocity commands: sampled in target pose frame, transformed to body frame for observation
        self.pose_command_vel_c = torch.zeros(
            self.num_envs, 3, device=self.device
        )  # [vel_x, vel_y, vel_yaw] in target frame

    def _update_metrics(self):
        # refresh the pose_command_b
        self.pose_command_b[:, :3] = quat_apply_inverse(
            self.robot.data.root_link_quat_w, self.pose_command_w[:, :3] - self.robot.data.root_link_pos_w
        )
        self.pose_command_b[:, 3:] = quat_unique(
            quat_mul(quat_inv(self.robot.data.root_link_quat_w), self.pose_command_w[:, 3:])
        )

        # compute the error
        pos_error = self.pose_command_w[:, :3] - self.robot.data.root_link_pos_w
        rot_error_angle = compute_rotation_distance(
            quat_unique(self.robot.data.root_link_quat_w), self.pose_command_w[:, 3:]
        )

        self.metrics["position_error"] = torch.norm(pos_error[:, :2], dim=-1)
        self.metrics["orientation_error"] = rot_error_angle
        self.se3_distance_ref -= self.decrease_vel * self._env.step_dt
        self.se3_distance_ref = torch.clamp(self.se3_distance_ref, min=0.0)
        self.pos_improvement = (self.optim_pos_distance - self.metrics["position_error"]).clip(min=0.0)
        self.orient_improvement = (self.optim_orient_distance - self.metrics["orientation_error"]).clip(min=0.0)
        self.optim_pos_distance[:] = torch.minimum(self.metrics["position_error"], self.optim_pos_distance)
        self.optim_orient_distance[:] = torch.minimum(self.metrics["orientation_error"], self.optim_orient_distance)

    def _update_se3_ref(self, env_ids: Sequence[int]):
        self.pose_command_b[:, :3] = quat_apply_inverse(
            self.robot.data.root_link_quat_w, self.pose_command_w[:, :3] - self.robot.data.root_link_pos_w
        )
        self.pose_command_b[:, 3:] = quat_unique(
            quat_mul(quat_inv(self.robot.data.root_link_quat_w), self.pose_command_w[:, 3:])
        )

        # compute the error
        pos_error = self.pose_command_w[:, :3] - self.robot.data.root_link_pos_w
        rot_error_angle = compute_rotation_distance(
            quat_unique(self.robot.data.root_link_quat_w), self.pose_command_w[:, 3:7]
        )

        self.metrics["position_error"] = torch.norm(pos_error[:, :2], dim=-1)
        self.metrics["orientation_error"] = rot_error_angle

        self.se3_distance_ref[env_ids] = (
                2 * self.metrics["position_error"][env_ids] + self.metrics["orientation_error"][env_ids]
        )
        self.optim_pos_distance[env_ids] = self.metrics["position_error"][env_ids]
        self.optim_orient_distance[env_ids] = self.metrics["orientation_error"][env_ids]

        self.pos_improvement[env_ids] = 0.0
        self.orient_improvement[env_ids] = 0.0

    def _update_command(self):
        # Get timestep
        dt = self._env.step_dt

        # Only update if velocities are configured
        if (hasattr(self.cfg.ranges, 'vel_x') and hasattr(self.cfg.ranges, 'vel_y')
                and hasattr(self.cfg.ranges, 'vel_yaw')):
            # Transform linear velocities from target frame to world frame
            # Create 3D velocity vector [vel_x, vel_y, 0] in target frame
            vel_t_3d = torch.cat([
                self.pose_command_vel_c[:, :2],
                torch.zeros(self.num_envs, 1, device=self.device)
            ], dim=-1)

            # Rotate by target orientation to get world frame velocity
            vel_w_3d = quat_apply(self.pose_command_w[:, 3:], vel_t_3d)

            # Update target position (only x and y, keep z constant)
            self.pose_command_w[:, :2] += vel_w_3d[:, :2] * dt

            # Update target orientation based on yaw velocity
            # Create incremental rotation quaternion around z-axis
            delta_yaw = self.pose_command_vel_c[:, 2] * dt
            delta_quat = quat_from_euler_xyz(
                torch.zeros_like(delta_yaw),
                torch.zeros_like(delta_yaw),
                delta_yaw
            )

            # Apply rotation: new_quat = current_quat * delta_quat
            self.pose_command_w[:, 3:] = quat_mul(self.pose_command_w[:, 3:], delta_quat)

            # Normalize and make unique
            self.pose_command_w[:, 3:] = quat_unique(self.pose_command_w[:, 3:])

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new pose targets
        # -- position
        r = torch.empty(len(env_ids), device=self.device)
        self.pose_command_w[env_ids, 0] = self.robot.data.root_link_pos_w[env_ids, 0] + r.uniform_(
            *self.cfg.ranges.pos_x
        )
        self.pose_command_w[env_ids, 1] = self.robot.data.root_link_pos_w[env_ids, 1] + r.uniform_(
            *self.cfg.ranges.pos_y
        )
        self.pose_command_w[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)
        # -- orientation
        euler_angles = torch.zeros_like(self.pose_command_w[env_ids, :3])
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        # make sure the quaternion has real part as positive
        self.pose_command_w[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat
        self.decrease_vel[env_ids] = sample_uniform(
            self.decrease_vel_range[0], self.decrease_vel_range[1], len(env_ids), device=self.device
        )
        # -- velocities (sampled in target pose frame)
        if (hasattr(self.cfg.ranges, 'vel_x') and hasattr(self.cfg.ranges, 'vel_y')
                and hasattr(self.cfg.ranges, 'vel_yaw')):
            self.pose_command_vel_c[env_ids, 0] = r.uniform_(*self.cfg.ranges.vel_x)
            self.pose_command_vel_c[env_ids, 1] = r.uniform_(*self.cfg.ranges.vel_y)
            self.pose_command_vel_c[env_ids, 2] = r.uniform_(*self.cfg.ranges.vel_yaw)

    def _resample(self, env_ids):
        # resample the time left before resampling
        if len(env_ids) != 0:
            # self.metrics["contact_time"][env_ids] = 0.0
            self._resample_command(env_ids)
            self._update_se3_ref(env_ids)
            # self._update_metrics()
            se3_error = 2 * self.metrics["position_error"][env_ids] + self.metrics["orientation_error"][env_ids]
            random_scale = sample_uniform(
                self.resampling_time_scale[0], self.resampling_time_scale[1], len(env_ids), device=self.device
            )
            self.time_left[env_ids] = (se3_error * random_scale).clip(
                min=self.resample_time_range[0], max=self.resample_time_range[1]
            )
            # increment the command counter
            self.command_counter[env_ids] += 1


@dataclass(kw_only=True)
class UniformWorldPoseCommandCfg(UniformPoseCommandCfg):
    se3_decrease_vel_range: tuple[float, float] = (0.5, 1.4)
    resampling_time_scale: tuple[float, float] = (6.0, 15.0)
    class_type: type[CommandTerm] = UniformWorldPoseCommand

    @dataclass
    class Ranges:
        pos_x: tuple[float, float]
        pos_y: tuple[float, float]
        vel_x: tuple[float, float]
        vel_y: tuple[float, float]
        vel_yaw: tuple[float, float]

        # Fixed values (not configurable)
        pos_z: tuple[float, float] = field(default=(0.7, 1.1), init=False)
        roll: tuple[float, float] = field(default=(0.0, 0.0), init=False)
        pitch: tuple[float, float] = field(default=(0.0, 0.0), init=False)
        yaw: tuple[float, float] = field(default=(-3.14, 3.14), init=False)

    ranges: Ranges
