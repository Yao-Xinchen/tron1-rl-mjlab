"""Custom command terms for the task."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
    combine_frame_transforms,
    compute_pose_error,
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    quat_unique,
    sample_uniform,
)

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.viewer.debug_visualizer import DebugVisualizer


class UniformPoseCommand(CommandTerm):
    cfg: UniformPoseCommandCfg

    def __init__(self, cfg: UniformPoseCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        self.robot: Entity = env.scene[cfg.entity_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        # pose command (x, y, z, qw, qx, qy, qz) in root frame
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)

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
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_link_pos_w,
            self.robot.data.root_link_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_link_pos_w[:, self.body_idx],
            self.robot.data.body_link_quat_w[:, self.body_idx],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        r = torch.empty(len(env_ids), device=self.device)
        self.pose_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        self.pose_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        self.pose_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)
        euler_angles = torch.zeros(len(env_ids), 3, device=self.device)
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        self.pose_command_b[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat

    def _update_command(self) -> None:
        pass


@dataclass(kw_only=True)
class UniformPoseCommandCfg(CommandTermCfg):
    entity_name: str
    body_name: str
    make_quat_unique: bool = False

    @dataclass
    class Ranges:
        pos_x: tuple[float, float]
        pos_y: tuple[float, float]
        pos_z: tuple[float, float]
        roll: tuple[float, float]
        pitch: tuple[float, float]
        yaw: tuple[float, float]

    ranges: Ranges

    def build(self, env: ManagerBasedRlEnv) -> UniformPoseCommand:
        return UniformPoseCommand(self, env)


class UniformWorldPoseCommand(UniformPoseCommand):
    """Pose command sampled in the world frame, with optional moving target velocity."""

    cfg: UniformWorldPoseCommandCfg

    def __init__(self, cfg: UniformWorldPoseCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        self._has_vel_commands = hasattr(cfg.ranges, 'vel_x')
        self._decay_rate_range = cfg.se3_decay_rate_range
        self.resampling_time_scale = cfg.resampling_time_scale
        self.resample_time_range = cfg.resampling_time_range

        # Rate at which se3_distance_ref decreases per second (sampled per episode)
        self.se3_decay_rate = torch.zeros(self.num_envs, device=self.device)
        # SE(3) reference distance that decays from its initial value toward zero
        self.se3_distance_ref = torch.ones(self.num_envs, device=self.device)

        # Best (minimum) errors achieved so far this episode
        self.min_pos_error = torch.zeros(self.num_envs, device=self.device)
        self.min_orient_error = torch.zeros(self.num_envs, device=self.device)
        # Per-step improvement over the episode best
        self.pos_improvement = torch.zeros(self.num_envs, device=self.device)
        self.orient_improvement = torch.zeros(self.num_envs, device=self.device)

        # Linear and yaw velocity commands expressed in the target pose frame
        self.vel_command_t = torch.zeros(self.num_envs, 3, device=self.device)

    def _refresh_errors(self) -> None:
        """Recompute pose_command_b and error metrics from the current world command."""
        self.pose_command_b[:, :3] = quat_apply_inverse(
            self.robot.data.root_link_quat_w,
            self.pose_command_w[:, :3] - self.robot.data.root_link_pos_w,
        )
        self.pose_command_b[:, 3:] = quat_unique(
            quat_mul(quat_inv(self.robot.data.root_link_quat_w), self.pose_command_w[:, 3:])
        )
        pos_error = self.pose_command_w[:, :3] - self.robot.data.root_link_pos_w
        rot_error = quat_error_magnitude(
            quat_unique(self.robot.data.root_link_quat_w), self.pose_command_w[:, 3:]
        )
        self.metrics["position_error"] = torch.norm(pos_error[:, :2], dim=-1)
        self.metrics["orientation_error"] = rot_error

    def _update_metrics(self) -> None:
        self._refresh_errors()
        self.se3_distance_ref = torch.clamp(
            self.se3_distance_ref - self.se3_decay_rate * self._env.step_dt, min=0.0
        )
        self.pos_improvement = (self.min_pos_error - self.metrics["position_error"]).clip(min=0.0)
        self.orient_improvement = (self.min_orient_error - self.metrics["orientation_error"]).clip(min=0.0)
        self.min_pos_error[:] = torch.minimum(self.metrics["position_error"], self.min_pos_error)
        self.min_orient_error[:] = torch.minimum(self.metrics["orientation_error"], self.min_orient_error)

    def _update_se3_ref(self, env_ids: Sequence[int]) -> None:
        """Initialize SE(3) reference distance and episode-best errors for reset environments."""
        self._refresh_errors()
        pos_err = self.metrics["position_error"][env_ids]
        ori_err = self.metrics["orientation_error"][env_ids]
        self.se3_distance_ref[env_ids] = 2 * pos_err + ori_err
        self.min_pos_error[env_ids] = pos_err
        self.min_orient_error[env_ids] = ori_err
        self.pos_improvement[env_ids] = 0.0
        self.orient_improvement[env_ids] = 0.0

    def _update_command(self) -> None:
        if not self._has_vel_commands:
            return
        dt = self._env.step_dt
        vel_t_3d = torch.cat(
            [self.vel_command_t[:, :2], torch.zeros(self.num_envs, 1, device=self.device)], dim=-1
        )
        vel_w = quat_apply(self.pose_command_w[:, 3:], vel_t_3d)
        self.pose_command_w[:, :2] += vel_w[:, :2] * dt
        delta_yaw = self.vel_command_t[:, 2] * dt
        delta_quat = quat_from_euler_xyz(
            torch.zeros_like(delta_yaw), torch.zeros_like(delta_yaw), delta_yaw
        )
        self.pose_command_w[:, 3:] = quat_unique(quat_mul(self.pose_command_w[:, 3:], delta_quat))

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        r = torch.empty(len(env_ids), device=self.device)
        self.pose_command_w[env_ids, 0] = self.robot.data.root_link_pos_w[env_ids, 0] + r.uniform_(
            *self.cfg.ranges.pos_x)
        self.pose_command_w[env_ids, 1] = self.robot.data.root_link_pos_w[env_ids, 1] + r.uniform_(
            *self.cfg.ranges.pos_y)
        self.pose_command_w[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)
        euler_angles = torch.zeros(len(env_ids), 3, device=self.device)
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        self.pose_command_w[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat
        self.se3_decay_rate[env_ids] = sample_uniform(
            *self._decay_rate_range, len(env_ids), device=self.device
        )
        if self._has_vel_commands:
            self.vel_command_t[env_ids, 0] = r.uniform_(*self.cfg.ranges.vel_x)
            self.vel_command_t[env_ids, 1] = r.uniform_(*self.cfg.ranges.vel_y)
            self.vel_command_t[env_ids, 2] = r.uniform_(*self.cfg.ranges.vel_yaw)

    def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
        for env_idx in visualizer.get_env_indices(self.num_envs):
            target_pos = self.pose_command_w[env_idx, :3]
            target_quat = self.pose_command_w[env_idx, 3:]
            rot_mat = matrix_from_quat(target_quat.unsqueeze(0)).squeeze(0)
            visualizer.add_frame(position=target_pos, rotation_matrix=rot_mat, scale=0.3, label=f"cmd_{env_idx}")
            visualizer.add_sphere(center=target_pos, radius=0.04, color=(1.0, 0.5, 0.0, 0.8),
                                  label=f"cmd_sphere_{env_idx}")

    def _resample(self, env_ids: Sequence[int]) -> None:
        if len(env_ids) == 0:
            return
        self._resample_command(env_ids)
        self._update_se3_ref(env_ids)
        random_scale = sample_uniform(
            self.resampling_time_scale[0], self.resampling_time_scale[1], len(env_ids), device=self.device
        )
        self.time_left[env_ids] = (self.se3_distance_ref[env_ids] * random_scale).clip(
            min=self.resample_time_range[0], max=self.resample_time_range[1]
        )
        self.command_counter[env_ids] += 1


@dataclass(kw_only=True)
class UniformWorldPoseCommandCfg(UniformPoseCommandCfg):
    se3_decay_rate_range: tuple[float, float] = (0.5, 1.4)
    resampling_time_scale: tuple[float, float] = (6.0, 15.0)

    @dataclass
    class Ranges:
        pos_x: tuple[float, float]
        pos_y: tuple[float, float]
        vel_x: tuple[float, float]
        vel_y: tuple[float, float]
        vel_yaw: tuple[float, float]

        # Fixed values — not user-configurable
        pos_z: tuple[float, float] = field(default=(0.7, 1.1), init=False)
        roll: tuple[float, float] = field(default=(0.0, 0.0), init=False)
        pitch: tuple[float, float] = field(default=(0.0, 0.0), init=False)
        yaw: tuple[float, float] = field(default=(-3.14, 3.14), init=False)

    ranges: Ranges

    def build(self, env: ManagerBasedRlEnv) -> UniformWorldPoseCommand:
        return UniformWorldPoseCommand(self, env)
