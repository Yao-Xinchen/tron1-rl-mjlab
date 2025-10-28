"""Curriculum functions for the task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .commands import UniformWorldPoseCommandCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def pos_commands_ranges_level(
        env: ManagerBasedRlEnv,
        max_range: dict[str, tuple[float, float]],
        update_interval: int = 80 * 24,
        command_name: str = "base_pose",
) -> torch.Tensor:
    command_cfg: UniformWorldPoseCommandCfg = env.command_manager.get_term(command_name).cfg
    x = command_cfg.ranges.pos_x[1]
    if (env.common_step_counter + 1) % update_interval == 0:
        # Update position ranges
        x = command_cfg.ranges.pos_x[1] + 0.1
        y = command_cfg.ranges.pos_y[1] + 0.1
        x = min(x, max_range["pos_x"][1])
        y = min(y, max_range["pos_y"][1])
        command_cfg.ranges.pos_x = (-x, x)
        command_cfg.ranges.pos_y = (-y, y)

        # Update velocity ranges if they exist
        if hasattr(command_cfg.ranges, 'vel_x') and "vel_x" in max_range:
            vel_x = command_cfg.ranges.vel_x[1] + 0.05
            vel_y = command_cfg.ranges.vel_y[1] + 0.05
            vel_yaw = command_cfg.ranges.vel_yaw[1] + 0.1
            vel_x = min(vel_x, max_range["vel_x"][1])
            vel_y = min(vel_y, max_range["vel_y"][1])
            vel_yaw = min(vel_yaw, max_range["vel_yaw"][1])
            command_cfg.ranges.vel_x = (-vel_x, vel_x)
            command_cfg.ranges.vel_y = (-vel_y, vel_y)
            command_cfg.ranges.vel_yaw = (-vel_yaw, vel_yaw)

    # return the mean terrain level
    return torch.ones(1, dtype=torch.float) * x
