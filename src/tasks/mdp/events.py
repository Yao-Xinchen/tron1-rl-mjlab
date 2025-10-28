"""Event functions for the task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.third_party.isaaclab.isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from mjlab.envs.manager_based_env import ManagerBasedEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def prepare_quantities(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Compute the nominal foot position in the body frame.

    This function computes the nominal foot position in the body frame. This function is only suitable for TRON robot.

    The computed nominal foot position is stored in the following attributes of env:
        - env._nominal_foot_position_b: Nominal foot positions in body frame
        - env._wheels_link_ids: Body indices of wheel links
        - env._wheels_joint_ids: Joint indices of wheel joints
        - env._foot_radius: Radius of the foot/wheel (0.127m)
    """
    asset: Entity = env.scene[asset_cfg.name]

    wheel_link_idx, _ = asset.find_bodies("wheel_[RL]_Link")
    wheel_joint_ids, _ = asset.find_joints("wheel_[RL]_Joint")
    base_idx, _ = asset.find_bodies("base_Link")

    wheels_pos_w = asset.data.body_link_pos_w[:, wheel_link_idx, :]
    base_pos_w = asset.data.body_link_pos_w[:, base_idx, :]
    base_quat = asset.data.body_link_quat_w[:, base_idx, :]

    nominal_foot_position_b = torch.zeros(len(wheel_link_idx), 3, device=env.device)

    for j in range(env.num_envs):
        if torch.any(asset.data.joint_pos[j, :] > 5e-2):
            continue
        for i in range(len(wheel_link_idx)):
            nominal_foot_position_b[i, :] = quat_apply_inverse(
                base_quat[j, 0, :], wheels_pos_w[j, i, :] - base_pos_w[j, 0, :]
            )
        break

    assert (nominal_foot_position_b != 0.0).any(), "Failed to compute nominal foot positions"

    env._nominal_foot_position_b = nominal_foot_position_b  # type: ignore
    env._wheels_link_ids = wheel_link_idx  # type: ignore
    env._wheels_joint_ids = wheel_joint_ids  # type: ignore
    env._foot_radius = 0.127  # type: ignore