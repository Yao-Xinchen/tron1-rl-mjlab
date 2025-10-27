"""Custom action terms for the task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions.actions_config import JointActionCfg
from mjlab.envs.mdp.actions.joint_actions import JointAction
from mjlab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from mjlab.envs.manager_based_env import ManagerBasedEnv


class JointVelocityAction(JointAction):
    def __init__(self, cfg: JointVelocityActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg=cfg, env=env)

        if cfg.use_default_offset:
            self._offset = torch.zeros_like(
                self._asset.data.default_joint_pos[:, self._joint_ids]
            )

    def apply_actions(self):
        # Use write_ctrl to send control signals to actuators
        self._asset.data.write_ctrl(
            self._processed_actions, self._actuator_ids
        )


@dataclass(kw_only=True)
class JointVelocityActionCfg(JointActionCfg):
    class_type: type[ActionTerm] = JointVelocityAction
    use_default_offset: bool = False
