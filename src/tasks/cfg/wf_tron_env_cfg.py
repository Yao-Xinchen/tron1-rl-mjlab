import math
from dataclasses import dataclass, field
import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewardTerm,
    TerminationTermCfg as DoneTerm,
    EventTermCfg as EventTerm,
    term,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig
from mjlab.rl import RslRlOnPolicyRunnerCfg

from ...assets import WF_TRON_ROBOT_CFG
from .terrain_cfg import TERRAINS_IMPORTER_CFG
from .. import mdp

SCENE_CFG = SceneCfg(
    num_envs=4096,
    extent=1.0,
    terrain=TERRAINS_IMPORTER_CFG,
    entities={"robot": WF_TRON_ROBOT_CFG},
)

VIEWER_CONFIG = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    asset_name="robot",
    body_name="base_Link",
    distance=3.0,
    elevation=10.0,
    azimuth=90.0,
)


@dataclass
class ActionCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", actuator_names=["abad_[RL]_Joint", "hip_[RL]_Joint", "knee_[RL]_Joint"],
        scale=0.5, use_default_offset=True
    )
    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot", actuator_names=["wheel_[RL]_Joint"],
        scale=5.0, use_default_offset=True
    )
