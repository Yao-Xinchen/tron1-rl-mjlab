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
from mjlab.utils.noise import GaussianNoiseCfg

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


@dataclass
class ObservationCfg:
    @dataclass
    class CommandsObsCfg(ObsGroup):
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @dataclass
    class PolicyObsCfg(ObsGroup):
        # robot base measurements
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=GaussianNoiseCfg(mean=0.0, std=0.05),
            clip=(-100.0, 100.0), scale=0.25,
        )
        proj_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=GaussianNoiseCfg(mean=0.0, std=0.025),
            clip=(-100.0, 100.0), scale=1.0,
        )

        # robot joint measurements exclude wheel pos
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg(
                name="robot",
                joint_names=["abad_[RL]_Joint", "hip_[RL]_Joint", "knee_[RL]_Joint"]
            )},
            noise=GaussianNoiseCfg(mean=0.0, std=0.01), scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=GaussianNoiseCfg(mean=0.0, std=0.01),
            clip=(-100.0, 100.0), scale=0.05,
        )

        # last action
        last_action = ObsTerm(
            func=mdp.last_action,
            noise=GaussianNoiseCfg(mean=0.0, std=0.01),
            clip=(-100.0, 100.0), scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @dataclass
    class HistoryObsCfg(PolicyObsCfg):
        # same as PolicyObsCfg but with history
        def __post_init__(self):
            super().__post_init__()
            self.history_length = 20
            self.flatten_history_dim = False

    @dataclass
    class CriticObsCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0), scale=1.0, )
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100.0, 100.0), scale=0.25, )
        proj_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0), scale=1.0, )

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg(
                name="robot",
                joint_names=["abad_[RL]_Joint", "hip_[RL]_Joint", "knee_[RL]_Joint"]
            )}, scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            clip=(-100.0, 100.0), scale=0.05,
        )

        last_action = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0, )

        joint_torque = ObsTerm(func=mdp.actuator_force, scale=0.01)
        joint_acc = ObsTerm(func=mdp.joint_acc, scale=0.1)
        feet_lin_vel = ObsTerm(
            func=mdp.body_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="wheel_.*")}, scale=0.1
        )
        joint_stiffness = ObsTerm(func=mdp.joint_stiffness, scale=0.025)
        joint_damping = ObsTerm(func=mdp.joint_damping, scale=1.0)
        base_height_error = ObsTerm(func=mdp.base_height_error, scale=3.0)
        foot_rel_position_w = ObsTerm(func=mdp.foot_rel_position_w, scale=1.5)
        contact_force = ObsTerm(
            func=mdp.contact_forces,
            params={"asset_cfg": SceneEntityCfg("robot"), "sensor_name": "contact_sensors"},
            scale=0.001
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    commands = CommandsObsCfg()
    policy = PolicyObsCfg()
    history = HistoryObsCfg()
    critic = CriticObsCfg()
