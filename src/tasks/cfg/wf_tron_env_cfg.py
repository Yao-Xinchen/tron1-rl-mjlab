import math
from dataclasses import dataclass, field
import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    EventTermCfg as EventTerm,
    CurriculumTermCfg as CurrTerm,
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
        base_pose_commands = ObsTerm(func=mdp.base_commands_b)
        # base_pose_commands = ObsTerm(func=mdp.fake_base_commands_b)
        base_se3_decrease_rate = ObsTerm(func=mdp.base_se3_decrease_rate)
        base_commands_vel = ObsTerm(func=mdp.base_commands_vel_c)

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


@dataclass
class EventsCfg:
    # Startup events
    prepare_quantities = EventTerm(
        func=mdp.prepare_quantities,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_field,
        mode="startup",
        params={
            "field": "body_mass",
            "ranges": (-2.0, 5.0),
            "operation": "add",
            "distribution": "uniform",
            "asset_cfg": SceneEntityCfg("robot", body_names="base_Link"),
        },
    )
    add_link_mass = EventTerm(
        func=mdp.randomize_field,
        mode="startup",
        params={
            "field": "body_mass",
            "ranges": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_[LR]_Link"),
        },
    )
    robot_physics_material = EventTerm(
        func=mdp.randomize_field,
        mode="startup",
        params={
            "field": "geom_friction",
            "ranges": {
                0: (0.4, 1.2),  # Static friction
                1: (0.2, 0.9),  # Dynamic friction (torsional)
                2: (0.0, 1.0),  # Rolling friction
            },
            "operation": "abs",
            "distribution": "uniform",
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
        },
    )
    robot_center_of_mass = EventTerm(
        func=mdp.randomize_field,
        mode="startup",
        params={
            "field": "body_ipos",
            "ranges": {
                0: (-0.075, 0.075),  # X axis
                1: (-0.075, 0.075),  # Y axis
                2: (-0.075, 0.075),  # Z axis
            },
            "operation": "add",
            "distribution": "uniform",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Reset events
    reset_robot_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.5, 0.5),
        },
    )
    randomize_joint_stiffness = EventTerm(
        func=mdp.randomize_field,
        mode="reset",
        params={
            "field": "jnt_stiffness",
            "ranges": (0.5, 2.0),
            "operation": "scale",
            "distribution": "log_uniform",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )
    randomize_joint_damping = EventTerm(
        func=mdp.randomize_field,
        mode="reset",
        params={
            "field": "dof_damping",
            "ranges": (0.5, 2.0),
            "operation": "scale",
            "distribution": "log_uniform",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )

    # Interval events
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}}
    )


@dataclass
class RewardsCfg:
    # safety
    safety_exp = RewTerm(
        func=mdp.safety_reward_exp, weight=1.0, params={"base_height_target": 0.9, "std": math.sqrt(0.5)}
    )

    # tasks
    track_base_position_exp = RewTerm(
        func=mdp.track_base_position_exp,
        weight=2.0,
        params={
            "command_name": "base_pose",
            "std": math.sqrt(0.5),
        },
    )
    track_base_orientation_exp = RewTerm(
        func=mdp.track_base_orientation_exp,
        weight=3.0,
        params={
            "command_name": "base_pose",
            "std": math.sqrt(0.5),
        },
    )
    track_base_pb = RewTerm(func=mdp.track_base_pb, weight=15.0)
    track_base_reference_exp = RewTerm(
        func=mdp.track_base_reference_exp,
        weight=1.5,
        params={"std": math.sqrt(0.5), "init_value": 0.98},
    )

    # penalties
    dof_weighted_torques_l2 = RewTerm(
        func=mdp.weighted_joint_torques_l2,
        weight=-4.0e-5,
        params={
            "torque_weight": {
                "abad_L_Joint": 0.2,
                "hip_L_Joint": 0.2,
                "knee_L_Joint": 0.2,
                "abad_R_Joint": 0.2,
                "hip_R_Joint": 0.2,
                "knee_R_Joint": 0.2,
                "wheel_L_Joint": 8.0,
                "wheel_R_Joint": 8.0,
            }
        },
    )
    dof_weighted_power_l1 = RewTerm(
        func=mdp.weighted_joint_power_l1,
        weight=-2.5e-4,
        params={
            "power_weight": {
                "abad_L_Joint": 1.0,
                "hip_L_Joint": 1.0,
                "knee_L_Joint": 1.0,
                # "foot_L_Joint": 1.0,
                "abad_R_Joint": 1.0,
                "hip_R_Joint": 1.0,
                "knee_R_Joint": 1.0,
                # "foot_R_Joint": 1.0,
                "wheel_L_Joint": 2.0,
                "wheel_R_Joint": 2.0,
            }
        },
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.002)
    action_smoothness = RewTerm(func=mdp.action_smoothness_penalty, weight=-0.006)

    dof_vel_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=-0.0005, params={"asset_cfg": SceneEntityCfg("robot", joint_names="wheel_.+")}
    )
    dof_vel_non_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="(?!wheel_).*")},
    )
    dof_non_wheel_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="(?!wheel_).*")},
    )


@dataclass
class TerminationsCfg:
    """Termination terms for the MDP"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    bad_orientation = DoneTerm(
        func=mdp.bad_orientation_stochastic,
        params={
            "limit_angle": math.pi * 0.4,
            "probability": 0.1,
        },  # Expect step = 1 / probability
    )

    bad_height = DoneTerm(
        func=mdp.bad_height_stochastic,
        params={
            "limit_height": 0.5,
            "probability": 0.1,
        },  # Expect step = 1 / probability
    )


@dataclass
class CurriculumCfg:
    pos_commands_ranges_level = CurrTerm(
        func=mdp.pos_commands_ranges_level,
        params={
            "max_range": {
                # pos
                "pos_x": (-1.0, 1.0),
                "pos_y": (-1.0, 1.0),
                # vel
                "vel_x": (-1.0, 1.0),
                "vel_y": (-1.0, 1.0),
                "vel_yaw": (-2.0, 2.0),
            },
            "update_interval": 80 * 24,  # 80 iterations * 24 steps per iteration
            "command_name": "base_pose",
        },
    )


SIM_CFG = SimulationCfg(
    mujoco=MujocoCfg(
        timestep=0.005,
        iterations=1,
    ),
)


@dataclass
class WfTronEnvCfg(ManagerBasedRlEnvCfg):
    scene: SceneCfg = field(default_factory=lambda: SCENE_CFG)
    observations: ObservationCfg = field(default_factory=ObservationCfg)
    actions: ActionCfg = field(default_factory=ActionCfg)
    rewards: RewardsCfg = field(default_factory=RewardsCfg)
    events: EventsCfg = field(default_factory=EventsCfg)
    terminations: TerminationsCfg = field(default_factory=TerminationsCfg)
    curriculum: CurriculumCfg = field(default_factory=CurriculumCfg)
    sim: SimulationCfg = field(default_factory=lambda: SIM_CFG)
    viewer: ViewerConfig = field(default_factory=lambda: VIEWER_CONFIG)
    decimation: int = 4
    episode_length_s: float = 20.0
    seed: int = 0
