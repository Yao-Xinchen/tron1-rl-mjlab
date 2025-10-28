from pathlib import Path
import mujoco

from mjlab.entity import Entity, EntityCfg, EntityArticulationInfoCfg
from mjlab.utils.spec_config import ActuatorCfg, ContactSensorCfg

current_dir: Path = Path(__file__).parent.resolve()

WF_TRON_XML: Path = current_dir / "xml" / "robot.xml"
assert WF_TRON_XML.exists(), f"XML file not found: {WF_TRON_XML}"


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(WF_TRON_XML))


WF_TRON_LEG_ACTUATORS = ActuatorCfg(
    joint_names_expr=["abad_[RL]_Joint", "hip_[RL]_Joint", "knee_[RL]_Joint"],
    effort_limit=80.0,
    stiffness=40.0,
    damping=1.8,
)

WF_TRON_WHEEL_ACTUATORS = ActuatorCfg(
    joint_names_expr=["wheel_[RL]_Joint"],
    effort_limit=40.0,
    stiffness=0.0,
    damping=0.5,
)

WF_TRON_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        WF_TRON_LEG_ACTUATORS,
        WF_TRON_WHEEL_ACTUATORS,
    ),
)

WF_TRON_CONTACT_SENSOR = ContactSensorCfg(
    name="contact_sensors",
    subtree1="base_Link",
    data=("found", "force"),
    reduce="netforce",
    num=10,
)

WF_TRON_ROBOT_CFG = EntityCfg(
    spec_fn=get_spec,
    articulation=WF_TRON_ARTICULATION,
    sensors=(WF_TRON_CONTACT_SENSOR,),
)

if __name__ == "__main__":
    import mujoco.viewer as viewer

    robot = Entity(WF_TRON_ROBOT_CFG)
    viewer.launch(robot.spec.compile())
