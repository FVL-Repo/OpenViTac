from tacex_assets.robots.franka.franka_gsmini_gripper_uipc_high_res import (
    FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG
)
from tacex_assets.robots.franka.franka_xensews_gripper_uipc import (
    FRANKA_PANDA_ARM_XENSEWS_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG
)
import math

from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from ..sensors.tactile import TactileCfg, create_tactile_cfg

@configclass
class RobotCfg:
    robot: ArticulationCfg = None
    tactiles: list[TactileCfg] = []

    gripper_offset: float = 0.131 # in m
    gripper_max_qpos: float = 0.039 # in m
    gripper_open_qpos: float | None = None
    gripper_close_qpos: float | None = None

    tactile_far_plane: float = 30.0 # in mm
    adaptive_grasp_depth_threshold: float = 27.5 # in mm, used for grasping
    contact_threshold: tuple[float, float] = (27.5, 28.0) # in mm, used in `gravity_rotate` api

def _use_dense_gelpad(robot: ArticulationCfg) -> ArticulationCfg:
    from pathlib import Path

    dense_usd = (
        Path(__file__).resolve().parents[2]
        / "third_party/TacEx/source/tacex_assets/tacex_assets/data/Robots/Franka/"
        "GelSight_Mini/Gripper/uipc_gelpads_dense_wrist.usd"
    )
    if not dense_usd.exists():
        raise FileNotFoundError(
            f"dense gelpad USD not found: {dense_usd}\n"
            "Generate it once before setting dense_gelpad: true."
        )
    return robot.replace(spawn=robot.spawn.replace(usd_path=str(dense_usd)))


def create_franka_gsmini_gripper(data_type:list[str], dense_gelpad: bool = False):
    robot = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": -0.010809095,
                "panda_joint2": 0.096037410,
                "panda_joint3": 0.000734462,
                "panda_joint4": -2.433035851,
                "panda_joint5": 0.035354517,
                "panda_joint6": 2.500859022,
                "panda_joint7": 0.741,
                "panda_finger.*": 0.02,
            }
        ),
    )
    if dense_gelpad:
        robot = _use_dense_gelpad(robot)
    tactiles = [
        create_tactile_cfg(
            prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_left",
            gelpad_prim_path="/World/envs/env_.*/Robot/gelpad_left",
            gelpad_attachment_body_name="gelsight_mini_case_left",
            name="left_tactile",
            sensor_type="gsmini",
            data_type=data_type,
            dense=dense_gelpad,
        ),
        create_tactile_cfg(
            prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_right",
            gelpad_prim_path="/World/envs/env_.*/Robot/gelpad_right",
            gelpad_attachment_body_name="gelsight_mini_case_right",
            name="right_tactile",
            sensor_type="gsmini",
            data_type=data_type,
            dense=dense_gelpad,
        )
    ]
    return RobotCfg(
        robot=robot,
        tactiles=tactiles,
        gripper_offset=0.131,
        gripper_max_qpos=0.039,
        tactile_far_plane=34.0,
        adaptive_grasp_depth_threshold=27.5,
        contact_threshold=(27.5, 28.0)
    )



def create_franka_neote_gripper(data_type:list[str], dense_gelpad: bool = False):
    """Neote currently reuses the GelSight Mini physical assembly.

    The Neote branch differs in the requested tactile image modalities
    (marker/force/particle rendering), not in robot USD, gelpad attachment,
    or tactile camera geometry.
    """
    robot = FRANKA_PANDA_ARM_GSMINI_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": -0.010809095,
                "panda_joint2": 0.096037410,
                "panda_joint3": 0.000734462,
                "panda_joint4": -2.433035851,
                "panda_joint5": 0.035354517,
                "panda_joint6": 2.500859022,
                "panda_joint7": 0.741,
                "panda_finger.*": 0.02,
            }
        ),
    )
    if dense_gelpad:
        robot = _use_dense_gelpad(robot)
    tactiles = [
        create_tactile_cfg(
            prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_left",
            gelpad_prim_path="/World/envs/env_.*/Robot/gelpad_left",
            gelpad_attachment_body_name="gelsight_mini_case_left",
            name="left_tactile",
            sensor_type="neote",
            data_type=data_type,
            dense=dense_gelpad,
        ),
        create_tactile_cfg(
            prim_path="/World/envs/env_.*/Robot/gelsight_mini_case_right",
            gelpad_prim_path="/World/envs/env_.*/Robot/gelpad_right",
            gelpad_attachment_body_name="gelsight_mini_case_right",
            name="right_tactile",
            sensor_type="neote",
            data_type=data_type,
            dense=dense_gelpad,
        )
    ]
    return RobotCfg(
        robot=robot,
        tactiles=tactiles,
        gripper_offset=0.131,
        gripper_max_qpos=0.039,
        tactile_far_plane=34.0,
        adaptive_grasp_depth_threshold=27.5,
        contact_threshold=(27.5, 28.0)
    )


def create_franka_xensews_gripper(data_type:list[str]):
    # Match the standalone gripper demo: finger_joint=0 opens the gripper,
    # and +45 degrees is the maximum intended close pose.
    robotiq_open_qpos = 0.0
    robotiq_close_qpos = math.radians(45.0)

    robot = FRANKA_PANDA_ARM_XENSEWS_GRIPPER_HIGH_PD_HIGH_RES_UIPC_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": 0.0,
                "panda_joint3": 0.0,
                "panda_joint4": -2.46,
                "panda_joint5": 0.0,
                "panda_joint6": 2.5,
                "panda_joint7": 0.741,
                "finger_joint": robotiq_open_qpos,
            }
        ),
    )
    # Keep the visual/UIPC gelpad bound to the same named XenseWS link. Cross-binding
    # makes the pads move apart when the gripper closes.
    tactiles = [
        create_tactile_cfg(
            prim_path="/World/envs/env_.*/Robot/root_joint/XenseWS_left",
            gelpad_prim_path="/World/envs/env_.*/Robot/XenseWS_gelpad_left",
            gelpad_attachment_body_name="XenseWS_left",
            gelpad_attachment_prim_path="/World/envs/env_.*/Robot/root_joint/XenseWS_left",
            name="left_tactile",
            sensor_type="xensews",
            data_type=data_type,
        ),
        create_tactile_cfg(
            prim_path="/World/envs/env_.*/Robot/root_joint/XenseWS_right",
            gelpad_prim_path="/World/envs/env_.*/Robot/XenseWS_gelpad_right",
            gelpad_attachment_body_name="XenseWS_right",
            gelpad_attachment_prim_path="/World/envs/env_.*/Robot/root_joint/XenseWS_right",
            name="right_tactile",
            sensor_type="xensews",
            data_type=data_type,
        )
    ]
    return RobotCfg(
        robot=robot,
        tactiles=tactiles,
        gripper_offset=0.1644,
        gripper_max_qpos=abs(robotiq_close_qpos - robotiq_open_qpos),
        gripper_open_qpos=robotiq_open_qpos,
        gripper_close_qpos=robotiq_close_qpos,
        tactile_far_plane=34.0,
        adaptive_grasp_depth_threshold=27.0,
        contact_threshold=(26.8, 27.3)
    )
