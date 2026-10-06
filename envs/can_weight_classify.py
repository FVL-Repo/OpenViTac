from ._base_task import *
import numpy as np

TASK_INSTRUCTION = "Classify the can weight by touch: place the light can on the blue plate and the heavy can on the yellow plate."

COKE_CAN_ASSET_ROOT = "task_assets/can_weight_classify"
COKE_CAN_PHYSICS_ASSET_PATH = f"{COKE_CAN_ASSET_ROOT}/coke_can_physics_proxy.usda"
COKE_CAN_VISUAL_ASSET_PATH = f"{COKE_CAN_ASSET_ROOT}/coke_can_visual.usda"

CAN_RADIUS_X = 0.03288319
CAN_RADIUS_Y = 0.032782015
CAN_HEIGHT = 0.12299996
CAN_RESET_XY_NOISE = 0.01
GRASP_HEIGHT = CAN_HEIGHT - 0.040
GRIPPER_CLOSE_QPOS_RANGE = (0.027, 0.028)
LIFT_HEIGHT = 0.075
LIFT_HEIGHT_NOISE = 0.01
TARGET_XY_NOISE = 0.01

# ManiSkill2_real2sim uses density 50 for opened/empty cans and 2000 for
# unopened coke cans. The mesh volume here is about 4.2e-4 m^3, so these map
# to roughly 21 g and 842 g in the UIPC proxy.
LIGHT_CAN_DENSITY = 50.0
HEAVY_CAN_DENSITY = 2000.0

START_POSE = Pose([0.38, 0.0, 0.004], [1, 0, 0, 0])
LIGHT_STANDBY_POSE = Pose([0.38, 1.0, 0.004], [1, 0, 0, 0])
HEAVY_STANDBY_POSE = Pose([0.38, -1.0, 0.004], [1, 0, 0, 0])


@configclass
class TaskCfg(BaseTaskCfg):
    weight_label: Literal["random", "light", "heavy"] = "random"
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(pos=(1.04, 0.0, 0.22), rot=(0.541675, 0.454519, 0.454519, 0.541675), convention="opengl"),
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.5, focus_distance=1.0, horizontal_aperture=3.6, clipping_range=(0.1, 100.0)
            ),
            width=480,
            height=270,
            update_period=1/120
        ),
        CameraCfg(
            name="wrist",
            prim_path="/World/envs/env_.*/Robot/WristCamera/Camera",
            data_types=["rgb", "depth"],
            spawn=None,
            width=480,
            height=270,
            update_period=1/120,
        )
    ]
    use_adaptive_grasp = False


class Task(BaseTask):
    def __init__(self, cfg: TaskCfg, mode: Literal['collect', 'eval'] = 'collect', render_mode: str | None = None, **kwargs):
        if cfg.weight_label not in ("random", "light", "heavy"):
            raise ValueError("weight_label must be 'random', 'light', or 'heavy'")

        cfg.sim.physics_material.dynamic_friction = 2.5
        cfg.sim.physics_material.static_friction = 2.5
        cfg.uipc_sim.contact.default_friction_ratio = 2.5
        super().__init__(cfg, mode, render_mode, **kwargs)

    def create_actors(self):
        self.blue_plate = self._actor_manager.add_from_usd_file(
            name='blue_plate',
            asset_path="task_assets/can_weight_classify/blue_square_plate.usd",
            pose=Pose([0.54, 0.12, 0.002], [1, 0, 0, 0]),
        )
        self.yellow_plate = self._actor_manager.add_from_usd_file(
            name='yellow_plate',
            asset_path="task_assets/can_weight_classify/yellow_square_plate.usd",
            pose=Pose([0.54, -0.12, 0.002], [1, 0, 0, 0]),
        )

        self.light_can = self._actor_manager.add_from_usd_file(
            name='light_can',
            asset_path=COKE_CAN_PHYSICS_ASSET_PATH,
            visual_asset_path=COKE_CAN_VISUAL_ASSET_PATH,
            pose=LIGHT_STANDBY_POSE,
            density=LIGHT_CAN_DENSITY,
            show_physics_mesh=False,
        )
        self.heavy_can = self._actor_manager.add_from_usd_file(
            name='heavy_can',
            asset_path=COKE_CAN_PHYSICS_ASSET_PATH,
            visual_asset_path=COKE_CAN_VISUAL_ASSET_PATH,
            pose=HEAVY_STANDBY_POSE,
            density=HEAVY_CAN_DENSITY,
            show_physics_mesh=False,
        )

    def _reset_actors(self):
        self.choice = self._resolve_weight_label(self.cfg.weight_label)
        start_pose = START_POSE.add_offset(Pose([
            self.rng.uniform(-CAN_RESET_XY_NOISE, CAN_RESET_XY_NOISE),
            self.rng.uniform(-CAN_RESET_XY_NOISE, CAN_RESET_XY_NOISE),
            0.0,
        ], [1, 0, 0, 0]))

        self.light_can.set_pose(LIGHT_STANDBY_POSE)
        self.heavy_can.set_pose(HEAVY_STANDBY_POSE)

        if self.choice == 'light':
            self.can = self.light_can
            self.target = self.blue_plate
            self.other_target = self.yellow_plate
        else:
            self.can = self.heavy_can
            self.target = self.yellow_plate
            self.other_target = self.blue_plate
        self.can.set_pose(start_pose)
        self.target_pose = self.target.get_pose().add_bias([
            self.rng.uniform(-TARGET_XY_NOISE, TARGET_XY_NOISE),
            self.rng.uniform(-TARGET_XY_NOISE, TARGET_XY_NOISE),
            0.004
        ])
        self.metadata["weight_label"] = self.choice
        self.metadata["weight_label_cfg"] = self.cfg.weight_label
        self.metadata["can_density"] = float(
            LIGHT_CAN_DENSITY if self.choice == 'light' else HEAVY_CAN_DENSITY
        )
        self.metadata["can_physics_asset"] = COKE_CAN_PHYSICS_ASSET_PATH
        self.metadata["can_visual_asset"] = COKE_CAN_VISUAL_ASSET_PATH
        self.metadata["can_radius_x"] = CAN_RADIUS_X
        self.metadata["can_radius_y"] = CAN_RADIUS_Y
        self.metadata["can_height"] = CAN_HEIGHT
        self.metadata["target_plate"] = self.target.cfg.name
        self.metadata["start_pose"] = start_pose.tolist()
        self.metadata["target_pose"] = self.target_pose.tolist()

    def _release_reset_constraints(self):
        self._actor_manager.remove_animate(force=True)

    def _resolve_weight_label(self, weight_label):
        if weight_label == "random":
            return str(self.rng.choice(['light', 'heavy']))
        return str(weight_label)

    def pre_move(self):
        self.delay(10)

        self.move(self.atom.open_gripper(1.0))

    def _play_once(self):
        can_pose = self.can.get_pose()
        target_pose = Pose(can_pose.p + np.array([0.0, 0.0, GRASP_HEIGHT]), [1, 0, 0, 0])
        cpose = construct_grasp_pose(
            target_pose.p,
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        )
        cid = self.can.register_point(cpose, type='contact')
        self.move(self.atom.grasp_actor(
            self.can, contact_point_id=cid, pre_dis=0.04, dis=0.0, is_close=False
        ))
        gripper_qpos = self.rng.uniform(*GRIPPER_CLOSE_QPOS_RANGE) / 0.039
        self.move(self.atom.close_gripper(gripper_qpos))
        lift_height = LIFT_HEIGHT + self.rng.uniform(-LIFT_HEIGHT_NOISE, LIFT_HEIGHT_NOISE)
        self.move(self.atom.move_by_displacement(z=lift_height))

        self.move(self.atom.place_actor(
            self.can,
            target_pose=self.target_pose,
            pre_dis=0.0,
            dis=0.0,
            is_open=False
        ), time_dilation_factor=0.5)
        self.delay(20, is_save=False)

    def check_success(self):
        can_pose = self.can.get_pose().rebase(self.target_pose)
        xy_threshold = 0.04
        z_threshold = 0.012
        return np.all(np.abs(can_pose.p) < np.array([xy_threshold, xy_threshold, z_threshold])) and \
            np.dot(can_pose.to_transformation_matrix()[:3, 2], np.array([0, 0, 1])) > 0.94
