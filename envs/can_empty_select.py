from ._base_task import *
import numpy as np


TASK_INSTRUCTION = "Find the empty can by touch and place it in the basket."

ASSET_ROOT = "task_assets/can_empty_select"
CAN_BRANDS = ("fanta", "pepsi")
CAN_SAFE_NAMES = {
    "fanta": "fanta",
    "pepsi": "pepsi",
}
CAN_PHYSICS_ASSET_PATHS = {
    brand: f"{ASSET_ROOT}/{CAN_SAFE_NAMES[brand]}_can_physics_proxy.usda"
    for brand in CAN_BRANDS
}
CAN_VISUAL_ASSET_PATHS = {
    brand: f"{ASSET_ROOT}/{CAN_SAFE_NAMES[brand]}_can_visual.usda"
    for brand in CAN_BRANDS
}
BASKET_PHYSICS_ASSET_PATH = f"{ASSET_ROOT}/basket_physics_proxy.usda"
BASKET_VISUAL_ASSET_PATH = f"{ASSET_ROOT}/basket_visual.usda"

LIGHT_CAN_DENSITY = 100.0
HEAVY_CAN_DENSITY = 10000.0
GRIPPER_CLOSE_QPOS_RANGE = (0.027, 0.028)
GRASP_PRE_DISTANCE = 0.04
GRASP_HEIGHT_MARGIN = 0.040
CAN_XY_NOISE = (0.01, 0.01, 0.0)
TEST_LIFT_HEIGHT = 0.05
EMPTY_CAN_LIFT_HEIGHT = 0.14
POST_RETURN_CLEARANCE = 0.14
WRONG_CAN_HOLD_DELAY_STEPS = 10
AFTER_RELEASE_DELAY_STEPS = 40

BASKET_POSE = Pose([0.5, -0.2, 0.002], [1, 0, 0, 0])
BASKET_DROP_POSE = Pose([0.5, -0.2, 0.172], [1, 0, 0, 0])
BASKET_CONTAIN_HALF_XY = 0.06108
BASKET_CONTAIN_Z_RANGE = (0.0, 0.05)

WORK_POSES = {
    "fanta": Pose([0.5, 0.02, 0.002], [1, 0, 0, 0]),
    "pepsi": Pose([0.5, 0.20, 0.002], [1, 0, 0, 0]),
}
STANDBY_POSES = [
    Pose([1.0, 0.0 + 0.1 * i, 0.004], [1, 0, 0, 0])
    for i in range(len(CAN_BRANDS) * 2)
]

CAN_HEIGHTS = {
    "fanta": 0.12199997,
    "pepsi": 0.12299984,
}

TASK_INITIAL_JOINT_POS = {
    "panda_joint1": 0.0,
    "panda_joint2": 0.0,
    "panda_joint3": 0.0,
    "panda_joint4": -2.46,
    "panda_joint5": 0.0,
    "panda_joint6": 2.5,
    "panda_joint7": 0.741,
    "panda_finger.*": 0.02,
}


@configclass
class TaskCfg(BaseTaskCfg):
    empty_can: Literal["random", "fanta", "pepsi"] = "random"
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(
                pos=(1.1, 0.02, 0.30),
                rot=(0.560985, 0.430459, 0.430459, 0.560985),
                convention="opengl",
            ),
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.5,
                focus_distance=1.0,
                horizontal_aperture=3.6,
                clipping_range=(0.1, 100.0),
            ),
            width=480,
            height=270,
            update_period=1 / 120,
        ),
        CameraCfg(
            name="wrist",
            prim_path="/World/envs/env_.*/Robot/WristCamera/Camera",
            data_types=["rgb", "depth"],
            spawn=None,
            width=480,
            height=270,
            update_period=1 / 120,
        ),
    ]
    use_adaptive_grasp = False
    step_lim = 400


class Task(BaseTask):
    def __init__(
        self,
        cfg: TaskCfg,
        mode: Literal["collect", "eval"] = "collect",
        render_mode: str | None = None,
        **kwargs,
    ):
        if cfg.empty_can not in ("random", *CAN_BRANDS):
            raise ValueError(f"empty_can must be 'random' or one of {CAN_BRANDS}")

        cfg.sim.physics_material.dynamic_friction = 2.5
        cfg.sim.physics_material.static_friction = 2.5
        cfg.uipc_sim.contact.default_friction_ratio = 2.5
        super().__init__(cfg, mode, render_mode, **kwargs)

    def load_robot_and_sensors(self, cfg: BaseTaskCfg):
        cfg = super().load_robot_and_sensors(cfg)
        joint_pos = {
            key: value
            for key, value in TASK_INITIAL_JOINT_POS.items()
            if key.startswith("panda_joint")
        }
        if getattr(cfg, "tactile_sensor_type", "") in ("xensews", "xensews_robotiq"):
            joint_pos = apply_xense_wrist_y_alignment(joint_pos)
            joint_pos["finger_joint"] = cfg.robot.gripper_open_qpos
        else:
            joint_pos["panda_finger.*"] = TASK_INITIAL_JOINT_POS["panda_finger.*"]
        cfg.robot.robot.init_state.joint_pos.update(joint_pos)
        return cfg

    def create_actors(self):
        fixed_cfg = UipcObjectCfg.AffineBodyConstitutionCfg(
            m_kappa=200.0,
            kinematic=True,
        )
        self.basket = self._actor_manager.add_from_usd_file(
            name="basket",
            asset_path=BASKET_PHYSICS_ASSET_PATH,
            visual_asset_path=BASKET_VISUAL_ASSET_PATH,
            pose=BASKET_POSE,
            constitution_cfg=fixed_cfg,
            density=5.0e3,
            show_physics_mesh=False,
        )

        self.can_actors = {}
        standby_idx = 0
        for brand in CAN_BRANDS:
            safe = CAN_SAFE_NAMES[brand]
            for weight_label, density in (
                ("light", LIGHT_CAN_DENSITY),
                ("heavy", HEAVY_CAN_DENSITY),
            ):
                actor = self._actor_manager.add_from_usd_file(
                    name=f"{safe}_{weight_label}_can",
                    asset_path=CAN_PHYSICS_ASSET_PATHS[brand],
                    visual_asset_path=CAN_VISUAL_ASSET_PATHS[brand],
                    pose=STANDBY_POSES[standby_idx],
                    density=density,
                    show_physics_mesh=False,
                )
                self.can_actors[(brand, weight_label)] = actor
                standby_idx += 1

    def _reset_actors(self):
        self.empty_can_brand = self._resolve_empty_can(self.cfg.empty_can)
        self.basket.set_pose(BASKET_POSE)

        standby_idx = 0
        for brand in CAN_BRANDS:
            for weight_label in ("light", "heavy"):
                self.can_actors[(brand, weight_label)].set_pose(
                    STANDBY_POSES[standby_idx]
                )
                standby_idx += 1

        self.active_cans = {}
        self.active_weights = {}
        self.active_work_poses = {}
        self.metadata["work_xy_noises"] = {}
        for brand in CAN_BRANDS:
            active_weight = "light" if brand == self.empty_can_brand else "heavy"
            active_can = self.can_actors[(brand, active_weight)]
            work_noise = self.create_noise(list(CAN_XY_NOISE))
            work_pose = WORK_POSES[brand].add_offset(work_noise)
            active_can.set_pose(work_pose)
            self.active_cans[brand] = active_can
            self.active_weights[brand] = active_weight
            self.active_work_poses[brand] = work_pose
            self.metadata["work_xy_noises"][brand] = work_noise.p.tolist()

        self.empty_can_actor = self.active_cans[self.empty_can_brand]
        self.target_pose = BASKET_DROP_POSE
        self.tried_brands = []

        self.metadata["empty_can"] = self.empty_can_brand
        self.metadata["empty_can_cfg"] = self.cfg.empty_can
        self.metadata["can_order"] = list(CAN_BRANDS)
        self.metadata["active_weights"] = dict(self.active_weights)
        self.metadata["work_base_poses"] = {
            brand: WORK_POSES[brand].tolist() for brand in CAN_BRANDS
        }
        self.metadata["work_poses"] = {
            brand: self.active_work_poses[brand].tolist() for brand in CAN_BRANDS
        }
        self.metadata["standby_poses"] = [
            pose.tolist() for pose in STANDBY_POSES
        ]
        self.metadata["basket_pose"] = BASKET_POSE.tolist()
        self.metadata["basket_drop_pose"] = BASKET_DROP_POSE.tolist()
        self.metadata["can_physics_assets"] = dict(CAN_PHYSICS_ASSET_PATHS)
        self.metadata["can_visual_assets"] = dict(CAN_VISUAL_ASSET_PATHS)
        self.metadata["basket_physics_asset"] = BASKET_PHYSICS_ASSET_PATH
        self.metadata["basket_visual_asset"] = BASKET_VISUAL_ASSET_PATH

    def _release_reset_constraints(self):
        self._actor_manager.remove_animate(force=True)

    def _resolve_empty_can(self, empty_can):
        if empty_can == "random":
            return str(self.rng.choice(CAN_BRANDS))
        return str(empty_can)

    def build_instruction(self) -> str:
        return TASK_INSTRUCTION

    def pre_move(self):
        self.delay(10)
        self.move(self.atom.open_gripper(1.0), tag="open_gripper_for_empty_can_search")

    def _grasp_can(self, brand):
        can = self.active_cans[brand]
        can_pose = can.get_pose()
        grasp_height = CAN_HEIGHTS[brand] - GRASP_HEIGHT_MARGIN
        grasp_pose = Pose(
            can_pose.p + np.array([0.0, 0.0, grasp_height]),
            [1, 0, 0, 0],
        )
        cpose = construct_grasp_pose(
            grasp_pose.p,
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        )
        cid = can.register_point(cpose, type="contact")
        self.move(
            self.atom.grasp_actor(
                can,
                contact_point_id=cid,
                pre_dis=GRASP_PRE_DISTANCE,
                dis=0.0,
                is_close=False,
            ),
            tag=f"approach_{brand}_{self.active_weights[brand]}_can",
        )

        gripper_qpos = self.rng.uniform(*GRIPPER_CLOSE_QPOS_RANGE) / 0.039
        self.move(
            self.atom.close_gripper(gripper_qpos),
            tag=f"close_{brand}_{self.active_weights[brand]}_can",
        )

        grasp_info = {
            "brand": brand,
            "weight": self.active_weights[brand],
            "can_pose_before_grasp": can_pose.tolist(),
            "grasp_pose": cpose.tolist(),
            "grasp_height": float(grasp_height),
            "gripper_qpos": float(gripper_qpos),
        }
        return can, grasp_info

    def _lift_current_can(self, brand, lift_height):
        self.move(
            self.atom.move_by_displacement(z=lift_height),
            tag=f"lift_{brand}_{self.active_weights[brand]}_can",
        )

    def _return_heavy_can(self, brand):
        can = self.active_cans[brand]
        self.delay(WRONG_CAN_HOLD_DELAY_STEPS, is_save=True)
        self.move(
            self.atom.place_actor(
                can,
                target_pose=self.active_work_poses[brand],
                pre_dis=0.0,
                dis=0.0,
                is_open=False,
            ),
            tag=f"return_{brand}_heavy_can_to_origin",
            time_dilation_factor=0.5,
        )
        self.move(self.atom.open_gripper(1.0), tag=f"release_{brand}_heavy_can")
        self.delay(5, is_save=False)
        self.move(
            self.atom.move_by_displacement(z=POST_RETURN_CLEARANCE),
            tag=f"lift_arm_after_returning_{brand}_heavy_can",
        )
        self.delay(5, is_save=False)

    def _drop_empty_can_in_basket(self, brand):
        can = self.active_cans[brand]
        self.move(
            self.atom.place_actor(
                can,
                target_pose=BASKET_DROP_POSE,
                pre_dis=0.0,
                dis=0.0,
                is_open=False,
            ),
            tag=f"move_{brand}_empty_can_above_basket",
            time_dilation_factor=0.5,
        )
        self.move(self.atom.open_gripper(1.0), tag=f"release_{brand}_empty_can")
        self.delay(AFTER_RELEASE_DELAY_STEPS, is_save=False)

    def _play_once(self):
        initial_ee_pose = self._robot_manager.get_ee_pose()
        self.metadata["initial_ee_pose_before_search"] = initial_ee_pose.tolist()
        self.metadata["attempts"] = []

        for brand in CAN_BRANDS:
            can, grasp_info = self._grasp_can(brand)
            is_empty = brand == self.empty_can_brand
            lift_height = EMPTY_CAN_LIFT_HEIGHT if is_empty else TEST_LIFT_HEIGHT
            self._lift_current_can(brand, lift_height)
            attempt = {
                **grasp_info,
                "is_empty": bool(is_empty),
                "lift_height": float(lift_height),
                "pose_after_lift": can.get_pose().tolist(),
            }
            self.tried_brands.append(brand)

            if is_empty:
                self._drop_empty_can_in_basket(brand)
                attempt["action"] = "drop_in_basket"
                attempt["final_pose"] = can.get_pose().tolist()
                self.metadata["attempts"].append(attempt)
                break

            self._return_heavy_can(brand)
            attempt["action"] = "return_to_origin_then_lift_arm"
            attempt["final_pose"] = can.get_pose().tolist()
            self.metadata["attempts"].append(attempt)
        else:
            raise RuntimeError("No empty can was selected during scripted search")

        self.metadata["tried_brands"] = list(self.tried_brands)

    def check_success(self):
        if not hasattr(self, "empty_can_actor"):
            return False

        basket_pose = self.basket.get_pose()
        empty_pose = self.empty_can_actor.get_pose()
        local_pose = empty_pose.rebase(basket_pose)
        local_p = local_pose.p
        in_x = abs(local_p[0]) <= BASKET_CONTAIN_HALF_XY
        in_y = abs(local_p[1]) <= BASKET_CONTAIN_HALF_XY
        in_z = BASKET_CONTAIN_Z_RANGE[0] <= local_p[2] <= BASKET_CONTAIN_Z_RANGE[1]
        success = bool(in_x and in_y and in_z)

        self.metadata["success_diagnostics"] = {
            "empty_can": self.empty_can_brand,
            "empty_can_pose": empty_pose.tolist(),
            "empty_can_pose_in_basket": local_pose.tolist(),
            "basket_contain_half_xy": float(BASKET_CONTAIN_HALF_XY),
            "basket_contain_z_range": list(BASKET_CONTAIN_Z_RANGE),
            "in_x": bool(in_x),
            "in_y": bool(in_y),
            "in_z": bool(in_z),
            "success": success,
        }
        return success
