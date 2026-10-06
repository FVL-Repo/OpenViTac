from ._base_task import *
import numpy as np
from uipc import view


is_random = True
TASK_INSTRUCTION = "Pick up the red gear, assemble it onto the base, and rotate it to turn the gear pair."
GEAR_PHASE_OFFSET_DEG = 6.0
GEAR_RED_Q = (1.0, 0.0, 0.0, 0.0)
GEAR_BLUE_Q = (
    float(np.cos(np.deg2rad(GEAR_PHASE_OFFSET_DEG) * 0.5)),
    0.0,
    0.0,
    float(np.sin(np.deg2rad(GEAR_PHASE_OFFSET_DEG) * 0.5)),
)
BASE_INITIAL_POSITION = np.array([0.45, 0.0, 0.002])
GEAR_CENTER_DISTANCE = 0.074
RED_GEAR_INITIAL_POSITION = np.array([0.45 - GEAR_CENTER_DISTANCE * 0.5, 0.0, 0.010])
BLUE_GEAR_INITIAL_POSITION = np.array([0.45 + GEAR_CENTER_DISTANCE * 0.5, 0.0, 0.010])
RESET_XY_NOISE = 0.010
RED_GEAR_PICK_INITIAL_POSITION = np.array([0.45, -0.13, 0.002])
RED_GEAR_PICK_XY_NOISE = 0.010
RED_GEAR_PICK_PHASE_NOISE_DEG = 6.0
GEAR_ASSEMBLY_LIFT_HEIGHT = 0.035
GEAR_ASSEMBLY_PRE_INSERT_HEIGHT = 0.035
GEAR_ASSEMBLY_PRE_ALIGN_DESCENT = 0.015
GEAR_ASSEMBLY_POST_INSERT_SETTLE_STEPS = 5
GEAR_INSERT_CENTER_TOLERANCE = 0.002
GEAR_INSERT_VERTICAL_TOLERANCE = 0.0015
GEAR_INSERT_YAW_TOLERANCE_DEG = 1.5
GEAR_PLATE_HEIGHT = 0.010
GEAR_SHAFT_TOP = 0.030
GEAR_SHAFT_CENTER_HEIGHT = 0.5 * (GEAR_PLATE_HEIGHT + GEAR_SHAFT_TOP)
GEAR_ASSEMBLY_MOTION_TIME_DILATION = 0.5
# The composed Xense cases, adapters, and Robotiq inner-finger visuals have
# collision disabled. Keep the gripper center above the shaft top so their
# swept visual envelope cannot enter the gear plate while the wrist rotates.
ROBOTIQ_GEAR_TARGET_CLEARANCE_ABOVE_SHAFT_TOP = 0.001
ROBOTIQ_GEAR_MIN_SAFE_CLEARANCE_ABOVE_SHAFT_TOP = 0.001
ROBOTIQ_GEAR_CLEARANCE_TRACKING_TOLERANCE = 0.0001
ROBOTIQ_GEAR_TARGET_LOCAL_Z = (
    GEAR_SHAFT_TOP + ROBOTIQ_GEAR_TARGET_CLEARANCE_ABOVE_SHAFT_TOP
)
GEAR_ASSET_ROOT = "task_assets/turn_gear_pair"
XENSE_GEAR_PHYSICS_ASSET_PATH = f"{GEAR_ASSET_ROOT}/gear_physics_proxy.usda"
XENSE_GEAR_BASE_PHYSICS_ASSET_PATH = (
    f"{GEAR_ASSET_ROOT}/gear_base_physics_proxy.usda"
)
XENSE_GEAR_TRANSLATION_STRENGTH = 5.0e3
GEAR_ROTATION_STRENGTH = 5.0e3
GEAR_DRIVE_MIN_YAW_STEP = 1.0e-6


def _z_quat_from_deg(angle_deg):
    half_angle = np.deg2rad(angle_deg) * 0.5
    return (
        float(np.cos(half_angle)),
        0.0,
        0.0,
        float(np.sin(half_angle)),
    )


@configclass
class TaskCfg(BaseTaskCfg):
    cameras = [
        # CameraCfg(
        #     name="head",
        #     prim_path="/World/envs/env_.*/Camera",
        #     offset=CameraCfg.OffsetCfg(pos=(0.74, 0.0, 0.066), rot=(0.512, 0.512, 0.487, 0.487), convention="opengl"),
        #     data_types=["rgb", "depth"],
        #     spawn=sim_utils.PinholeCameraCfg(
        #         focal_length=2.5, focus_distance=1.0, horizontal_aperture=3.6, clipping_range=(0.1, 100.0)
        #     ),
        #     width=480,
        #     height=270,
        #     update_period=1/120
        # ),
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(pos=(0.74, 0.13, 0.15), rot=(0.370608, 0.310977, 0.562556, 0.670428), convention="opengl"),
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=1.6, focus_distance=1.0, horizontal_aperture=2.4, clipping_range=(0.1, 100.0)
            ),
            width=480,
            height=270,
            update_period=1/120,
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
    step_lim = 400


class Task(BaseTask):
    @staticmethod
    def _is_xense_cfg(cfg):
        return getattr(cfg, "tactile_sensor_type", "") in (
            "xensews",
            "xensews_robotiq",
        )

    def _uses_robotiq_gripper(self):
        return (
            getattr(getattr(self, "_robot_manager", None), "robot_type", None)
            == "franka_robotiq"
        )

    def __init__(self, cfg: BaseTaskCfg, mode: Literal["collect", "eval"] = "collect", render_mode: str | None = None, **kwargs):
        cfg.sim.physics_material.dynamic_friction = 2.5
        cfg.sim.physics_material.static_friction = 2.5
        cfg.uipc_sim.contact.default_friction_ratio = 2.5
        super().__init__(cfg, mode, render_mode, **kwargs)

    def load_robot_and_sensors(self, cfg: BaseTaskCfg):
        cfg = super().load_robot_and_sensors(cfg)
        if self._is_xense_cfg(cfg):
            joint_pos = dict(cfg.robot.robot.init_state.joint_pos)
            joint_pos = apply_xense_wrist_y_alignment(joint_pos)
            cfg.robot.robot.init_state.joint_pos.update(joint_pos)
        return cfg

    def create_actors(self):
        is_xense = self._is_xense_cfg(self.cfg)
        # base 中心放在 x=0.45；底座局部 z=0 是底面，所以世界 z=0.002 表示放在桌面上方 2mm。
        base_pose = Pose(BASE_INITIAL_POSITION, GEAR_RED_Q)
        # 两个齿轮装配后的中心距取 0.074m，所以红色在 0.45-0.037，蓝色在 0.45+0.037。
        # 齿轮局部 z=0 是底面；底板 5mm + 小方柱 2mm，顶面约 0.002+0.007=0.009m。
        # 齿轮底面设 0.010m，留约 1mm 初始间隙。
        red_pose = Pose(RED_GEAR_PICK_INITIAL_POSITION, GEAR_RED_Q)
        blue_pose = Pose(BLUE_GEAR_INITIAL_POSITION, GEAR_BLUE_Q)

        self.base = self._actor_manager.add_from_usd_file(
            name="base",
            asset_path=(
                XENSE_GEAR_BASE_PHYSICS_ASSET_PATH
                if is_xense
                else f"{GEAR_ASSET_ROOT}/gear_pair_base.usd"
            ),
            pose=base_pose,
            density=1e5,
            visual_asset_path=(
                f"{GEAR_ASSET_ROOT}/gear_pair_base.usd" if is_xense else None
            ),
            show_physics_mesh=not is_xense,
            keep_constrained=is_xense,
        )
        self.red_gear = self._actor_manager.add_from_usd_file(
            name="red_gear",
            asset_path=(
                XENSE_GEAR_PHYSICS_ASSET_PATH
                if is_xense
                else f"{GEAR_ASSET_ROOT}/gear_pair_red.usd"
            ),
            pose=red_pose,
            density=1e2,
            visual_asset_path=(
                f"{GEAR_ASSET_ROOT}/gear_pair_red.usd" if is_xense else None
            ),
            show_physics_mesh=not is_xense,
            keep_constrained=True,
        )
        self.blue_gear = self._actor_manager.add_from_usd_file(
            name="blue_gear",
            asset_path=(
                XENSE_GEAR_PHYSICS_ASSET_PATH
                if is_xense
                else f"{GEAR_ASSET_ROOT}/gear_pair_blue.usd"
            ),
            pose=blue_pose,
            density=1e2,
            visual_asset_path=(
                f"{GEAR_ASSET_ROOT}/gear_pair_blue.usd" if is_xense else None
            ),
            show_physics_mesh=not is_xense,
            keep_constrained=True,
        )

        # Keep the real gear meshes shape-stable without replacing their
        # collision geometry with simplified shafts.
        translation_strength = (
            XENSE_GEAR_TRANSLATION_STRENGTH if is_xense else 5.0e3
        )
        for gear in (self.red_gear, self.blue_gear):
            strength_ratio = gear.uipc_meshes[0].instances().find("strength_ratio")
            if strength_ratio is None:
                raise RuntimeError("Missing SoftTransformConstraint strength_ratio")
            strength_view = view(strength_ratio)
            strength_view[:, 0, :] = translation_strength
            strength_view[:, 1, :] = GEAR_ROTATION_STRENGTH

    def _reset_actors(self):
        self._gear_drive_enabled = False
        self._gear_drive_angle = 0.0
        self._gear_drive_prev_gripper_pose = None
        self._gear_drive_pose_updates = 0
        self._red_gear_approached = False
        self._red_gear_assembled = False
        self._robotiq_grasp_clearance_monitor_enabled = False
        self._robotiq_grasp_planned_clearance_safe = True
        self._robotiq_grasp_min_clearance = None
        self._robotiq_grasp_height_failure_reason = None

        # 底座和蓝齿轮整体在 xy 平面内随机平移，保持装配目标相对位置不变。
        xy_offset = (
            self.rng.uniform(-RESET_XY_NOISE, RESET_XY_NOISE, size=2)
            if is_random
            else np.zeros(2)
        )
        offset = np.array([xy_offset[0], xy_offset[1], 0.0])
        red_pick_xy_noise = (
            self.rng.uniform(
                -RED_GEAR_PICK_XY_NOISE,
                RED_GEAR_PICK_XY_NOISE,
                size=2,
            )
            if is_random
            else np.zeros(2)
        )
        red_pick_phase_noise_deg = (
            float(
                self.rng.uniform(
                    -RED_GEAR_PICK_PHASE_NOISE_DEG,
                    RED_GEAR_PICK_PHASE_NOISE_DEG,
                )
            )
            if is_random
            else 0.0
        )
        red_pick_offset = np.array(
            [red_pick_xy_noise[0], red_pick_xy_noise[1], 0.0]
        )

        base_pose = Pose(BASE_INITIAL_POSITION + offset, GEAR_RED_Q)
        red_pick_pose = Pose(
            RED_GEAR_PICK_INITIAL_POSITION + offset + red_pick_offset,
            _z_quat_from_deg(red_pick_phase_noise_deg),
        )
        red_assembly_target_pose = Pose(
            RED_GEAR_INITIAL_POSITION + offset,
            GEAR_RED_Q,
        )
        blue_pose = Pose(BLUE_GEAR_INITIAL_POSITION + offset, GEAR_BLUE_Q)
        self.base.set_pose(base_pose)
        self.red_gear.set_pose(red_pick_pose)
        self.blue_gear.set_pose(blue_pose)
        self._reset_red_pose = red_pick_pose
        self._red_gear_pick_pose = red_pick_pose
        self._red_gear_assembly_target_pose = red_assembly_target_pose
        self._red_gear_insert_pose = red_assembly_target_pose
        self._reset_blue_pose = blue_pose

        self.metadata["base_pose"] = base_pose.tolist()
        self.metadata["reset_xy_offset"] = xy_offset.tolist()
        self.metadata["red_gear_pick_xy_noise"] = red_pick_xy_noise.tolist()
        self.metadata["red_gear_pick_phase_noise_deg"] = float(
            red_pick_phase_noise_deg
        )
        self.metadata["gear_center_distance"] = GEAR_CENTER_DISTANCE
        self.metadata["gear_phase_offset_deg"] = GEAR_PHASE_OFFSET_DEG
        self.metadata["gear_collision_asset"] = (
            "xense_stepped_plate_and_shaft_proxy_with_real_visual"
            if self._is_xense_cfg(self.cfg)
            else "real_gear_mesh_usd"
        )
        self.metadata["gear_proxy_geometry"] = {
            "plate_height_m": GEAR_PLATE_HEIGHT,
            "shaft_top_m": GEAR_SHAFT_TOP,
            "shaft_radius_m": 0.015,
            "center_hole_radius_m": 0.007075,
        }
        self.metadata["gear_constraint_translation_strength"] = float(
            XENSE_GEAR_TRANSLATION_STRENGTH
            if self._is_xense_cfg(self.cfg)
            else 5.0e3
        )
        self.metadata["gear_constraint_rotation_strength"] = float(
            GEAR_ROTATION_STRENGTH
        )
        self.metadata["red_gear_pose"] = red_pick_pose.tolist()
        self.metadata["red_gear_pick_pose"] = red_pick_pose.tolist()
        self.metadata["red_gear_assembly_target_pose"] = (
            red_assembly_target_pose.tolist()
        )
        self.metadata["red_gear_insert_pose"] = red_assembly_target_pose.tolist()
        self.metadata["blue_gear_pose"] = blue_pose.tolist()

    def _release_reset_constraints(self):
        self._actor_manager.remove_animate(force=True)

    @staticmethod
    def _rotate_pose_about_world_z(pose, angle):
        pose_mat = pose.to_transformation_matrix()
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        world_rotation = np.array(
            [
                [cos_angle, -sin_angle, 0.0],
                [sin_angle, cos_angle, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        target_mat = pose_mat.copy()
        target_mat[:3, :3] = world_rotation @ pose_mat[:3, :3]
        return Pose.from_matrix(target_mat)

    def _sync_driven_gears_to_gripper(self):
        if not getattr(self, "_gear_drive_enabled", False):
            return

        current_gripper_pose = self._robot_manager.get_gripper_center_pose()
        previous_gripper_pose = self._gear_drive_prev_gripper_pose
        self._gear_drive_prev_gripper_pose = current_gripper_pose
        if previous_gripper_pose is None:
            return

        current_rotation = current_gripper_pose.to_transformation_matrix()[:3, :3]
        previous_rotation = previous_gripper_pose.to_transformation_matrix()[:3, :3]
        relative_rotation = current_rotation @ previous_rotation.T
        yaw_step = float(
            np.arctan2(relative_rotation[1, 0], relative_rotation[0, 0])
        )
        if abs(yaw_step) <= GEAR_DRIVE_MIN_YAW_STEP:
            return

        self._gear_drive_angle += yaw_step
        self.red_gear.set_pose(
            self._rotate_pose_about_world_z(self.initial_red_pose, self._gear_drive_angle)
        )
        self.blue_gear.set_pose(
            self._rotate_pose_about_world_z(self.initial_blue_pose, -self._gear_drive_angle)
        )
        self._actor_manager.update(dt=0.0)
        self._gear_drive_pose_updates += 1

    def _stop_gear_drive(self):
        self._gear_drive_enabled = False
        self._gear_drive_prev_gripper_pose = None
        self.metadata["gear_drive_enabled_final"] = False
        self.metadata["gear_drive_pose_updates"] = int(
            self._gear_drive_pose_updates
        )

    def _start_gear_drive(self):
        self._gear_drive_angle = 0.0
        self._gear_drive_prev_gripper_pose = self._robot_manager.get_gripper_center_pose()
        self._gear_drive_enabled = True
        self.metadata["gear_drive_mode"] = "wrist_soft_joint_real_mesh_1_to_1"
        self.metadata["gear_drive_enabled_initial"] = True

    def _step(self, is_save: bool = True):
        # The full-mesh UIPC revolute joint is prohibitively slow with Xense
        # contact. Drive real-gear transform targets from measured wrist motion.
        self._sync_driven_gears_to_gripper()
        result = super()._step(is_save=is_save)
        self._record_robotiq_grasp_clearance()
        return result

    def _record_robotiq_grasp_clearance(self):
        if not getattr(self, "_robotiq_grasp_clearance_monitor_enabled", False):
            return
        if not self._uses_robotiq_gripper():
            return

        initial_red_pose = getattr(self, "initial_red_pose", None)
        if initial_red_pose is None:
            return

        gripper_pose = self._robot_manager.get_gripper_center_pose()
        local_z = float(gripper_pose.p[2] - initial_red_pose.p[2])
        clearance = local_z - GEAR_SHAFT_TOP
        previous_min = self._robotiq_grasp_min_clearance
        if previous_min is None or clearance < previous_min:
            self._robotiq_grasp_min_clearance = clearance

        measured_safe = (
            self._robotiq_grasp_min_clearance
            + ROBOTIQ_GEAR_CLEARANCE_TRACKING_TOLERANCE
            >= ROBOTIQ_GEAR_MIN_SAFE_CLEARANCE_ABOVE_SHAFT_TOP
        )
        if not measured_safe:
            self._robotiq_grasp_height_failure_reason = (
                "actual_clearance_below_robotiq_safety_margin"
            )

        self.metadata["robotiq_gear_gripper_current_local_z"] = local_z
        self.metadata["robotiq_gear_gripper_current_clearance"] = clearance
        self.metadata["robotiq_gear_gripper_min_clearance"] = float(
            self._robotiq_grasp_min_clearance
        )
        self.metadata["robotiq_gear_measured_clearance_safe"] = bool(measured_safe)

    def _record_initial_gear_poses(self):
        if (
            hasattr(self, "_red_gear_assembly_target_pose")
            and hasattr(self, "_reset_blue_pose")
        ):
            self.initial_red_pose = self._red_gear_assembly_target_pose
            self.initial_blue_pose = self._reset_blue_pose
            initial_pose_source = "original_task_red_gear_pose"
        elif self._is_xense_cfg(self.cfg) and hasattr(self, "_reset_red_pose"):
            self.initial_red_pose = self._reset_red_pose
            self.initial_blue_pose = self._reset_blue_pose
            initial_pose_source = "reset_target_pose"
        else:
            self.initial_red_pose = self.red_gear.get_pose()
            self.initial_blue_pose = self.blue_gear.get_pose()
            initial_pose_source = "settled_actor_pose"
        self.metadata["initial_red_pose"] = self.initial_red_pose.tolist()
        self.metadata["initial_blue_pose"] = self.initial_blue_pose.tolist()
        self.metadata["initial_gear_pose_source"] = initial_pose_source

    def _record_turn_initial_gear_poses(self):
        self.initial_red_pose = self.red_gear.get_pose()
        self.initial_blue_pose = self.blue_gear.get_pose()
        self.metadata["initial_red_pose"] = self.initial_red_pose.tolist()
        self.metadata["initial_blue_pose"] = self.initial_blue_pose.tolist()
        self.metadata["initial_gear_pose_source"] = "post_insert_settled_actor_pose"

    def reset(
        self,
        seed: int = -1,
        instructions: list[str] | None = None,
        options: dict[str, Any] | None = None,
    ):
        ret = super().reset(seed=seed, instructions=instructions, options=options)
        if self.cfg.skip_pre_move or (
            options is not None and not options.get("run_pre_move", True)
        ):
            self._record_initial_gear_poses()
        return ret

    def pre_move(self):
        self._record_initial_gear_poses()
        is_xense = getattr(self.cfg, "tactile_sensor_type", "") in (
            "xensews",
            "xensews_robotiq",
        )
        open_percent = 0.5
        self.metadata["gripper_open_percent"] = float(open_percent)
        if not is_xense:
            self.move(self.atom.open_gripper(open_percent), delay=False)
        else:
            self._approach_red_gear()
            self._update_render()

    def prepare_initial_state(self):
        if not self._is_xense_cfg(self.cfg):
            return
        self._record_initial_gear_poses()
        open_percent = 0.5
        self.metadata["gripper_open_percent"] = float(open_percent)
        self.move(
            self.atom.open_gripper(open_percent),
            tag="setup_open_gripper_for_policy",
            is_save=False,
            delay=False,
        )
        self.delay(20, is_save=False)

    def _approach_red_gear(self, is_save: bool = True):
        if not hasattr(self, "initial_red_pose"):
            self._record_initial_gear_poses()
        red_pose = self.red_gear.get_pose()
        grasp_pose, grasp_height_source, grasp_target_local_z = (
            self._get_red_gear_grasp_pose(red_pose)
        )
        if self._uses_robotiq_gripper():
            planned_clearance = grasp_target_local_z - GEAR_SHAFT_TOP
            self._robotiq_grasp_planned_clearance_safe = (
                planned_clearance + 1e-12
                >= ROBOTIQ_GEAR_MIN_SAFE_CLEARANCE_ABOVE_SHAFT_TOP
            )
            self._robotiq_grasp_clearance_monitor_enabled = True
            if not self._robotiq_grasp_planned_clearance_safe:
                self._robotiq_grasp_height_failure_reason = (
                    "planned_clearance_below_robotiq_safety_margin"
                )
            self.metadata["gear_visual_clearance_above_shaft_top"] = float(
                ROBOTIQ_GEAR_TARGET_CLEARANCE_ABOVE_SHAFT_TOP
            )
            self.metadata["gear_gripper_visual_collision_enabled"] = False
            self.metadata["robotiq_gear_min_safe_clearance"] = float(
                ROBOTIQ_GEAR_MIN_SAFE_CLEARANCE_ABOVE_SHAFT_TOP
            )
            self.metadata["robotiq_gear_clearance_tracking_tolerance"] = float(
                ROBOTIQ_GEAR_CLEARANCE_TRACKING_TOLERANCE
            )
            self.metadata["robotiq_gear_planned_clearance_safe"] = bool(
                self._robotiq_grasp_planned_clearance_safe
            )

        self.metadata["gear_grasp_height_source"] = grasp_height_source
        self.metadata["gear_grasp_target_local_z"] = float(grasp_target_local_z)
        self.metadata["gear_grasp_world_y_bias"] = float(
            self.get_xense_grasp_height_bias("xense_gear_grasp_world_y_bias")
        )
        self.metadata["gear_grasp_pose_randomized"] = False
        self.metadata["pre_grasp_dis"] = 0.050
        self.metadata["red_grasp_pose"] = grasp_pose.tolist()

        cid = self.red_gear.register_point(grasp_pose, type="contact")
        # pre_dis=0.050 会让 planner 先到抓取点正上方 5cm，再沿竖直方向下压到抓取高度。
        approach_actions = self.atom.grasp_actor(
            self.red_gear,
            contact_point_id=cid,
            pre_dis=0.050,
            dis=0.0,
            is_close=False,
        )
        self.move(
            approach_actions,
            tag="grasp_red_gear",
            is_save=is_save,
            delay=False,
        )
        self._red_gear_approached = True
        self.record_xense_grasp_debug("xense_after_approach_red_gear", self.red_gear)

    def _get_red_gear_grasp_pose(self, red_pose):
        uses_robotiq = self._uses_robotiq_gripper()
        if uses_robotiq:
            grasp_target_local_z = ROBOTIQ_GEAR_TARGET_LOCAL_Z
            grasp_height_source = "collision_disabled_gripper_visual_envelope"
        else:
            grasp_target_local_z = GEAR_SHAFT_CENTER_HEIGHT
            grasp_height_source = "gear_shaft_center"
        grasp_world_y_bias = self.get_xense_grasp_height_bias("xense_gear_grasp_world_y_bias")
        target_pose = (
            red_pose
            .add_bias(
                [0.0, 0.0, grasp_target_local_z],
                coord="world",
            )
            .add_bias([0.0, grasp_world_y_bias, 0.0], coord="world")
        )
        # grasp_from=[0,0,1] 表示从目标点上方沿 -Z 方向接近，爪子姿态尽量竖直向下。
        grasp_pose = construct_grasp_pose(
            target_pose.p,
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 0.0, 0.0]),
        )
        return grasp_pose, grasp_height_source, grasp_target_local_z

    def _close_red_gear(self, enable_drive: bool = True):
        close_percent = self.get_xense_close_percent("xense_gear_close_percent")
        self.move(
            self.atom.close_gripper(pos=close_percent),
            tag="close_gripper",
            delay=False,
            gripper_depth_threshold=self.get_xense_adaptive_grasp_depth_threshold(
                "xense_gear_adaptive_grasp_depth_threshold"
            ),
            gripper_require_both_contacts=self.get_xense_adaptive_grasp_require_both_contacts(
                "xense_gear_adaptive_grasp_require_both_contacts"
            ),
        )
        if enable_drive:
            self._start_gear_drive()
        self.settle_xense_after_close(is_save=False)
        self.record_xense_grasp_debug("xense_after_close_red_gear", self.red_gear)
        self.metadata["gripper_close_percent"] = float(close_percent)
        self._record_red_inhand_slip("after_initial_close")

    def _grasp_red_gear(self, enable_drive: bool = True):
        self._approach_red_gear()
        self._close_red_gear(enable_drive=enable_drive)

    def _record_red_inhand_slip(self, prefix):
        gripper_pose = self._robot_manager.get_gripper_center_pose()
        inhand_pose = self.red_gear.get_pose().rebase(to_coord=gripper_pose)
        self.metadata[f"{prefix}_red_inhand_pose"] = inhand_pose.tolist()
        reference_pose = getattr(self, "_red_gear_inhand_reference_pose", None)
        if reference_pose is None:
            self._red_gear_inhand_reference_pose = inhand_pose
            self.metadata["red_inhand_reference_prefix"] = prefix
            return

        delta = inhand_pose.p - reference_pose.p
        self.metadata[f"{prefix}_red_inhand_slip"] = delta.tolist()
        self.metadata[f"{prefix}_red_inhand_slip_norm"] = float(
            np.linalg.norm(delta)
        )
        self.metadata[f"{prefix}_red_inhand_slip_z"] = float(delta[2])

    def _get_red_insert_result(self):
        target_pose = getattr(
            self,
            "_red_gear_assembly_target_pose",
            getattr(self, "_red_gear_insert_pose", None),
        )
        (
            center_delta,
            center_xy_delta,
            vertical_delta,
            yaw_signed_deg,
        ) = self._get_actor_turn_result(self.red_gear, target_pose)
        return {
            "red_insert_center_delta": center_delta,
            "red_insert_center_xy_delta": center_xy_delta,
            "red_insert_vertical_delta": vertical_delta,
            "red_insert_yaw_signed_deg": yaw_signed_deg,
            "red_insert_yaw_delta_deg": abs(float(yaw_signed_deg)),
        }

    def _is_red_insert_success(self, result):
        return (
            result["red_insert_center_xy_delta"] <= GEAR_INSERT_CENTER_TOLERANCE
            and result["red_insert_vertical_delta"] <= GEAR_INSERT_VERTICAL_TOLERANCE
            and result["red_insert_yaw_delta_deg"] <= GEAR_INSERT_YAW_TOLERANCE_DEG
        )

    def _record_red_insert_result(self):
        result = self._get_red_insert_result()
        for key, value in result.items():
            self.metadata[key] = float(value)
        success = self._is_red_insert_success(result)
        self.metadata["red_gear_insert_success"] = bool(success)
        return result

    def _insert_red_gear_onto_base(self):
        assembly_target_pose = getattr(
            self,
            "_red_gear_assembly_target_pose",
            getattr(self, "_red_gear_insert_pose", None),
        )
        if assembly_target_pose is None:
            raise RuntimeError("Missing red gear assembly target pose; reset must run first.")

        is_xense = self._is_xense_cfg(self.cfg)
        pre_insert_position = assembly_target_pose.p.copy()
        pre_insert_position[2] += GEAR_ASSEMBLY_PRE_INSERT_HEIGHT
        pre_align_position = assembly_target_pose.p.copy()
        pre_align_position[2] += (
            GEAR_ASSEMBLY_PRE_INSERT_HEIGHT - GEAR_ASSEMBLY_PRE_ALIGN_DESCENT
        )
        pre_align_pose = Pose(pre_align_position, assembly_target_pose.q)

        self.metadata["gear_assembly_lift_height"] = float(
            GEAR_ASSEMBLY_LIFT_HEIGHT
        )
        self.metadata["gear_assembly_pre_insert_height"] = float(
            GEAR_ASSEMBLY_PRE_INSERT_HEIGHT
        )
        self.metadata["gear_assembly_pre_align_descent"] = float(
            GEAR_ASSEMBLY_PRE_ALIGN_DESCENT
        )
        self.metadata["gear_assembly_post_insert_settle_steps"] = int(
            GEAR_ASSEMBLY_POST_INSERT_SETTLE_STEPS
        )
        self.metadata["gear_assembly_motion_time_dilation"] = float(
            GEAR_ASSEMBLY_MOTION_TIME_DILATION
        )
        self.metadata["red_gear_pre_insert_pose"] = Pose(
            pre_insert_position,
            assembly_target_pose.q,
        ).tolist()
        self.metadata["red_gear_pre_align_pose"] = pre_align_pose.tolist()
        self.metadata["red_gear_alignment_target_pose"] = (
            assembly_target_pose.tolist()
        )

        metadata_prefix = "xense_red_gear" if is_xense else "red_gear"
        lift_position = self.red_gear.get_pose().p.copy()
        lift_position[2] += GEAR_ASSEMBLY_LIFT_HEIGHT
        self.metadata["red_gear_lift_target_position"] = lift_position.tolist()
        self.move_actor_with_gripper_center_to_position(
            self.red_gear,
            lift_position,
            tag="lift_red_gear_from_pick_pose",
            segments=1,
            settle_steps=0,
            time_dilation_factor=GEAR_ASSEMBLY_MOTION_TIME_DILATION,
            metadata_prefix=f"{metadata_prefix}_lift_path",
        )
        self._record_red_inhand_slip("after_lift")
        self.record_xense_grasp_debug("xense_after_lift_red_gear", self.red_gear)

        self.move_actor_with_gripper_center_to_position(
            self.red_gear,
            pre_insert_position,
            tag="carry_red_gear_above_base",
            segments=1,
            settle_steps=0,
            time_dilation_factor=GEAR_ASSEMBLY_MOTION_TIME_DILATION,
            metadata_prefix=f"{metadata_prefix}_carry_path",
        )
        self._record_red_inhand_slip("after_carry")
        self.record_xense_grasp_debug("xense_after_carry_red_gear", self.red_gear)

        aligned_gripper_center_pose = self.gripper_center_pose_for_actor_target(
            self.red_gear,
            pre_align_pose,
        )
        self.metadata["red_gear_aligned_gripper_center_pose"] = (
            aligned_gripper_center_pose.tolist()
        )
        self.move(
            [Action(
                "move",
                target_pose=self._robot_manager.gripper_center_to_ee(
                    aligned_gripper_center_pose
                ),
            )],
            tag="lower_and_align_red_gear_phase_above_insert",
            delay=False,
            time_dilation_factor=GEAR_ASSEMBLY_MOTION_TIME_DILATION,
        )
        self._record_red_inhand_slip("after_phase_align")
        self.record_xense_grasp_debug("xense_after_lower_and_align_red_gear", self.red_gear)

        aligned_result = self._get_red_insert_result()
        self.metadata["red_pre_insert_phase_yaw_error_deg"] = float(
            aligned_result["red_insert_yaw_signed_deg"]
        )
        self.metadata["red_pre_insert_phase_center_xy_delta"] = float(
            aligned_result["red_insert_center_xy_delta"]
        )

        insert_start_actor_pose = self.red_gear.get_pose()
        insert_descent = float(
            max(0.0, insert_start_actor_pose.p[2] - assembly_target_pose.p[2])
        )
        insert_start_gripper_center_pose = (
            self._robot_manager.get_gripper_center_pose()
        )
        insert_target_gripper_center_pose = (
            insert_start_gripper_center_pose.add_bias(
                [0.0, 0.0, -insert_descent],
                coord="world",
            )
        )
        self.metadata[f"{metadata_prefix}_insert_path_actor_start_pose"] = (
            insert_start_actor_pose.tolist()
        )
        self.metadata[f"{metadata_prefix}_insert_path_actor_target_position"] = (
            assembly_target_pose.p.tolist()
        )
        self.metadata["red_gear_insert_descent"] = insert_descent
        self.move_gripper_center_path(
            [insert_target_gripper_center_pose],
            tag="insert_red_gear_onto_base",
            delay=False,
            settle_steps=GEAR_ASSEMBLY_POST_INSERT_SETTLE_STEPS,
            time_dilation_factor=GEAR_ASSEMBLY_MOTION_TIME_DILATION,
            metadata_prefix=f"{metadata_prefix}_insert_path",
        )
        self._record_red_inhand_slip("after_insert")

        if GEAR_ASSEMBLY_POST_INSERT_SETTLE_STEPS > 0:
            self.delay(GEAR_ASSEMBLY_POST_INSERT_SETTLE_STEPS, is_save=True)
        self._red_gear_assembled = True
        self.record_xense_grasp_debug("xense_after_insert_red_gear", self.red_gear)
        result = self._record_red_insert_result()
        for key, value in result.items():
            self.metadata[f"red_physical_insert_{key}"] = float(value)
        print(
            "[turn_gear_pair] "
            f"insert red gear, xy drift: {result['red_insert_center_xy_delta']:.4f} m, "
            f"z drift: {result['red_insert_vertical_delta']:.4f} m, "
            f"yaw error: {result['red_insert_yaw_signed_deg']:.2f} deg",
            flush=True,
        )

    def _play_once(self):
        if self._is_xense_cfg(self.cfg):
            if not getattr(self, "_red_gear_approached", False):
                self._approach_red_gear()
            self._close_red_gear(enable_drive=False)
        else:
            self._grasp_red_gear(enable_drive=False)
        self._insert_red_gear_onto_base()
        self._record_turn_initial_gear_poses()
        self._start_gear_drive()
        # One 45-degree wrist turn is sufficient for this task variant.
        for turn_idx in range(1):
            self.move(
                self.atom.move_by_displacement(
                    rpy=[0.0, 0.0, np.pi / 4],
                    xyz_coord="local",
                ),
                tag=f"turn_red_gear_{turn_idx + 1}",
                delay=False,
                time_dilation_factor=0.5,
                constraint_pose=[0, 0, 0, 1, 1, 1],
            )
            # Apply the final gripper pose, which otherwise arrives one UIPC
            # frame after the last arm control target.
            self._sync_driven_gears_to_gripper()
            self._step(is_save=True)
            result = self._get_gear_pair_turn_result()
            self._record_turn_result(result, suffix=f"_after_turn_{turn_idx + 1}")
            self._save_metadata()
            print(
                "[turn_gear_pair] "
                f"turn {turn_idx + 1}, red yaw: {result['red_yaw_signed_deg']:.2f} deg, "
                f"blue yaw: {result['blue_yaw_signed_deg']:.2f} deg, "
                f"red drift: {result['red_center_delta']:.4f} m, "
                f"blue drift: {result['blue_center_delta']:.4f} m",
                flush=True,
            )
            if self._is_physical_success(result):
                break

        # The last set_pose targets remain active through the existing soft
        # transform constraints. Stop following the wrist so later render or
        # settle steps cannot re-inject the same pose or rotate the pair again.
        self._stop_gear_drive()
        result = self._get_gear_pair_turn_result()
        self._record_turn_result(result)
        self.metadata["gear_drive_angle_deg"] = float(
            np.rad2deg(self._gear_drive_angle)
        )
        self.record_xense_grasp_debug("xense_after_turn_red_gear", self.red_gear)
        print(
            "[turn_gear_pair] "
            f"red yaw: {result['red_yaw_signed_deg']:.2f} deg, "
            f"blue yaw: {result['blue_yaw_signed_deg']:.2f} deg, "
            f"red drift: {result['red_center_delta']:.4f} m, "
            f"blue drift: {result['blue_center_delta']:.4f} m"
        )

    @staticmethod
    def _get_actor_turn_result(actor, initial_pose):
        if initial_pose is None:
            return float("inf"), float("inf"), float("inf"), 0.0

        actor_pose = actor.get_pose()
        init_mat = initial_pose.to_transformation_matrix()
        curr_mat = actor_pose.to_transformation_matrix()
        init_x = init_mat[:3, 0].copy()
        curr_x = curr_mat[:3, 0].copy()
        init_x[2] = 0.0
        curr_x[2] = 0.0
        init_x /= np.linalg.norm(init_x) + 1e-8
        curr_x /= np.linalg.norm(curr_x) + 1e-8

        yaw_delta = np.arctan2(
            np.dot(np.cross(init_x, curr_x), np.array([0.0, 0.0, 1.0])),
            np.dot(init_x, curr_x),
        )
        position_delta = actor_pose.p - initial_pose.p
        center_delta = np.linalg.norm(position_delta)
        center_xy_delta = np.linalg.norm(position_delta[:2])
        vertical_delta = abs(float(position_delta[2]))
        return (
            float(center_delta),
            float(center_xy_delta),
            vertical_delta,
            float(np.rad2deg(yaw_delta)),
        )

    def _get_gear_pair_turn_result(self):
        (
            red_center_delta,
            red_center_xy_delta,
            red_vertical_delta,
            red_yaw_signed_deg,
        ) = self._get_actor_turn_result(
            self.red_gear, getattr(self, "initial_red_pose", None)
        )
        (
            blue_center_delta,
            blue_center_xy_delta,
            blue_vertical_delta,
            blue_yaw_signed_deg,
        ) = self._get_actor_turn_result(
            self.blue_gear, getattr(self, "initial_blue_pose", None)
        )
        return {
            "red_center_delta": red_center_delta,
            "blue_center_delta": blue_center_delta,
            "red_center_xy_delta": red_center_xy_delta,
            "blue_center_xy_delta": blue_center_xy_delta,
            "red_vertical_delta": red_vertical_delta,
            "blue_vertical_delta": blue_vertical_delta,
            "red_yaw_signed_deg": red_yaw_signed_deg,
            "blue_yaw_signed_deg": blue_yaw_signed_deg,
        }

    def _record_turn_result(self, result, suffix=""):
        for key, value in result.items():
            self.metadata[f"{key}{suffix}"] = float(value)
        self.metadata[f"red_yaw_delta_deg{suffix}"] = abs(
            float(result["red_yaw_signed_deg"])
        )
        self.metadata[f"blue_yaw_delta_deg{suffix}"] = abs(
            float(result["blue_yaw_signed_deg"])
        )

    def _is_physical_success(self, result):
        red_yaw = result["red_yaw_signed_deg"]
        blue_yaw = result["blue_yaw_signed_deg"]
        rotation_success = (
            abs(red_yaw) >= 30.0
            and abs(blue_yaw) >= 30.0
            and red_yaw * blue_yaw < 0.0
        )
        if self._is_xense_cfg(self.cfg):
            position_success = (
                result["red_center_xy_delta"] <= 0.01
                and result["blue_center_xy_delta"] <= 0.01
                and result["red_vertical_delta"] <= 0.005
                and result["blue_vertical_delta"] <= 0.005
            )
        else:
            position_success = (
                result["red_center_delta"] <= 0.01
                and result["blue_center_delta"] <= 0.01
            )
        return rotation_success and position_success

    def _is_robotiq_grasp_height_safe(self):
        if not self._uses_robotiq_gripper():
            return True

        min_clearance = self._robotiq_grasp_min_clearance
        measured_safe = (
            min_clearance is not None
            and min_clearance + ROBOTIQ_GEAR_CLEARANCE_TRACKING_TOLERANCE
            >= ROBOTIQ_GEAR_MIN_SAFE_CLEARANCE_ABOVE_SHAFT_TOP
        )
        height_safe = bool(
            self._robotiq_grasp_planned_clearance_safe and measured_safe
        )
        if not height_safe and self._robotiq_grasp_height_failure_reason is None:
            self._robotiq_grasp_height_failure_reason = (
                "robotiq_grasp_clearance_not_measured"
                if min_clearance is None
                else "actual_clearance_below_robotiq_safety_margin"
            )

        self.metadata["robotiq_gear_grasp_height_safe"] = height_safe
        self.metadata["robotiq_gear_grasp_height_failure_reason"] = (
            self._robotiq_grasp_height_failure_reason
        )
        return height_safe

    def check_success(self):
        result = self._get_gear_pair_turn_result()
        self._record_turn_result(result)
        physical_success = self._is_physical_success(result)
        grasp_height_safe = self._is_robotiq_grasp_height_safe()
        insert_success = bool(self.metadata.get("red_gear_insert_success", True))
        self.metadata["gear_physical_success"] = bool(physical_success)
        self.metadata["gear_assembly_success"] = bool(insert_success)
        return physical_success and grasp_height_safe and insert_success
