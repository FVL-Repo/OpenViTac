from ._base_task import *
import numpy as np


TASK_INSTRUCTION = "Grasp the light bulb and screw it into the socket."

ASSET_ROOT = "task_assets/screw_bulb"
SOCKET_ASSET_PATH = f"{ASSET_ROOT}/lamp_socket_two_turn_thread.usd"
BULB_ASSET_PATH = f"{ASSET_ROOT}/light_bulb_two_turn_thread.usd"

SOCKET_HEIGHT = 0.055
BULB_HEIGHT = 0.0868
BULB_GLASS_MAX_RADIUS = 0.0200
BULB_THREAD_MAJOR_RADIUS = 0.0146
THREAD_PITCH = 0.010
SCREW_TURN_ANGLE = np.pi / 2
SCREW_TURN_COUNT = 2
TARGET_ROTATION_DEG = 180.0
SUCCESS_ROTATION_DEG = 145.0
SUCCESS_INSERTION_TOLERANCE = 0.001
SUCCESS_INSERTION_DEPTH = max(
    0.0,
    THREAD_PITCH * SUCCESS_ROTATION_DEG / 360.0 - SUCCESS_INSERTION_TOLERANCE,
)
INITIAL_PILOT_INSERTION_DEPTH = 0.008
RESET_XY_NOISE = 0.010

SOCKET_BASE_POSITION = np.array([0.55, 0.0, 0.002])
IDENTITY_Q = (1.0, 0.0, 0.0, 0.0)

GRASP_LOCAL_Z = 0.058
GRASP_PRE_DISTANCE = 0.050
GRIPPER_OPEN_PERCENT = 0.85
GRIPPER_RELEASE_PERCENT = 0.80
GRIPPER_ADAPTIVE_CLOSE_TARGET = 0.0
WRIST_TURN_STEPS = 120


@configclass
class TaskCfg(BaseTaskCfg):
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(
                pos=(0.72, 0.16, 0.16),
                rot=(0.437426, 0.300596, 0.512602, 0.675369),
                convention="opengl",
            ),
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=1.6,
                focus_distance=1.0,
                horizontal_aperture=2.4,
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
    step_lim = 380
    xense_bulb_adaptive_grasp_depth_threshold: float | None = 27.5
    xense_bulb_adaptive_grasp_require_both_contacts: bool | None = True


class Task(BaseTask):
    def __init__(
        self,
        cfg: TaskCfg,
        mode: Literal["collect", "eval"] = "collect",
        render_mode: str | None = None,
        **kwargs,
    ):
        cfg.sim.physics_material.dynamic_friction = 2.5
        cfg.sim.physics_material.static_friction = 2.5
        cfg.uipc_sim.contact.default_friction_ratio = 2.5
        super().__init__(cfg=cfg, mode=mode, render_mode=render_mode, **kwargs)

    @staticmethod
    def _is_xense_cfg(cfg):
        return getattr(cfg, "tactile_sensor_type", "") in (
            "xensews",
            "xensews_robotiq",
        )

    def load_robot_and_sensors(self, cfg: BaseTaskCfg):
        cfg = super().load_robot_and_sensors(cfg)
        if self._is_xense_cfg(cfg):
            joint_pos = dict(cfg.robot.robot.init_state.joint_pos)
            joint_pos = apply_xense_wrist_y_alignment(joint_pos)
            cfg.robot.robot.init_state.joint_pos.update(joint_pos)
        return cfg

    def create_actors(self):
        socket_pose = Pose(SOCKET_BASE_POSITION, IDENTITY_Q)
        bulb_pose = self._make_bulb_pose(socket_pose, yaw=0.0, insertion_depth=0.0)

        self.socket = self._actor_manager.add_from_usd_file(
            name="lamp_socket",
            asset_path=SOCKET_ASSET_PATH,
            pose=socket_pose,
            density=1e6,
            keep_constrained=False,
        )
        self.bulb = self._actor_manager.add_from_usd_file(
            name="light_bulb",
            asset_path=BULB_ASSET_PATH,
            pose=bulb_pose,
            density=5e2,
            keep_constrained=False,
        )

    def _reset_actors(self):
        xy_offset = self.rng.uniform(-RESET_XY_NOISE, RESET_XY_NOISE, size=2)
        socket_position = SOCKET_BASE_POSITION + np.array([xy_offset[0], xy_offset[1], 0.0])
        socket_pose = Pose(socket_position, IDENTITY_Q)
        bulb_pose = self._make_bulb_pose(socket_pose, yaw=0.0, insertion_depth=0.0)

        self.socket.set_pose(socket_pose)
        self.bulb.set_pose(bulb_pose)
        self._reset_socket_pose = socket_pose
        self._reset_bulb_pose = bulb_pose

        self.metadata["socket_pose"] = socket_pose.tolist()
        self.metadata["reset_bulb_target_pose"] = bulb_pose.tolist()
        self.metadata["reset_xy_offset"] = xy_offset.tolist()
        self.metadata["initial_pilot_insertion_depth"] = float(
            INITIAL_PILOT_INSERTION_DEPTH
        )
        self.metadata["initial_pilot_insertion_depth_semantics"] = (
            "unthreaded pilot seating only; not counted as completed screw rotation"
        )
        self.metadata["thread_pitch"] = float(THREAD_PITCH)
        self.metadata["turn_angle_each_deg"] = float(np.rad2deg(SCREW_TURN_ANGLE))
        self.metadata["planned_total_rotation_deg"] = float(TARGET_ROTATION_DEG)
        self.metadata["success_rotation_threshold_deg"] = float(SUCCESS_ROTATION_DEG)
        self.metadata["success_insertion_depth_threshold"] = float(
            SUCCESS_INSERTION_DEPTH
        )
        self.metadata["success_insertion_tolerance"] = float(
            SUCCESS_INSERTION_TOLERANCE
        )
        self.metadata["bulb_thread_major_radius"] = float(BULB_THREAD_MAJOR_RADIUS)
        self.metadata["bulb_glass_diameter"] = float(2.0 * BULB_GLASS_MAX_RADIUS)

    def _release_reset_constraints(self):
        self._actor_manager.remove_animate(force=True)

    def build_instruction(self) -> str:
        return TASK_INSTRUCTION

    def pre_move(self):
        self.delay(10)
        self.move(
            self.atom.open_gripper(GRIPPER_OPEN_PERCENT),
            tag="open_gripper_for_bulb",
            delay=False,
        )

    def _make_bulb_pose(
        self,
        socket_pose: Pose,
        yaw: float,
        insertion_depth: float,
    ) -> Pose:
        bulb_bottom_z = (
            socket_pose.p[2]
            + SOCKET_HEIGHT
            - INITIAL_PILOT_INSERTION_DEPTH
            - insertion_depth
        )
        pose = Pose(
            [socket_pose.p[0], socket_pose.p[1], bulb_bottom_z],
            IDENTITY_Q,
        )
        return pose.add_rotation([0.0, 0.0, yaw], coord="local")

    def _record_initial_bulb_pose(self):
        self.initial_bulb_pose = self.bulb.get_pose()
        self.initial_socket_pose = self.socket.get_pose()
        self.metadata["initial_bulb_pose"] = self.initial_bulb_pose.tolist()
        self.metadata["initial_socket_pose"] = self.initial_socket_pose.tolist()

    def _approach_bulb(self):
        self._record_initial_bulb_pose()
        bulb_pose = self.bulb.get_pose()
        grasp_position = bulb_pose.add_bias([0.0, 0.0, GRASP_LOCAL_Z]).p
        # Keep the current wrist orientation for the approach. A constructed
        # exact top-down grasp can put GelSight/Panda near a wrist singularity
        # and causes CuRobo planning failures for this bulb location.
        grasp_q = self._robot_manager.get_gripper_center_pose().q
        pre_grasp_pose = Pose(
            grasp_position + np.array([0.0, 0.0, GRASP_PRE_DISTANCE]),
            grasp_q,
        )
        grasp_pose = Pose(
            grasp_position,
            grasp_q,
        )
        self.metadata["bulb_grasp_local_z"] = float(GRASP_LOCAL_Z)
        self.metadata["bulb_pre_grasp_pose"] = pre_grasp_pose.tolist()
        self.metadata["bulb_grasp_pose"] = grasp_pose.tolist()
        self.metadata["bulb_grasp_pre_distance"] = float(GRASP_PRE_DISTANCE)

        self.move(
            self.atom.move_to_pose(
                self._robot_manager.gripper_center_to_ee(pre_grasp_pose)
            ),
            tag="move_above_light_bulb",
            time_dilation_factor=0.5,
            delay=False,
        )
        self.move(
            self.atom.move_to_pose(
                self._robot_manager.gripper_center_to_ee(grasp_pose)
            ),
            tag="approach_light_bulb",
            time_dilation_factor=0.5,
            delay=False,
        )

    def _close_bulb(self, turn_idx: int):
        depth_threshold = self.get_xense_adaptive_grasp_depth_threshold(
            "xense_bulb_adaptive_grasp_depth_threshold"
        )
        require_both_contacts = self.get_xense_adaptive_grasp_require_both_contacts(
            "xense_bulb_adaptive_grasp_require_both_contacts"
        )
        self.move(
            self.atom.close_gripper(pos=GRIPPER_ADAPTIVE_CLOSE_TARGET),
            tag=f"close_light_bulb_{turn_idx}",
            delay=False,
            gripper_depth_threshold=depth_threshold,
            gripper_require_both_contacts=require_both_contacts,
        )
        self.metadata["gripper_close_percent"] = float(GRIPPER_ADAPTIVE_CLOSE_TARGET)
        self.metadata["gripper_adaptive_depth_threshold"] = (
            None if depth_threshold is None else float(depth_threshold)
        )
        self.metadata["gripper_adaptive_require_both_contacts"] = require_both_contacts
        self.metadata["gripper_close_target_gap_estimate"] = float(
            2.0 * self._robot_manager.gripper_max_qpos * GRIPPER_ADAPTIVE_CLOSE_TARGET
        )
        self.settle_xense_after_close(is_save=False)
        self.record_xense_grasp_debug(f"xense_after_close_light_bulb_{turn_idx}", self.bulb)

    def _release_bulb(self, turn_idx: int):
        self.move(
            self.atom.open_gripper(GRIPPER_RELEASE_PERCENT),
            tag=f"release_light_bulb_{turn_idx}",
            delay=False,
        )

    def _rotate_wrist_joint(
        self,
        delta: float,
        steps: int = WRIST_TURN_STEPS,
        tag: str = "rotate_wrist_joint",
        is_save: bool = True,
    ):
        if self.plan_success is False:
            return False

        self.atom_id += 1
        self.atom_tag = tag
        arm_ids = self._robot_manager._arm_ids
        start_qpos = self._robot_manager.robot.data.joint_pos[0, arm_ids].clone()
        target_qpos = start_qpos.clone()
        target_qpos[-1] += delta

        self.metadata[f"{tag}_start_arm_qpos"] = start_qpos.detach().cpu().tolist()
        self.metadata[f"{tag}_target_arm_qpos"] = target_qpos.detach().cpu().tolist()
        self.metadata[f"{tag}_delta_rad"] = float(delta)
        self.metadata[f"{tag}_delta_deg"] = float(np.rad2deg(delta))
        self.metadata[f"{tag}_steps"] = int(steps)

        last_qpos = start_qpos
        for step_idx in range(1, steps + 1):
            alpha = step_idx / steps
            qpos = start_qpos + (target_qpos - start_qpos) * alpha
            vel = (qpos - last_qpos) / self.cfg.sim.dt
            self._robot_manager.set_arm(qpos, vel)
            self._step(is_save)
            last_qpos = qpos
        self._update_render()
        return True

    def _screw_with_downward_advance(
        self,
        delta_yaw: float,
        advance: float,
        tag: str,
        time_dilation_factor: float = 0.5,
        is_save: bool = True,
    ):
        if self.plan_success is False:
            return False

        self.atom_id += 1
        self.atom_tag = tag
        start_center_pose = self._robot_manager.get_gripper_center_pose()
        target_center_pose = Pose(
            start_center_pose.p + np.array([0.0, 0.0, -advance]),
            start_center_pose.q,
        )
        arm_seq = self._robot_manager.plan_arm(
            self._robot_manager.gripper_center_to_ee(target_center_pose),
            time_dilation_factor=time_dilation_factor,
        )
        if arm_seq["status"] == "Fail":
            self.logger.error(
                f"Arm motion planning failed for coupled screw motion: {tag}"
            )
            self.plan_success = False
            return False

        plan_steps = int(arm_seq["num_steps"])
        steps = max(plan_steps, WRIST_TURN_STEPS)
        arm_ids = self._robot_manager._arm_ids
        current_qpos = self._robot_manager.robot.data.joint_pos[0, arm_ids].clone()
        last_qpos = current_qpos
        plan_start_wrist_qpos = arm_seq["position"][0, -1].clone()

        self.metadata[f"{tag}_start_gripper_center_pose"] = start_center_pose.tolist()
        self.metadata[f"{tag}_target_gripper_center_pose"] = target_center_pose.tolist()
        self.metadata[f"{tag}_advance_m"] = float(advance)
        self.metadata[f"{tag}_delta_rad"] = float(delta_yaw)
        self.metadata[f"{tag}_delta_deg"] = float(np.rad2deg(delta_yaw))
        self.metadata[f"{tag}_plan_steps"] = plan_steps
        self.metadata[f"{tag}_execution_steps"] = steps
        self.metadata[f"{tag}_coupled_screw_motion"] = True

        for step_idx in range(1, steps + 1):
            alpha = step_idx / steps
            arm_idx = min(
                int(np.floor(alpha * max(plan_steps - 1, 0))),
                max(plan_steps - 1, 0),
            )
            qpos = arm_seq["position"][arm_idx].clone()
            qpos[-1] = plan_start_wrist_qpos + delta_yaw * alpha
            vel = (qpos - last_qpos) / self.cfg.sim.dt
            self._robot_manager.set_arm(qpos, vel)
            self._step(is_save)
            last_qpos = qpos
        self._update_render()
        return True

    def _turn_bulb_once(self, turn_idx: int):
        screw_advance = THREAD_PITCH * abs(SCREW_TURN_ANGLE) / (2.0 * np.pi)
        ok = self._screw_with_downward_advance(
            delta_yaw=SCREW_TURN_ANGLE,
            advance=screw_advance,
            tag=f"screw_light_bulb_{turn_idx}",
        )
        if not ok:
            return
        result = self._get_turn_result()
        self._record_turn_result(result, suffix=f"_after_turn_{turn_idx}")
        self._save_metadata()
        print(
            "[screw_light_bulb] "
            f"turn {turn_idx}, yaw: {result['bulb_yaw_signed_deg']:.2f} deg, "
            f"insertion: {result['bulb_insertion_depth']:.4f} m, "
            f"center drift: {result['bulb_center_xy_delta']:.4f} m",
            flush=True,
        )

    def _return_gripper_yaw(self):
        self._rotate_wrist_joint(
            -SCREW_TURN_ANGLE,
            tag="return_gripper_yaw_for_regrasp",
        )

    def _play_once(self):
        self._approach_bulb()
        for turn_idx in range(1, SCREW_TURN_COUNT + 1):
            self._close_bulb(turn_idx)
            self._turn_bulb_once(turn_idx)
            self._release_bulb(turn_idx)
            if turn_idx < SCREW_TURN_COUNT:
                self._return_gripper_yaw()
        self.delay(20, is_save=False)

    @staticmethod
    def _get_actor_yaw_delta_deg(actor_pose: Pose, initial_pose: Pose) -> float:
        init_mat = initial_pose.to_transformation_matrix()
        curr_mat = actor_pose.to_transformation_matrix()
        init_x = init_mat[:3, 0].copy()
        curr_x = curr_mat[:3, 0].copy()
        init_x[2] = 0.0
        curr_x[2] = 0.0
        init_x /= np.linalg.norm(init_x) + 1e-8
        curr_x /= np.linalg.norm(curr_x) + 1e-8
        return float(
            np.rad2deg(
                np.arctan2(
                    np.dot(np.cross(init_x, curr_x), np.array([0.0, 0.0, 1.0])),
                    np.dot(init_x, curr_x),
                )
            )
        )

    def _get_turn_result(self):
        bulb_pose = self.bulb.get_pose()
        initial_pose = getattr(self, "initial_bulb_pose", bulb_pose)
        position_delta = bulb_pose.p - initial_pose.p
        yaw_signed_deg = self._get_actor_yaw_delta_deg(bulb_pose, initial_pose)
        yaw_delta_deg = abs(yaw_signed_deg)
        expected_insertion_depth = THREAD_PITCH * yaw_delta_deg / 360.0
        actual_insertion_depth = max(0.0, -position_delta[2])
        return {
            "bulb_pose": bulb_pose.tolist(),
            "bulb_center_delta": float(np.linalg.norm(position_delta)),
            "bulb_center_xy_delta": float(np.linalg.norm(position_delta[:2])),
            "bulb_vertical_delta": float(position_delta[2]),
            "bulb_insertion_depth": float(actual_insertion_depth),
            "bulb_expected_insertion_depth_from_yaw": float(expected_insertion_depth),
            "bulb_insertion_depth_error": float(
                expected_insertion_depth - actual_insertion_depth
            ),
            "bulb_yaw_signed_deg": yaw_signed_deg,
            "bulb_yaw_delta_deg": yaw_delta_deg,
        }

    def _record_turn_result(self, result, suffix=""):
        for key, value in result.items():
            if isinstance(value, list):
                self.metadata[f"{key}{suffix}"] = value
            else:
                self.metadata[f"{key}{suffix}"] = float(value)

    def check_success(self):
        result = self._get_turn_result()
        self._record_turn_result(result)
        rotation_success = result["bulb_yaw_delta_deg"] >= SUCCESS_ROTATION_DEG
        insertion_success = result["bulb_insertion_depth"] >= SUCCESS_INSERTION_DEPTH
        success = rotation_success and insertion_success
        self.metadata["screw_light_bulb_rotation_success"] = bool(rotation_success)
        self.metadata["screw_light_bulb_insertion_success"] = bool(insertion_success)
        self.metadata["screw_light_bulb_success"] = bool(success)
        return bool(success)
