from ._base_task import *
import numpy as np


is_random = True
CUP_COLORS = ("yellow", "green", "blue")
CUP_BASE_XY = {
    "yellow": (0.36, 0.00),
    "green": (0.48, 0.00),
    "blue": (0.60, 0.00),
}
CUP_BASE_ANCHORS = tuple(CUP_BASE_XY[color] for color in CUP_COLORS)
CUP_ASSET_PATHS = {
    color: f"task_assets/move_cup/cup_{color}.usd" for color in CUP_COLORS
}
XENSE_CUP_PHYSICS_ASSET_PATH = "task_assets/move_cup/cup_physics_proxy.usda"
CUP_BASE_Z = 0.002
CUP_RESET_XY_NOISE = 0.030
CUP_MIN_RESET_XY_DISTANCE = 0.075
MAX_RESET_SAMPLE_ATTEMPTS = 100
PRE_PLACE_XYZ_NOISE = 0.010
PRE_PLACE_Y_DISTANCE = 0.120
PRE_PLACE_WORLD_Z = 0.120
SUCCESS_TABLE_HEIGHT_TOLERANCE = 0.010
SUCCESS_UPRIGHT_SCORE_THRESHOLD = 0.95
SUCCESS_MAX_X_ERROR = 0.030
SUCCESS_GRIPPER_OPEN_RATIO = 0.90
TASK_INSTRUCTION = "Move the blue cup to the left of the yellow cup."


@configclass
class TaskCfg(BaseTaskCfg):
    target_cup: Literal["random", "yellow", "green", "blue"] = "blue"
    reference_cup: Literal["random", "yellow", "green", "blue"] = "yellow"
    placement_side: Literal["random", "left", "right"] = "left"
    # Optional layout override in CUP_COLORS order: (yellow, green, blue).
    # Each value indexes CUP_BASE_ANCHORS. None keeps the original layout.
    cup_base_pose_indices: tuple[int, ...] | None = None
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(
                pos=(0.93, 0.0, 0.25),
                rot=(0.593537, 0.415599, 0.395306, 0.564555),
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
    step_lim = 700


class Task(BaseTask):
    def __init__(
        self,
        cfg: TaskCfg,
        mode: Literal["collect", "eval"] = "collect",
        render_mode: str | None = None,
        **kwargs,
    ):
        if cfg.target_cup not in ("random", *CUP_COLORS):
            raise ValueError(f"target_cup must be 'random' or one of {CUP_COLORS}")
        if cfg.reference_cup not in ("random", *CUP_COLORS):
            raise ValueError(f"reference_cup must be 'random' or one of {CUP_COLORS}")
        if cfg.target_cup != "random" and cfg.target_cup == cfg.reference_cup:
            raise ValueError("target_cup and reference_cup must be different")
        if cfg.placement_side not in ("random", "left", "right"):
            raise ValueError("placement_side must be 'random', 'left', or 'right'")

        cfg.sim.physics_material.dynamic_friction = 3.0
        cfg.sim.physics_material.static_friction = 3.0
        cfg.uipc_sim.contact.default_friction_ratio = 3.0
        super().__init__(cfg, mode, render_mode, **kwargs)

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
            joint_pos["finger_joint"] = cfg.robot.gripper_open_qpos
        else:
            joint_pos = {"panda_finger.*": cfg.robot.gripper_max_qpos}
        cfg.robot.robot.init_state.joint_pos.update(joint_pos)
        return cfg

    def create_actors(self):
        is_xense = self._is_xense_cfg(self.cfg)
        indices = self.cfg.cup_base_pose_indices
        if indices is None:
            indices = tuple(range(len(CUP_COLORS)))
        indices = tuple(int(index) for index in indices)
        if len(indices) != len(CUP_COLORS):
            raise ValueError(
                "cup_base_pose_indices must provide one pose index for each "
                f"cup in {CUP_COLORS}, got {indices}"
            )
        if len(set(indices)) != len(indices):
            raise ValueError(f"cup_base_pose_indices must be unique, got {indices}")
        if any(index < 0 or index >= len(CUP_BASE_ANCHORS) for index in indices):
            raise ValueError(
                "cup_base_pose_indices values must be in "
                f"0..{len(CUP_BASE_ANCHORS) - 1}, got {indices}"
            )
        self.cup_base_pose_assignments = {
            cup_name: (int(index), CUP_BASE_ANCHORS[int(index)])
            for cup_name, index in zip(CUP_COLORS, indices)
        }
        self.cups = {}
        for cup_name in CUP_COLORS:
            _, (x, y) = self.cup_base_pose_assignments[cup_name]
            asset_path = (
                XENSE_CUP_PHYSICS_ASSET_PATH if is_xense else CUP_ASSET_PATHS[cup_name]
            )
            self.cups[cup_name] = self._actor_manager.add_from_usd_file(
                name=f"{cup_name}_cup",
                asset_path=asset_path,
                pose=Pose([x, y, CUP_BASE_Z], (1.0, 0.0, 0.0, 0.0)),
                density=1e3,
                visual_asset_path=CUP_ASSET_PATHS[cup_name] if is_xense else None,
                show_physics_mesh=not is_xense,
                keep_constrained=is_xense,
            )

    def _reset_actors(self):
        self.cup_poses, cup_xy_noises = self._sample_cup_reset_poses()
        for cup_name, cup_pose in self.cup_poses.items():
            self.cups[cup_name].set_pose(cup_pose)

        self.target_cup_name = self.cfg.target_cup
        if self.target_cup_name == "random":
            self.target_cup_name = str(self.rng.choice(CUP_COLORS))
        available_references = [
            name for name in CUP_COLORS if name != self.target_cup_name
        ]
        self.reference_cup_name = self.cfg.reference_cup
        if self.reference_cup_name == "random":
            self.reference_cup_name = str(self.rng.choice(available_references))
        if self.reference_cup_name == self.target_cup_name:
            raise ValueError("Resolved target_cup and reference_cup must be different")

        self.placement_side = self.cfg.placement_side
        if self.placement_side == "random":
            self.placement_side = str(self.rng.choice(("left", "right")))
        self.target_cup = self.cups[self.target_cup_name]
        self.reference_cup = self.cups[self.reference_cup_name]

        self.metadata["is_random"] = bool(is_random)
        self.metadata["target_cup"] = self.target_cup_name
        self.metadata["reference_cup"] = self.reference_cup_name
        self.metadata["placement_side"] = self.placement_side
        self.metadata["cup_base_pose_indices"] = {
            name: int(index)
            for name, (index, _) in self.cup_base_pose_assignments.items()
        }
        self.metadata["cup_xy_noises"] = cup_xy_noises
        self.metadata["cup_poses"] = {
            name: pose.tolist() for name, pose in self.cup_poses.items()
        }

    def _release_reset_constraints(self):
        self._actor_manager.remove_animate(force=True)

    def _sample_cup_reset_poses(self):
        for _ in range(MAX_RESET_SAMPLE_ATTEMPTS):
            noises = {
                name: self._random_vec(CUP_RESET_XY_NOISE, size=2)
                for name in CUP_COLORS
            }
            poses = {
                name: Pose(
                    [
                        self.cup_base_pose_assignments[name][1][0] + noises[name][0],
                        self.cup_base_pose_assignments[name][1][1] + noises[name][1],
                        CUP_BASE_Z,
                    ],
                    (1.0, 0.0, 0.0, 0.0),
                )
                for name in CUP_COLORS
            }
            if all(
                np.linalg.norm(poses[a].p[:2] - poses[b].p[:2])
                >= CUP_MIN_RESET_XY_DISTANCE
                for i, a in enumerate(CUP_COLORS)
                for b in CUP_COLORS[i + 1 :]
            ):
                return poses, {
                    name: noise.tolist() for name, noise in noises.items()
                }
        raise RuntimeError("Could not sample non-overlapping cup poses")

    def build_instruction(self) -> str:
        return (
            f"Move the {self.target_cup_name} cup to the {self.placement_side} "
            f"of the {self.reference_cup_name} cup."
        )

    def pre_move(self):
        initial_settle_steps = 10
        if self._is_xense_cfg(self.cfg):
            initial_settle_steps = int(
                getattr(
                    self.cfg,
                    "xense_cup_initial_settle_steps",
                    getattr(self.cfg, "xense_initial_settle_steps", 1),
                )
            )
        if initial_settle_steps > 0:
            self.delay(initial_settle_steps)
        self.metadata["gripper_open_qpos"] = float(
            self._robot_manager.gripper_max_qpos
        )
        self.metadata["gripper_open_ratio"] = 1.0
        self.metadata["gripper_qpos_after_init_delay"] = float(
            self._robot_manager.get_gripper_qpos()
        )
        if self._is_xense_cfg(self.cfg):
            self._approach_target_cup()
            self._update_render()

    def _random_vec(self, scale: float, size: int = 3):
        if not is_random:
            return np.zeros(size)
        return self.rng.uniform(-scale, scale, size=size)

    def _random_scalar(self, scale: float):
        if not is_random:
            return 0.0
        return float(self.rng.uniform(-scale, scale))

    def _placement_y_direction(self):
        return 1.0 if self.placement_side == "left" else -1.0

    def _record_target_cup_shape(self, label: str):
        vertices = np.asarray(self.target_cup.vertices, dtype=np.float64).copy()
        centered = vertices - vertices.mean(axis=0)
        covariance = centered.T @ centered / len(centered)
        principal_spread = np.sqrt(np.maximum(np.linalg.eigvalsh(covariance), 0.0))

        cup_pose = self.target_cup.get_pose()
        local_vertices = (vertices - cup_pose.p) @ cup_pose.R
        shape = {
            "point_count": int(len(vertices)),
            "local_extent": np.ptp(local_vertices, axis=0).tolist(),
            "principal_spread": principal_spread.tolist(),
        }
        reference = getattr(self, "_target_cup_shape_reference_vertices", None)
        if reference is not None:
            reference_centered = reference - reference.mean(axis=0)
            reference_covariance = (
                reference_centered.T @ reference_centered / len(reference_centered)
            )
            reference_spread = np.sqrt(
                np.maximum(np.linalg.eigvalsh(reference_covariance), 0.0)
            )
            spread_ratio = principal_spread / np.maximum(reference_spread, 1e-12)
            u, _, vh = np.linalg.svd(reference_centered.T @ centered)
            correction = np.eye(3)
            correction[-1, -1] = np.sign(np.linalg.det(u @ vh))
            aligned_reference = reference_centered @ (u @ correction @ vh)
            rms_deformation = np.sqrt(
                np.mean(np.sum((aligned_reference - centered) ** 2, axis=1))
            )
            reference_rms_radius = np.sqrt(
                np.mean(np.sum(reference_centered**2, axis=1))
            )
            shape["principal_spread_ratio"] = spread_ratio.tolist()
            shape["min_principal_spread_ratio"] = float(np.min(spread_ratio))
            shape["normalized_nonrigid_error"] = float(
                rms_deformation / max(reference_rms_radius, 1e-12)
            )

        self.metadata[f"target_cup_shape_{label}"] = shape
        return shape

    def _move_held_cup(self, target_pose, tag, metadata_prefix):
        if self._is_xense_cfg(self.cfg):
            self.move_actor_by_world_displacement_to_position(
                self.target_cup,
                target_pose.p,
                tag=tag,
                metadata_prefix=metadata_prefix,
            )
        else:
            self.move(
                self.atom.place_actor(
                    self.target_cup,
                    target_pose=target_pose,
                    pre_dis=0.0,
                    dis=0.0,
                    is_open=False,
                    constrain="free",
                ),
                tag=tag,
                time_dilation_factor=0.5,
            )

    def _fully_open_gripper_for_release(self, tag):
        """Open to the commanded endpoint without contact-based early stop."""
        use_adaptive_grasp = self.cfg.use_adaptive_grasp
        try:
            self.cfg.use_adaptive_grasp = False
            return self.move(self.atom.open_gripper(1.0), tag=tag)
        finally:
            self.cfg.use_adaptive_grasp = use_adaptive_grasp

    def _approach_target_cup(self):
        is_xense = self._is_xense_cfg(self.cfg)
        cup_pose = self.target_cup.get_pose()
        current_eef_pose = self._robot_manager.get_ee_pose()

        grasp_height_bias = self.get_xense_grasp_height_bias(
            "xense_cup_grasp_height_bias"
        )
        grasp_pos = cup_pose.p + np.array(
            [0.0, 0.0, 0.0925 - 0.5 * 0.011 - 0.010 + grasp_height_bias]
        )
        grasp_yaw_noise = (
            0.0 if is_xense else self._random_scalar(np.deg2rad(10.0))
        )
        grasp_q = Pose(current_eef_pose.p, current_eef_pose.q).add_rotation(
            [0.0, 0.0, grasp_yaw_noise]
        ).q
        pre_grasp_gripper_center_pose = Pose(
            grasp_pos + np.array([0.0, 0.0, 0.050]), grasp_q
        )
        pre_grasp_ee_pose = self._robot_manager.gripper_center_to_ee(
            pre_grasp_gripper_center_pose
        )
        self.metadata["grasp_yaw_noise_rad"] = float(grasp_yaw_noise)
        self.metadata["grasp_yaw_noise_deg"] = float(np.rad2deg(grasp_yaw_noise))
        self.metadata["pre_grasp_gripper_center_pose"] = (
            pre_grasp_gripper_center_pose.tolist()
        )
        self.metadata["pre_grasp_ee_pose"] = pre_grasp_ee_pose.tolist()
        self.move(
            self.atom.move_to_pose(pre_grasp_ee_pose),
            tag=f"move_above_{self.target_cup_name}_cup",
            time_dilation_factor=0.5,
        )

        grasp_noise_scale = 0.001 if is_xense else 0.005
        grasp_noise = self._random_vec(grasp_noise_scale)
        grasp_pose = Pose(grasp_pos + grasp_noise, grasp_q)
        self.metadata["grasp_noise_scale"] = float(grasp_noise_scale)
        self.metadata["grasp_noise"] = grasp_noise.tolist()
        self.metadata["target_cup_grasp_pose"] = grasp_pose.tolist()
        cid = self.target_cup.register_point(grasp_pose, type="contact")
        self.move(
            self.atom.grasp_actor(
                self.target_cup,
                contact_point_id=cid,
                pre_dis=0.0,
                dis=0.0,
                is_close=False,
            ),
            tag=f"approach_{self.target_cup_name}_cup",
            time_dilation_factor=0.5,
        )
        self.record_xense_grasp_debug("xense_after_approach_target_cup", self.target_cup)

    def _close_target_cup(self):
        is_xense = self._is_xense_cfg(self.cfg)
        grasp_height_bias = self.get_xense_grasp_height_bias(
            "xense_cup_grasp_height_bias"
        )
        close_percent = self.get_xense_close_percent("xense_cup_close_percent")
        self.move(
            self.atom.close_gripper(pos=close_percent),
            tag=f"close_{self.target_cup_name}_cup",
            gripper_depth_threshold=self.get_xense_adaptive_grasp_depth_threshold(
                "xense_cup_adaptive_grasp_depth_threshold"
            ),
            gripper_require_both_contacts=(
                self.get_xense_adaptive_grasp_require_both_contacts(
                    "xense_cup_adaptive_grasp_require_both_contacts"
                )
            ),
        )
        self.settle_xense_after_close(is_save=False)
        self.record_xense_grasp_debug("xense_after_close_target_cup", self.target_cup)
        self._target_cup_shape_after_close = self._record_target_cup_shape(
            "after_close"
        )
        self.metadata["grasp_height_bias"] = float(grasp_height_bias)
        self.metadata["gripper_close_percent"] = float(close_percent)

    def _play_once(self):
        is_xense = self._is_xense_cfg(self.cfg)
        self._target_cup_shape_reference_vertices = np.asarray(
            self.target_cup.vertices, dtype=np.float64
        ).copy()
        self._record_target_cup_shape("before_close")
        if is_xense:
            self._close_target_cup()
        else:
            self._approach_target_cup()
            self._close_target_cup()

        cup_pose = self.target_cup.get_pose()
        lift_noise = self._random_vec(0.010)
        lifted_pose = Pose(
            cup_pose.p + np.array([0.0, 0.0, 0.120]) + lift_noise,
            cup_pose.q,
        )
        self.metadata["lift_noise"] = lift_noise.tolist()
        self.metadata["lifted_target_cup_pose"] = lifted_pose.tolist()
        self._move_held_cup(
            lifted_pose,
            tag=f"lift_{self.target_cup_name}_cup",
            metadata_prefix="xense_target_cup_lift_path",
        )

        reference_pose = self.reference_cup.get_pose()
        pre_place_noise = self._random_vec(PRE_PLACE_XYZ_NOISE)
        pre_place_pose = Pose(
            np.array(
                [
                    reference_pose.p[0],
                    reference_pose.p[1]
                    + self._placement_y_direction() * PRE_PLACE_Y_DISTANCE,
                    PRE_PLACE_WORLD_Z,
                ]
            )
            + pre_place_noise,
            self.target_cup.get_pose().q,
        )
        self.metadata["pre_place_noise"] = pre_place_noise.tolist()
        self.metadata["reference_cup_pose_for_pre_place"] = reference_pose.tolist()
        self.metadata["pre_place_target_cup_pose"] = pre_place_pose.tolist()
        self._move_held_cup(
            pre_place_pose,
            tag=f"move_{self.target_cup_name}_cup_to_pre_place",
            metadata_prefix="xense_target_cup_pre_place_path",
        )

        pre_place_pose = self.target_cup.get_pose()
        reference_pose = self.reference_cup.get_pose()
        placed_pose = Pose(
            [pre_place_pose.p[0], pre_place_pose.p[1], reference_pose.p[2]],
            pre_place_pose.q,
        )
        self.metadata["reference_cup_pose_for_place_height"] = reference_pose.tolist()
        self.metadata["placed_target_cup_pose"] = placed_pose.tolist()
        self._move_held_cup(
            placed_pose,
            tag=f"place_{self.target_cup_name}_cup_down",
            metadata_prefix="xense_target_cup_place_path",
        )
        self.record_xense_grasp_debug("xense_before_release_target_cup", self.target_cup)
        self._fully_open_gripper_for_release(
            tag=f"release_{self.target_cup_name}_cup",
        )
        self.delay(5, is_save=False)

    def check_success(self):
        target_pose = self.target_cup.get_pose()
        reference_pose = self.reference_cup.get_pose()
        target_shape_final = self._record_target_cup_shape("final")
        gripper_qpos = float(self._robot_manager.get_gripper_qpos())
        gripper_open_percentage = float(
            self._robot_manager.get_gripper_percentage()
        )

        target_table_height_error = float(abs(target_pose.p[2] - reference_pose.p[2]))
        target_on_table_ok = (
            target_table_height_error <= SUCCESS_TABLE_HEIGHT_TOLERANCE
        )
        target_up_dir = target_pose.to_transformation_matrix()[:3, :3] @ np.array(
            [0.0, 0.0, 1.0]
        )
        target_upright_score = float(
            np.dot(target_up_dir, np.array([0.0, 0.0, 1.0]))
        )
        target_upright_ok = target_upright_score > SUCCESS_UPRIGHT_SCORE_THRESHOLD
        target_y_gap = float(target_pose.p[1] - reference_pose.p[1])
        signed_y_gap = self._placement_y_direction() * target_y_gap
        target_on_selected_side_ok = signed_y_gap > 0.0
        target_x_error = float(abs(target_pose.p[0] - reference_pose.p[0]))
        target_x_ok = target_x_error <= SUCCESS_MAX_X_ERROR
        gripper_open_ok = gripper_open_percentage >= SUCCESS_GRIPPER_OPEN_RATIO

        is_xense = self._is_xense_cfg(self.cfg)
        min_shape_ratio = float(
            getattr(self.cfg, "xense_cup_min_principal_ratio", 0.90)
        )
        max_nonrigid_error = float(
            getattr(self.cfg, "xense_cup_max_nonrigid_error", 0.08)
        )
        shape_samples = (
            getattr(self, "_target_cup_shape_after_close", {}),
            target_shape_final,
        )
        target_shape_ok = (not is_xense) or all(
            sample.get("min_principal_spread_ratio", 0.0) >= min_shape_ratio
            and sample.get("normalized_nonrigid_error", float("inf"))
            <= max_nonrigid_error
            for sample in shape_samples
        )

        self.metadata["final_target_cup_pose"] = target_pose.tolist()
        self.metadata["final_reference_cup_pose"] = reference_pose.tolist()
        self.metadata["success_target_up_dir"] = target_up_dir.tolist()
        self.metadata["success_target_upright_score"] = target_upright_score
        self.metadata["target_cup_shape_thresholds"] = {
            "min_principal_spread_ratio": min_shape_ratio,
            "max_nonrigid_error": max_nonrigid_error,
        }
        self.metadata["success_checks"] = {
            "target_cup": self.target_cup_name,
            "reference_cup": self.reference_cup_name,
            "placement_side": self.placement_side,
            "target_on_table_ok": bool(target_on_table_ok),
            "target_table_height_error": target_table_height_error,
            "target_upright_ok": bool(target_upright_ok),
            "target_on_selected_side_ok": bool(target_on_selected_side_ok),
            "target_y_gap": target_y_gap,
            "signed_y_gap": signed_y_gap,
            "target_x_ok": bool(target_x_ok),
            "target_x_error": target_x_error,
            "gripper_open_ok": bool(gripper_open_ok),
            "gripper_open_percentage": gripper_open_percentage,
            "gripper_open_ratio_threshold": SUCCESS_GRIPPER_OPEN_RATIO,
            "gripper_qpos": gripper_qpos,
            "target_shape_ok": bool(target_shape_ok),
        }
        return bool(
            target_on_table_ok
            and target_upright_ok
            and target_on_selected_side_ok
            and target_x_ok
            and gripper_open_ok
            and target_shape_ok
        )
