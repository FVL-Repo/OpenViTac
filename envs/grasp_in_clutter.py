from ._base_task import *
import numpy as np


BLOCK_HEIGHT = 0.0400
BLOCK_XY_NOISE = (0.020, 0.020, 0.0)
GRASP_ROTATE_NOISE = np.deg2rad(10.0)
GRASP_HEIGHT = BLOCK_HEIGHT * 0.5
GRASP_HEIGHT_NOISE = 0.003
LIFT_HEIGHT = 0.1000
SUCCESS_MIN_LIFT = 0.0500
SUCCESS_MAX_LIFT = 0.1500
XENSE_BLOCK_Z_CLEARANCE = 0.0

BLOCK_BASE_POSES = (
    Pose([0.44, 0.00, 0.002], [1, 0, 0, 0]),
    Pose([0.36, -0.09, 0.002], [1, 0, 0, 0]),
    Pose([0.36, 0.09, 0.002], [1, 0, 0, 0]),
    Pose([0.44, -0.12, 0.002], [1, 0, 0, 0]),
    Pose([0.44, 0.12, 0.002], [1, 0, 0, 0]),
    Pose([0.54, -0.08, 0.002], [1, 0, 0, 0]),
    Pose([0.53, 0.08, 0.002], [1, 0, 0, 0]),
)

BLOCK_SPECS = (
    {
        "name": "block_blue_half_cylinder",
        "description": "blue half cylinder",
        "asset_path": "task_assets/grasp_in_clutter/block_blue_half_cylinder.usd",
    },
    {
        "name": "block_blue_quarter_cylinder",
        "description": "blue quarter cylinder",
        "asset_path": "task_assets/grasp_in_clutter/block_blue_quarter_cylinder.usd",
    },
    {
        "name": "block_blue_star_prism",
        "description": "blue star prism",
        "asset_path": "task_assets/grasp_in_clutter/block_blue_star_prism.usd",
    },
    {
        "name": "block_red_ellipse_cylinder",
        "description": "red ellipse cylinder",
        "asset_path": "task_assets/grasp_in_clutter/block_red_ellipse_cylinder.usd",
    },
    {
        "name": "block_red_hexagonal_prism",
        "description": "red hexagonal prism",
        "asset_path": "task_assets/grasp_in_clutter/block_red_hexagonal_prism.usd",
    },
    {
        "name": "block_yellow_cylinder",
        "description": "yellow cylinder",
        "asset_path": "task_assets/grasp_in_clutter/block_yellow_cylinder.usd",
    },
    {
        "name": "block_yellow_triangular_prism",
        "description": "yellow triangular prism",
        "asset_path": "task_assets/grasp_in_clutter/block_yellow_triangular_prism.usd",
    },
)
TARGET_BLOCKS = tuple(spec["name"] for spec in BLOCK_SPECS)
DEFAULT_TARGET_BLOCK = "block_yellow_cylinder"
TASK_INSTRUCTION = "Grasp the blue half cylinder from the clutter and lift it up."
TASK_INITIAL_JOINT_POS = {
    "panda_joint1": -0.010809095,
    "panda_joint2": 0.096037410,
    "panda_joint3": 0.000734462,
    "panda_joint4": -2.433035851,
    "panda_joint5": 0.035354517,
    "panda_joint6": 2.500859022,
    "panda_joint7": 0.741,
}


@configclass
class TaskCfg(BaseTaskCfg):
    target_block: str = DEFAULT_TARGET_BLOCK
    block_base_pose_indices: tuple[int, ...] = tuple(range(len(BLOCK_SPECS)))
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(pos=(0.87, 0.0, 0.20), rot=(0.579228, 0.405580, 0.405580, 0.579228), convention="opengl"),
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.5, focus_distance=1.0, horizontal_aperture=2.4, clipping_range=(0.1, 100.0)
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
            update_period=1/120,
        ),
    ]
    step_lim = 300


class Task(BaseTask):
    def __init__(self, cfg: TaskCfg, mode: Literal["collect", "eval"] = "collect", render_mode: str | None = None, **kwargs):
        if cfg.target_block not in TARGET_BLOCKS and cfg.target_block != "random":
            raise ValueError(f"target_block must be one of {TARGET_BLOCKS} or 'random'")
        self.configured_target_block = cfg.target_block
        # clutter 抓取依赖稳定摩擦；提高刚体材质和 UIPC 接触摩擦，减少夹取目标时打滑。
        cfg.sim.physics_material.dynamic_friction = 2.5
        cfg.sim.physics_material.static_friction = 2.5
        cfg.uipc_sim.contact.default_friction_ratio = 2.5
        super().__init__(cfg, mode, render_mode, **kwargs)

    def load_robot_and_sensors(self, cfg: BaseTaskCfg):
        cfg = super().load_robot_and_sensors(cfg)
        joint_pos = TASK_INITIAL_JOINT_POS
        if getattr(cfg, "tactile_sensor_type", "") in ("xensews", "xensews_robotiq"):
            joint_pos = apply_xense_wrist_y_alignment(joint_pos)
        cfg.robot.robot.init_state.joint_pos.update(joint_pos)
        return cfg

    def create_actors(self):
        pose_indices = tuple(int(index) for index in self.cfg.block_base_pose_indices)
        if sorted(pose_indices) != list(range(len(BLOCK_BASE_POSES))):
            raise ValueError(
                "block_base_pose_indices must be a permutation of "
                f"0..{len(BLOCK_BASE_POSES) - 1}, got {pose_indices}"
            )
        self.initial_pose_assignments = {
            spec["name"]: (int(index), BLOCK_BASE_POSES[int(index)])
            for spec, index in zip(BLOCK_SPECS, pose_indices)
        }
        self.wooden_blocks = {}
        self.wooden_block_specs = {spec["name"]: spec for spec in BLOCK_SPECS}
        is_xense = getattr(self.cfg, "tactile_sensor_type", "") in (
            "xensews",
            "xensews_robotiq",
        )
        # Hold XSense blocks at their sampled reset poses only during reset settling.
        for spec in BLOCK_SPECS:
            _, initial_pose = self.initial_pose_assignments[spec["name"]]
            actor = self._actor_manager.add_from_usd_file(
                name=spec["name"],
                asset_path=spec["asset_path"],
                pose=initial_pose,
                density=1e3,
                keep_constrained=is_xense,
            )
            self.wooden_blocks[spec["name"]] = actor

    def _reset_actors(self):
        target_name = self.configured_target_block
        if target_name == "random":
            target_name = str(self.rng.choice(TARGET_BLOCKS))
        self.target_block_name = target_name
        self.target_block = self.wooden_blocks[target_name]
        self.metadata["target_description"] = self.wooden_block_specs[target_name]["description"]
        self.metadata["target_block_name"] = target_name
        self.metadata["block_poses"] = {}
        self.metadata["block_xy_noise"] = {}
        self.metadata["block_base_pose_indices"] = {}
        self._xense_initial_gripper_ready = False

        for name, actor in self.wooden_blocks.items():
            pose_index, base_pose = self.initial_pose_assignments[name]
            offset = self.create_noise(list(BLOCK_XY_NOISE))
            pose = base_pose.add_offset(offset)
            if getattr(self.cfg, "tactile_sensor_type", "") in (
                "xensews",
                "xensews_robotiq",
            ):
                pose = pose.add_bias([0.0, 0.0, XENSE_BLOCK_Z_CLEARANCE])
            actor.set_pose(pose)

            self.metadata["block_xy_noise"][name] = offset.p.tolist()
            self.metadata["block_poses"][name] = pose.tolist()
            self.metadata["block_base_pose_indices"][name] = int(pose_index)
            if name == target_name:
                self.target_initial_pose = pose

    def build_instruction(self) -> str:
        description = self.wooden_block_specs[self.target_block_name]["description"]
        return f"Grasp the {description} from the clutter and lift it up."

    def _release_reset_constraints(self):
        self._actor_manager.remove_animate(force=True)

    def pre_move(self):
        # 正式动作前等待物理状态稳定，再打开夹爪准备从目标物体上方接近。
        is_xense = getattr(self.cfg, "tactile_sensor_type", "") in (
            "xensews",
            "xensews_robotiq",
        )
        initial_settle_steps = 10
        if is_xense:
            initial_settle_steps = int(
                getattr(
                    self.cfg,
                    "xense_half_cylinder_initial_settle_steps",
                    getattr(self.cfg, "xense_initial_settle_steps", 1),
                )
            )
        if initial_settle_steps > 0:
            self.delay(initial_settle_steps)
        if not is_xense:
            self.move(self.atom.open_gripper(0.6), tag="open_gripper_for_policy")

    def _setup_xense_initial_gripper(self):
        if self._xense_initial_gripper_ready:
            return
        self.move(
            self.atom.open_gripper(0.7),
            tag="setup_open_gripper_for_policy",
            is_save=False,
        )
        self.delay(20, is_save=False)
        self._xense_initial_gripper_ready = True

    def _build_grasp_plan(self):
        is_xense = getattr(self.cfg, "tactile_sensor_type", "") in (
            "xensews",
            "xensews_robotiq",
        )
        target_pose = self.target_block.get_pose()
        # 以目标半圆柱当前位姿为基准，在半高附近构造抓取点，并绕局部 y 轴加入少量随机旋转。
        grasp_rotate = self.rng.uniform(-GRASP_ROTATE_NOISE, GRASP_ROTATE_NOISE)
        grasp_height_bias = self.get_xense_grasp_height_bias(
            "xense_half_cylinder_grasp_height_bias"
        )
        grasp_world_y_bias = self.get_xense_grasp_height_bias(
            "xense_half_cylinder_grasp_world_y_bias"
        )
        grasp_height = (
            GRASP_HEIGHT
            + self.rng.uniform(-GRASP_HEIGHT_NOISE, GRASP_HEIGHT_NOISE)
            + grasp_height_bias
        )
        if is_xense:
            # A half cylinder can settle on its curved side, making its local Z
            # axis horizontal. The thicker Robotiq fingertips must approach it
            # vertically from a world-frame waypoint instead of following that
            # local axis into the table and neighboring clutter.
            target_mat = target_pose.to_transformation_matrix()
            gripper_up = target_mat[:3, 0].copy()
            gripper_up[2] = 0.0
            if np.linalg.norm(gripper_up) < 1e-6:
                gripper_up = np.array([1.0, 0.0, 0.0])
            yaw_cos = np.cos(grasp_rotate)
            yaw_sin = np.sin(grasp_rotate)
            gripper_up = np.array([
                yaw_cos * gripper_up[0] - yaw_sin * gripper_up[1],
                yaw_sin * gripper_up[0] + yaw_cos * gripper_up[1],
                0.0,
            ])
            grasp_position = target_pose.p.copy()
            grasp_position += np.array([
                0.0,
                grasp_world_y_bias,
                grasp_height,
            ])
            grasp_pose = construct_grasp_pose(
                grasp_position,
                np.array([0.0, 0.0, 1.0]),
                gripper_up,
            )
        else:
            grasp_target_pose = (
                target_pose
                .add_bias([0.0, 0.0, grasp_height])
                .add_bias([0.0, grasp_world_y_bias, 0.0], coord="world")
                .add_rotation([0.0, grasp_rotate, 0.0])
            )
            target_mat = grasp_target_pose.to_transformation_matrix()
            # construct_grasp_pose uses the contact point, approach axis, and
            # gripper lateral axis to build the final grasp pose.
            grasp_pose = construct_grasp_pose(
                grasp_target_pose.p,
                target_mat[:3, 2],
                target_mat[:3, 0],
            )
        # 将抓取点注册到目标 actor 局部坐标系，后续 grasp_actor 会按该 contact point 规划接近动作。
        contact_point_id = self.target_block.register_point(grasp_pose, type="contact")

        approach_actions = self.atom.grasp_actor(
            self.target_block,
            contact_point_id=contact_point_id,
            is_close=False,
            pre_dis=0.05,
        )
        approach_target_pose = self._robot_manager.ee_to_gripper_center(
            approach_actions[0].target_pose
        )
        plan = {
            "is_xense": is_xense,
            "grasp_rotate": grasp_rotate,
            "grasp_height_bias": grasp_height_bias,
            "grasp_world_y_bias": grasp_world_y_bias,
            "grasp_height": grasp_height,
            "grasp_pose": grasp_pose,
            "approach_actions": approach_actions,
            "approach_target_pose": approach_target_pose,
        }
        if is_xense:
            pregrasp_clearance = float(
                getattr(self.cfg, "xense_half_cylinder_pregrasp_clearance", 0.08)
            )
            pregrasp_pose = Pose(
                approach_target_pose.p + np.array([0.0, 0.0, pregrasp_clearance]),
                approach_target_pose.q,
            )
            self.metadata["xense_half_cylinder_pregrasp_clearance"] = pregrasp_clearance
            self.metadata["xense_half_cylinder_pregrasp_pose"] = pregrasp_pose.tolist()
            plan["pregrasp_pose"] = pregrasp_pose
        return plan

    def prepare_initial_state(self):
        is_xense = getattr(self.cfg, "tactile_sensor_type", "") in (
            "xensews",
            "xensews_robotiq",
        )
        if not is_xense:
            return
        self._setup_xense_initial_gripper()

    def _grasp_target(self):
        plan = self._build_grasp_plan()
        is_xense = plan["is_xense"]
        grasp_rotate = plan["grasp_rotate"]
        grasp_height_bias = plan["grasp_height_bias"]
        grasp_world_y_bias = plan["grasp_world_y_bias"]
        grasp_height = plan["grasp_height"]
        grasp_pose = plan["grasp_pose"]
        approach_actions = plan["approach_actions"]
        approach_target_pose = plan["approach_target_pose"]
        if is_xense:
            pregrasp_pose = plan["pregrasp_pose"]
            self.move(
                [Action(
                    "move",
                    target_pose=self._robot_manager.gripper_center_to_ee(pregrasp_pose),
                )],
                tag=f"pregrasp_{self.target_block_name}",
                delay=False,
            )
        if self.plan_success:
            self.move(approach_actions, tag=f"approach_{self.target_block_name}")

        if is_xense:
            actual_gripper_pose = self._robot_manager.get_gripper_center_pose()
            target_quat = approach_target_pose.q / np.linalg.norm(approach_target_pose.q)
            actual_quat = actual_gripper_pose.q / np.linalg.norm(actual_gripper_pose.q)
            quat_dot = float(np.clip(
                np.abs(np.dot(target_quat, actual_quat)),
                0.0,
                1.0,
            ))
            self.metadata["xense_approach_half_cylinder_target_gripper_center_pose"] = (
                approach_target_pose.tolist()
            )
            self.metadata["xense_approach_half_cylinder_position_error_m"] = float(
                np.linalg.norm(actual_gripper_pose.p - approach_target_pose.p)
            )
            self.metadata["xense_approach_half_cylinder_orientation_error_deg"] = float(
                np.rad2deg(2.0 * np.arccos(quat_dot))
            )
        self.record_xense_grasp_debug(
            "xense_after_approach_half_cylinder",
            self.target_block,
        )

        close_percent = self.get_xense_close_percent(
            "xense_half_cylinder_close_percent"
        )
        self.move(
            self.atom.close_gripper(pos=close_percent),
            tag=f"close_{self.target_block_name}",
            gripper_depth_threshold=self.get_xense_adaptive_grasp_depth_threshold(
                "xense_half_cylinder_adaptive_grasp_depth_threshold"
            ),
            gripper_require_both_contacts=self.get_xense_adaptive_grasp_require_both_contacts(
                "xense_half_cylinder_adaptive_grasp_require_both_contacts"
            ),
        )
        self.settle_xense_after_close(is_save=False)
        self.record_xense_grasp_debug(
            "xense_after_close_half_cylinder",
            self.target_block,
        )

        # 保存抓取随机量和最终抓取姿态，便于复现失败样本或分析抓取分布。
        self.metadata["grasp_rotate_rad"] = float(grasp_rotate)
        self.metadata["grasp_rotate_deg"] = float(np.rad2deg(grasp_rotate))
        self.metadata["grasp_height"] = float(grasp_height)
        self.metadata["grasp_height_bias"] = float(grasp_height_bias)
        self.metadata["grasp_world_y_bias"] = float(grasp_world_y_bias)
        self.metadata["gripper_close_percent"] = float(close_percent)
        self.metadata["grasp_pose"] = grasp_pose.tolist()

    def _play_once(self):
        self._grasp_target()
        # 抓住目标后竖直上提 10cm；该任务只验证目标是否被稳定提起到期望高度范围。
        self.move(self.atom.move_by_displacement(z=LIFT_HEIGHT), tag=f"lift_{self.target_block_name}")
        # 上提后等待但不保存等待帧，避免纯稳定过程混入动作监督数据。
        self.delay(20, is_save=False)

    def _get_success_diagnostics(self):
        target_pose = self.target_block.get_pose()
        # 用当前目标高度减去 reset 后初始高度，得到真实上提距离。
        lifted_height = target_pose.p[2] - self.target_initial_pose.p[2]
        height_ok = bool(SUCCESS_MIN_LIFT <= lifted_height <= SUCCESS_MAX_LIFT)

        return {
            "target_block_name": self.target_block_name,
            "target_initial_pose": self.target_initial_pose.tolist(),
            "target_final_pose": target_pose.tolist(),
            "lifted_height": float(lifted_height),
            "success_min_lift": float(SUCCESS_MIN_LIFT),
            "success_max_lift": float(SUCCESS_MAX_LIFT),
            "height_ok": height_ok,
        }

    def check_success(self):
        # 成功条件只看目标半圆柱是否被提到 9-11cm 的范围内，诊断信息写入 metadata。
        diagnostics = self._get_success_diagnostics()
        self.metadata["success_diagnostics"] = diagnostics
        return diagnostics["height_ok"]

    def get_rl_metrics(self):
        target_pose = self.target_block.get_pose()
        gripper_pose = self._robot_manager.get_gripper_center_pose()
        lifted_height = float(target_pose.p[2] - self.target_initial_pose.p[2])
        gripper_distance = float(np.linalg.norm(target_pose.p - gripper_pose.p))
        gripper_qpos = float(self._robot_manager.get_gripper_qpos())
        policy_open_qpos = 0.5 * float(self.cfg.robot.gripper_max_qpos)
        closure = float(np.clip(
            (policy_open_qpos - gripper_qpos) / max(policy_open_qpos - 0.008, 1e-4),
            0.0,
            1.0,
        ))
        proximity = float(np.exp(-np.square(gripper_distance / 0.08)))
        grasp_proxy = proximity * closure
        if getattr(self, "policy_step_count", 0) == 0:
            self._rl_max_lifted_height = lifted_height
        else:
            self._rl_max_lifted_height = max(
                float(getattr(self, "_rl_max_lifted_height", lifted_height)),
                lifted_height,
            )
        has_been_lifted = self._rl_max_lifted_height >= 0.02
        dropped_after_lift = has_been_lifted and (
            lifted_height < self._rl_max_lifted_height - 0.02
            or gripper_distance > 0.18
        )
        return {
            "lifted_height": lifted_height,
            "normalized_lift": float(lifted_height / SUCCESS_MIN_LIFT),
            "gripper_distance": gripper_distance,
            "gripper_qpos": gripper_qpos,
            "proximity": proximity,
            "grasp_proxy": grasp_proxy,
            "max_lifted_height": float(self._rl_max_lifted_height),
            "dropped": bool(lifted_height < -0.02 or dropped_after_lift),
        }

    def compute_rl_reward(self, previous_metrics, current_metrics, action, success):
        previous_lift = float(previous_metrics.get("normalized_lift", 0.0))
        current_lift = float(current_metrics["normalized_lift"])
        previous_distance = float(
            previous_metrics.get("gripper_distance", current_metrics["gripper_distance"])
        )
        current_distance = float(current_metrics["gripper_distance"])
        previous_grasp = float(
            previous_metrics.get("grasp_proxy", current_metrics["grasp_proxy"])
        )
        current_grasp = float(current_metrics["grasp_proxy"])
        progress_reward = 5.0 * (current_lift - previous_lift)
        approach_progress = float(
            np.clip(previous_distance - current_distance, -0.02, 0.02)
        )
        approach_reward = 25.0 * approach_progress
        proximity_reward = 0.02 * float(current_metrics["proximity"])
        grasp_progress_reward = 2.0 * (current_grasp - previous_grasp)
        grasp_hold_reward = 0.02 * current_grasp
        height_reward = 0.05 * float(np.clip(current_lift, -1.0, 1.25))
        control_penalty = 1e-3 * float(np.mean(np.square(action)))
        reward = (
            progress_reward
            + approach_reward
            + proximity_reward
            + grasp_progress_reward
            + grasp_hold_reward
            + height_reward
            - control_penalty
        )
        if success:
            reward += 10.0
        if current_metrics["dropped"]:
            reward -= 5.0
        return float(reward)

    def check_rl_early_stop(self, metrics):
        return bool(metrics.get("dropped", False))
