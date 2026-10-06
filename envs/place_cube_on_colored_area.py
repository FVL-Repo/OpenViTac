from ._base_task import *
import numpy as np

# Frame 尺寸：外框 100mm x 100mm，内框 90mm x 90mm，厚度 2mm

FRAME_OUTER_SIZE = 0.1000
FRAME_INNER_SIZE = 0.0900
FRAME_THICKNESS = 0.0020
CUBE_SIZE = 0.0300

LEFT_FRAME_BASE_POSE = Pose([0.5, -0.08, 0.002], [1, 0, 0, 0])
RIGHT_FRAME_BASE_POSE = Pose([0.5, 0.08, 0.002], [1, 0, 0, 0])
CUBE_BASE_POSE = Pose([0.38, 0.0, 0.002], [1, 0, 0, 0])

# reset 时只在 xy 平面加小扰动，z 维保持 0，避免物体初始高度被随机噪声带离桌面。
FRAME_XY_NOISE = (0.01, 0.01, 0.0)
CUBE_XY_NOISE = (0.02, 0.04, 0.0)
PLACE_XY_NOISE = 0.02
GRASP_ROTATE_NOISE = np.deg2rad(10.0)
GRASP_HEIGHT = CUBE_SIZE * 0.5
GRASP_HEIGHT_NOISE = 0.003
LIFT_HEIGHT = 0.0300
SUCCESS_CUBE_MAX_Z = 0.004
SUCCESS_GRIPPER_OPEN_RATIO = 0.4
TARGET_AREA_COLORS = ("yellow", "blue")
FRAME_ORDERS = ("yellow_left", "blue_left")
TASK_INSTRUCTION = "Place the red cube on the yellow area."
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
    target_area: Literal["random", "yellow", "blue"] = "yellow"
    frame_order: Literal["yellow_left", "blue_left"] = "blue_left"
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(pos=(0.8, 0.0, 0.15), rot=(0.54167522, 0.454519478, 0.454519478, 0.54167522), convention="opengl"),
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
            update_period=1/120,
        ),
    ]
    step_lim = 300


class Task(BaseTask):
    def __init__(self, cfg: TaskCfg, mode: Literal["collect", "eval"] = "collect", render_mode: str | None = None, **kwargs):
        if cfg.target_area not in ("random", *TARGET_AREA_COLORS):
            raise ValueError(f"target_area must be 'random', 'yellow', or 'blue'")
        if cfg.frame_order not in FRAME_ORDERS:
            raise ValueError(f"frame_order must be one of {FRAME_ORDERS}")
        self.frame_order = cfg.frame_order
        if self.frame_order == "yellow_left":
            self.yellow_frame_base_pose = LEFT_FRAME_BASE_POSE
            self.blue_frame_base_pose = RIGHT_FRAME_BASE_POSE
        else:
            self.yellow_frame_base_pose = RIGHT_FRAME_BASE_POSE
            self.blue_frame_base_pose = LEFT_FRAME_BASE_POSE
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
        self.yellow_area = self._actor_manager.add_from_usd_file(
            name="yellow_area",
            asset_path="task_assets/place_cube_on_colored_area/square_frame_yellow.usd",
            pose=self.yellow_frame_base_pose,
            density=1e6,
            keep_constrained=True,
        )
        self.blue_area = self._actor_manager.add_from_usd_file(
            name="blue_area",
            asset_path="task_assets/place_cube_on_colored_area/square_frame_blue.usd",
            pose=self.blue_frame_base_pose,
            density=1e6,
            keep_constrained=True,
        )
        self.wooden_cube = self._actor_manager.add_from_usd_file(
            name="wooden_cube",
            asset_path="task_assets/place_cube_on_colored_area/Cube_red.usd",
            pose=CUBE_BASE_POSE,
            density=1e3
        )

    def _reset_actors(self):
        # 每个 episode 都重新采样黄色区域和方块的初始 xy 位置，让策略看到轻微的位姿变化。
        area_offset = self.create_noise(list(FRAME_XY_NOISE))
        cube_offset = self.create_noise(list(CUBE_XY_NOISE))
        yellow_pose = self.yellow_frame_base_pose.add_offset(area_offset)
        blue_pose = self.blue_frame_base_pose.add_offset(area_offset)
        cube_pose = CUBE_BASE_POSE.add_offset(cube_offset)

        self.yellow_area.set_pose(yellow_pose)
        self.blue_area.set_pose(blue_pose)
        self.wooden_cube.set_pose(cube_pose)

        self.target_area_color = self.cfg.target_area
        if self.target_area_color == "random":
            self.target_area_color = str(self.rng.choice(TARGET_AREA_COLORS))
        self.target_area = {
            "yellow": self.yellow_area,
            "blue": self.blue_area,
        }[self.target_area_color]

        self.metadata["area_xy_noise"] = area_offset.p.tolist()
        self.metadata["cube_xy_noise"] = cube_offset.p.tolist()
        self.metadata["yellow_area_pose"] = yellow_pose.tolist()
        self.metadata["blue_area_pose"] = blue_pose.tolist()
        self.metadata["wooden_cube_pose"] = cube_pose.tolist()
        self.metadata["target_area_color"] = self.target_area_color
        self.metadata["frame_order"] = self.frame_order

    def _release_reset_constraints(self):
        self._actor_manager.remove_animate(force=True)

    def build_instruction(self) -> str:
        return f"Place the red cube on the {self.target_area_color} area."

    def _get_local_z_bounds(self, actor, fallback_half_height):
        # 优先使用 actor 预先缓存的表面点估计局部 z 上下界；这样比硬编码高度更适配非规则几何。
        if hasattr(actor, "origin_surf_pts"):
            local_points = np.asarray(actor.origin_surf_pts)
            return float(np.min(local_points[:, 2])), float(np.max(local_points[:, 2]))
        # 如果没有表面点，就退回到传入的半高估计，保持采样逻辑可用。
        return -fallback_half_height, fallback_half_height

    def _sample_place_pose(self):
        area_pose = self.target_area.get_pose()
        cube_min_z, cube_max_z = self._get_local_z_bounds(self.wooden_cube, CUBE_SIZE * 0.5)
        area_min_z, area_max_z = self._get_local_z_bounds(self.target_area, FRAME_THICKNESS * 0.5)
        # 放置点在黄色区域中心附近随机 +/-5mm，避免每条轨迹完全重合。
        place_xy_noise = self.rng.uniform(-PLACE_XY_NOISE, PLACE_XY_NOISE, size=2)
        # 目标 z 让方块底面刚好落在黄色区域表面：area_z + area_top - cube_bottom。
        place_z = area_pose.p[2] + area_max_z - cube_min_z
        place_pose = Pose(
            [
                area_pose.p[0] + place_xy_noise[0],
                area_pose.p[1] + place_xy_noise[1],
                place_z,
            ],
            [1, 0, 0, 0],
        )
        self.metadata["place_xy_noise"] = place_xy_noise.tolist()
        self.metadata["area_local_z_bounds"] = [area_min_z, area_max_z]
        self.metadata["cube_local_z_bounds"] = [cube_min_z, cube_max_z]
        self.metadata["target_place_pose"] = place_pose.tolist()
        return place_pose

    def pre_move(self):
        # 初始先等待若干仿真步，让物体在物理引擎中稳定，再打开夹爪准备抓取。
        # Xense/Robotiq 的前几次空仿真步非常贵；保留 settle，但避免十个纯等待步把正式视频验证拖到超时。
        is_xense = getattr(self.cfg, "tactile_sensor_type", "") in ("xensews", "xensews_robotiq")
        initial_settle_steps = 10
        if is_xense:
            initial_settle_steps = int(getattr(self.cfg, "xense_cube_initial_settle_steps", 1))
        if initial_settle_steps > 0:
            self.delay(initial_settle_steps)
        if not is_xense:
            self.move(self.atom.open_gripper(0.5), tag="open_gripper_for_policy")

    def prepare_initial_state(self):
        if getattr(self.cfg, "tactile_sensor_type", "") not in ("xensews", "xensews_robotiq"):
            return
        self.move(
            self.atom.open_gripper(0.6),
            tag="setup_open_gripper_for_policy",
            is_save=False,
        )
        self.delay(20, is_save=False)

    def _grasp_cube(self):
        cube_pose = self.wooden_cube.get_pose()
        is_xense = getattr(self.cfg, "tactile_sensor_type", "") in (
            "xensews",
            "xensews_robotiq",
        )
        # 抓取姿态以方块中心为基准，上移到半高附近；绕局部 y 轴加少量随机旋转，提高数据多样性。
        grasp_rotate = (
            0.0
            if is_xense
            else self.rng.uniform(-GRASP_ROTATE_NOISE, GRASP_ROTATE_NOISE)
        )
        grasp_height_bias = self.get_xense_grasp_height_bias("xense_cube_grasp_height_bias")
        grasp_world_y_bias = self.get_xense_grasp_height_bias("xense_cube_grasp_world_y_bias")
        grasp_height_noise = (
            0.0
            if is_xense
            else self.rng.uniform(-GRASP_HEIGHT_NOISE, GRASP_HEIGHT_NOISE)
        )
        grasp_height = (
            GRASP_HEIGHT
            + grasp_height_noise
            + grasp_height_bias
        )
        target_pose = (
            cube_pose
            .add_bias([0.0, 0.0, grasp_height])
            .add_bias([0.0, grasp_world_y_bias, 0.0], coord="world")
            .add_rotation([0.0, grasp_rotate, 0.0])
        )
        target_mat = target_pose.to_transformation_matrix()
        # construct_grasp_pose 需要抓取点、接近方向和夹爪横向方向；这里沿目标局部 z 方向接近。
        grasp_pose = construct_grasp_pose(
            target_pose.p,
            target_mat[:3, 2],
            target_mat[:3, 0],
        )
        # register_point 会把抓取点注册到物体局部坐标，用于后续 atom.grasp_actor 生成接近轨迹。
        contact_point_id = self.wooden_cube.register_point(grasp_pose, type="contact")

        approach_actions = self.atom.grasp_actor(
            self.wooden_cube,
            contact_point_id=contact_point_id,
            is_close=False,
            pre_dis=0.05,
        )
        self.move(approach_actions, tag="approach_wooden_cube")
        self.record_xense_grasp_debug("xense_after_approach_wooden_cube", self.wooden_cube)

        close_percent = self.get_xense_close_percent("xense_cube_close_percent")
        self.move(
            self.atom.close_gripper(pos=close_percent),
            tag="close_wooden_cube",
            gripper_depth_threshold=self.get_xense_adaptive_grasp_depth_threshold(
                "xense_cube_adaptive_grasp_depth_threshold"
            ),
            gripper_require_both_contacts=self.get_xense_adaptive_grasp_require_both_contacts(
                "xense_cube_adaptive_grasp_require_both_contacts"
            ),
        )
        self.settle_xense_after_close(is_save=False)
        self.record_xense_grasp_debug("xense_after_close_wooden_cube", self.wooden_cube)

        self.metadata["grasp_rotate_rad"] = float(grasp_rotate)
        self.metadata["grasp_rotate_deg"] = float(np.rad2deg(grasp_rotate))
        self.metadata["grasp_height"] = float(grasp_height)
        self.metadata["grasp_height_noise"] = float(grasp_height_noise)
        self.metadata["grasp_height_bias"] = float(grasp_height_bias)
        self.metadata["grasp_world_y_bias"] = float(grasp_world_y_bias)
        self.metadata["gripper_close_percent"] = float(close_percent)
        self.metadata["grasp_pose"] = grasp_pose.tolist()

    def _play_once(self):
        self._grasp_cube()
        place_pose = self._sample_place_pose()

        # 抓牢后先竖直上提 3cm，给方块离开桌面和越过黄色区域边框留出余量。
        self.move(self.atom.move_by_displacement(z=LIFT_HEIGHT), tag="lift_wooden_cube")
        self.move(self.atom.place_actor(
            self.wooden_cube,
            target_pose=place_pose,
            pre_dis=0.02,
            dis=0.005,
            is_open=False,
        ), tag=f"place_wooden_cube_on_{self.target_area_color}_area", time_dilation_factor=0.5)
        self.move(self.atom.open_gripper(0.5), tag="release_wooden_cube")
        # 松爪后不保存等待帧，避免把纯稳定过程混入动作数据，同时让 success 检查读到更稳定的物理状态。
        self.delay(20, is_save=False)

    def _get_success_diagnostics(self):
        area_pose = self.target_area.get_pose()
        cube_pose = self.wooden_cube.get_pose()
        gripper_qpos = float(self._robot_manager.get_gripper_qpos())
        gripper_open_percentage = float(self._robot_manager.get_gripper_percentage())
        # 将方块位姿转到黄色区域坐标系下，后续 xy 判断就可以直接和方框内半径比较。
        cube_in_area = cube_pose.rebase(area_pose)
        inner_half_size = FRAME_INNER_SIZE * 0.5
        cube_half_size = CUBE_SIZE * 0.5
        # center_* 只检查方块中心是否在框内；footprint_* 进一步要求方块外轮廓也完全落在内框里。
        center_x_ok = bool(abs(cube_in_area.p[0]) <= inner_half_size)
        center_y_ok = bool(abs(cube_in_area.p[1]) <= inner_half_size)
        footprint_x_ok = bool(abs(cube_in_area.p[0]) + cube_half_size <= inner_half_size)
        footprint_y_ok = bool(abs(cube_in_area.p[1]) + cube_half_size <= inner_half_size)
        cube_z_ok = bool(cube_in_area.p[2] <= SUCCESS_CUBE_MAX_Z)
        gripper_open_ok = bool(gripper_open_percentage > SUCCESS_GRIPPER_OPEN_RATIO)
        return {
            "target_area_color": self.target_area_color,
            "target_area_pose": area_pose.tolist(),
            "yellow_area_pose": self.yellow_area.get_pose().tolist(),
            "blue_area_pose": self.blue_area.get_pose().tolist(),
            "wooden_cube_pose": cube_pose.tolist(),
            "cube_pose_in_area": cube_in_area.tolist(),
            "cube_xy_in_area": cube_in_area.p[:2].tolist(),
            "cube_z_in_area": float(cube_in_area.p[2]),
            "frame_outer_half_size": float(FRAME_OUTER_SIZE * 0.5),
            "frame_inner_half_size": float(inner_half_size),
            "cube_half_size": float(cube_half_size),
            "center_x_ok": center_x_ok,
            "center_y_ok": center_y_ok,
            "footprint_x_ok": footprint_x_ok,
            "footprint_y_ok": footprint_y_ok,
            "cube_z_ok": cube_z_ok,
            "cube_max_z_threshold": float(SUCCESS_CUBE_MAX_Z),
            "gripper_open_ok": gripper_open_ok,
            "gripper_open_percentage": gripper_open_percentage,
            "gripper_open_ratio_threshold": float(SUCCESS_GRIPPER_OPEN_RATIO),
            "gripper_qpos": gripper_qpos,
            "xy_ok": bool(footprint_x_ok and footprint_y_ok),
        }

    def check_success(self):
        # 成功判定要求方块 footprint 位于目标框内、贴近区域平面，且夹爪最终处于张开状态。
        diagnostics = self._get_success_diagnostics()
        self.metadata["success_diagnostics"] = diagnostics
        return bool(
            diagnostics["xy_ok"]
            and diagnostics["cube_z_ok"]
            and diagnostics["gripper_open_ok"]
        )
