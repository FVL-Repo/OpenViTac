from ._base_task import *
import numpy as np


# 可选模式：
# - "vertical_single": 使用原来的竖直抓取姿态，并一次性沿世界 -X 拉动。
# - "tilted_segmented": 使用 30 度倾斜抓取姿态，并分段沿世界 -X 拉动。
PULL_DRAWER_MODE = "tilted_segmented"
TASK_INSTRUCTION = "Grasp the upper drawer handle and pull the upper drawer open."


# 本任务实现“抓住上层抽屉把手并向外拉开”的脚本式采集/评测逻辑。
# 下面这些几何常量来自 assets/objects/*.obj 的网格顶点尺寸，用于把
# cabinet、drawer 和 handle 的相对位置对齐到同一个局部坐标系中。

# 柜体 USD 的默认朝向需要绕 z 轴旋转 90 度后，正面才会朝向机器人。
ROT90_Z_Q = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]

# 柜体在世界坐标系中的基准位姿。z=0.002 用于让模型略高于桌面，减少穿模。
CABINET_BASE_POSE = Pose([0.75, 0.0, 0.002], ROT90_Z_Q)

# 每次 reset 时对柜体 xy 位置加入的随机扰动幅度，z 不随机。
CABINET_XY_NOISE = (0.02, 0.02, 0.0)

# cabinet_body 和 drawer 网格的前表面 y 坐标不同，这里计算抽屉相对柜体
# 需要补偿的局部 y 偏移，使抽屉面板和柜体正面在初始状态下对齐。
CABINET_FRONT_Y = 0.115
DRAWER_FRONT_Y = 0.105
DRAWER_Y_OFFSET = CABINET_FRONT_Y - DRAWER_FRONT_Y

# 抽屉的局部 z 偏移。额外的 clearance 用于给抽屉和柜体之间留出微小间隙，
# 避免 reset 后 UIPC/物理仿真初始状态过度接触。
DRAWER_Z_CLEARANCE = 0.0010
XENSE_DRAWER_Z_CLEARANCE = DRAWER_Z_CLEARANCE
LOWER_DRAWER_Z_OFFSET = 0.0085 + DRAWER_Z_CLEARANCE
UPPER_DRAWER_Z_OFFSET = 0.0700 + DRAWER_Z_CLEARANCE

# 上层抽屉把手在 upper_drawer 局部坐标系中的抓取点位置。
HANDLE_X = 0.0
HANDLE_Y = 0.124
HANDLE_Z = 0.025

# 构造抓取姿态时使用的两个方向向量：
# HANDLE_GRASP_FROM 表示夹爪从哪个方向靠近把手；
# HANDLE_GRIPPER_UP 表示夹爪自身的“上”方向，用于确定末端姿态的滚转角。
GRASP_TILT_DEG = 30.0
GRASP_TILT_RAD = np.deg2rad(GRASP_TILT_DEG)

HANDLE_GRASP_FROM = np.array([
    -np.sin(GRASP_TILT_RAD),
    0.0,
    np.cos(GRASP_TILT_RAD),
]) if PULL_DRAWER_MODE == "tilted_segmented" else np.array([0.0, 0.0, 1.0])
HANDLE_GRIPPER_UP = np.array([1.0, 0.0, 0.0])

# 任务动作和成功判定阈值。脚本会沿世界坐标 x 负方向拉 10 cm；
# 成功条件要求实际拉出距离超过 8 cm，且抽屉高度变化不能超过 1 cm。
PULL_DISTANCE = 0.10
PULL_STEPS = 5
SUCCESS_PULL_DISTANCE = 0.08
SUCCESS_Z_THRESHOLD = 0.01
TASK_INITIAL_JOINT_POS = {
    "panda_joint1": -0.010809095,
    "panda_joint2": 0.096037410,
    "panda_joint3": 0.000734462,
    "panda_joint4": -2.433035851,
    "panda_joint5": 0.035354517,
    "panda_joint6": 2.500859022,
    "panda_joint7": 0.741,
}

if PULL_DRAWER_MODE not in ["vertical_single", "tilted_segmented"]:
    raise ValueError(f"Unsupported PULL_DRAWER_MODE: {PULL_DRAWER_MODE}")


@configclass
class TaskCfg(BaseTaskCfg):
    # 覆盖基类默认相机参数：head 相机稍微偏向柜体正面，便于同时看到抽屉和把手。
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(pos=(0.5, 0.5, 0.16), rot=(-0.061628, -0.061628, 0.704416, 0.704416), convention="opengl"),
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

    # 策略评测时最多执行 400 个高层动作。
    step_lim = 400


class Task(BaseTask):
    def __init__(self, cfg: TaskCfg, mode:Literal['collect', 'eval'] = 'collect', render_mode: str|None = None, **kwargs):
        # 抽屉任务依赖稳定的接触和摩擦。这里同时提高 Isaac 侧刚体材质摩擦
        # 和 UIPC 接触摩擦，减少抓住把手后打滑或抽屉异常弹出的情况。
        cfg.sim.physics_material.dynamic_friction = 2.5
        cfg.sim.physics_material.static_friction = 2.5
        cfg.uipc_sim.contact.default_friction_ratio = 2.5
        super().__init__(cfg, mode, render_mode, **kwargs)
        # BaseTask takes its first UIPC reset steps before calling
        # _reset_actors(). Stage the assembly constraints now so those first
        # steps do not solve an unconstrained cabinet/drawer contact stack.
        self._stage_drawer_assembly(
            self.cabinet_init_pose,
            self.lower_drawer_init_pose,
            self.upper_drawer_init_pose,
        )

    def load_robot_and_sensors(self, cfg: BaseTaskCfg):
        cfg = super().load_robot_and_sensors(cfg)
        joint_pos = TASK_INITIAL_JOINT_POS
        if getattr(cfg, "tactile_sensor_type", "") in ("xensews", "xensews_robotiq"):
            joint_pos = apply_xense_wrist_y_alignment(joint_pos)
        cfg.robot.robot.init_state.joint_pos.update(joint_pos)
        return cfg

    def _is_xense(self):
        return self.cfg.tactile_sensor_type in ("xensews", "xensews_robotiq")

    def _drawer_pose(self, cabinet_pose: Pose, z_offset: float) -> Pose:
        """根据柜体位姿和抽屉层高，计算抽屉在世界坐标系中的初始位姿。"""
        if self._is_xense():
            z_offset += XENSE_DRAWER_Z_CLEARANCE - DRAWER_Z_CLEARANCE
        return cabinet_pose.add_bias([0.0, DRAWER_Y_OFFSET, z_offset], coord='local')

    def _set_reference_poses(self, cabinet_pose: Pose, lower_drawer_pose: Pose, upper_drawer_pose: Pose):
        """记录本轮 reset 后的参考位姿，后续成功判定会和这些初始位姿比较。"""
        self.cabinet_init_pose = cabinet_pose.clone()
        self.lower_drawer_init_pose = lower_drawer_pose.clone()
        self.upper_drawer_init_pose = upper_drawer_pose.clone()

    def _stage_drawer_assembly(self, cabinet_pose: Pose, lower_drawer_pose: Pose, upper_drawer_pose: Pose):
        """Stage assembly poses before UIPC advances the contact solver."""
        self.cabinet.set_pose(cabinet_pose)
        self.lower_drawer.set_pose(lower_drawer_pose)
        self.upper_drawer.set_pose(upper_drawer_pose)
        self._actor_manager.update(dt=0.0)

    def create_actors(self):
        # create_actors 只在环境初始化时调用一次，用于把 USD 资产注册成 UIPC actor。
        # 后续每个 episode 的随机化不重新创建资产，而是在 _reset_actors 中移动已有 actor。
        cabinet_pose = CABINET_BASE_POSE.clone()
        lower_drawer_pose = self._drawer_pose(cabinet_pose, LOWER_DRAWER_Z_OFFSET)
        upper_drawer_pose = self._drawer_pose(cabinet_pose, UPPER_DRAWER_Z_OFFSET)

        # 柜体密度设得很大，近似作为固定基座；上下抽屉保留较低密度，允许被拉动。
        self.cabinet = self._actor_manager.add_from_usd_file(
            name='cabinet',
            asset_path="task_assets/pull_drawer/cabinet_body.usd",
            # visual_asset_path="cabinet_body_picture.usd",
            pose=cabinet_pose,
            density=1e5,
            keep_constrained=True,
            # show_physics_mesh=False,
        )
        self.lower_drawer = self._actor_manager.add_from_usd_file(
            name='lower_drawer',
            asset_path="task_assets/pull_drawer/lower_drawer.usd",
            pose=lower_drawer_pose,
            density=1e3,
            keep_constrained=True,
        )
        self.upper_drawer = self._actor_manager.add_from_usd_file(
            name='upper_drawer',
            asset_path="task_assets/pull_drawer/upper_drawer.usd",
            pose=upper_drawer_pose,
            density=1e3
        )
        self._set_reference_poses(cabinet_pose, lower_drawer_pose, upper_drawer_pose)

    def _reset_actors(self):
        # 每轮 reset 时给整个柜体加入小范围位置随机化，再用相同局部偏移重建抽屉位姿，
        # 这样抽屉与柜体的相对结构保持不变，但任务场景会有轻微分布变化。
        cabinet_pose = CABINET_BASE_POSE.add_offset(
            self.create_noise(list(CABINET_XY_NOISE)),
            coord='world'
        )
        lower_drawer_pose = self._drawer_pose(cabinet_pose, LOWER_DRAWER_Z_OFFSET)
        upper_drawer_pose = self._drawer_pose(cabinet_pose, UPPER_DRAWER_Z_OFFSET)

        self._stage_drawer_assembly(cabinet_pose, lower_drawer_pose, upper_drawer_pose)
        self._set_reference_poses(cabinet_pose, lower_drawer_pose, upper_drawer_pose)

        # 保存初始位姿到 metadata，便于离线检查每条数据对应的随机化状态。
        self.metadata['cabinet_init_pose'] = cabinet_pose.tolist()
        self.metadata['lower_drawer_init_pose'] = lower_drawer_pose.tolist()
        self.metadata['upper_drawer_init_pose'] = upper_drawer_pose.tolist()
        self.metadata['drawer_z_clearance'] = float(
            XENSE_DRAWER_Z_CLEARANCE if self._is_xense() else DRAWER_Z_CLEARANCE
        )
        self.metadata['static_fixture_constraints'] = False
        self.metadata['reset_constraint_actor_names'] = [
            'cabinet',
            'lower_drawer',
            'upper_drawer',
        ]

    def _release_reset_constraints(self):
        self._actor_manager.remove_animate(force=True)

    def pre_move(self):
        # pre_move 在正式记录动作前执行：先稳定仿真，再张开夹爪准备靠近把手。
        is_xense = getattr(self.cfg, "tactile_sensor_type", "") in ("xensews", "xensews_robotiq")
        initial_settle_steps = 10
        if is_xense:
            initial_settle_steps = int(
                getattr(
                    self.cfg,
                    "xense_drawer_initial_settle_steps",
                    getattr(self.cfg, "xense_initial_settle_steps", 1),
                )
            )
        if initial_settle_steps > 0:
            self.delay(initial_settle_steps)
        if not is_xense:
            self.move(self.atom.open_gripper(0.5), tag="open_gripper_for_policy")

    def prepare_initial_state(self):
        if not self._is_xense():
            return
        self.move(
            self.atom.open_gripper(0.5),
            tag="setup_open_gripper_for_policy",
            is_save=False,
        )
        self.delay(20, is_save=False)

    def _grasp_drawer_handle(self):
        # 把局部把手坐标转换到当前 upper_drawer 的世界坐标中。
        target_pose = self.upper_drawer.get_pose().add_bias(
            [HANDLE_X, HANDLE_Y, HANDLE_Z],
            coord='local'
        )
        grasp_z_bias = self.get_xense_grasp_height_bias("xense_drawer_grasp_z_bias")
        target_pose = target_pose.add_bias([0.0, 0.0, grasp_z_bias], coord='world')

        # 根据抓取点和方向向量构造夹爪末端姿态，并注册为 upper_drawer 的 contact point。
        # 后续 atom.grasp_actor 会根据这个 contact point 生成靠近把手的动作序列。
        cpose = construct_grasp_pose(
            target_pose.p,
            HANDLE_GRASP_FROM,
            HANDLE_GRIPPER_UP
        )
        cid = self.upper_drawer.register_point(cpose, type='contact')

        self.move(self.atom.grasp_actor(
            self.upper_drawer,
            contact_point_id=cid,
            pre_dis=0.08,
            dis=0.0,
            is_close=False
        ), tag="approach_upper_drawer_handle")
        self.record_xense_grasp_debug(
            "xense_after_approach_upper_drawer_handle",
            self.upper_drawer,
        )

        # 靠近把手后再闭合夹爪。这里把 close 单独放一步，方便采集到“接近”和“夹紧”阶段。
        close_percent = self.get_xense_close_percent("xense_drawer_close_percent")
        self.move(
            self.atom.close_gripper(pos=close_percent),
            tag="close_upper_drawer_handle",
            gripper_depth_threshold=self.get_xense_adaptive_grasp_depth_threshold(
                "xense_drawer_adaptive_grasp_depth_threshold"
            ),
            gripper_require_both_contacts=self.get_xense_adaptive_grasp_require_both_contacts(
                "xense_drawer_adaptive_grasp_require_both_contacts"
            ),
        )
        self.settle_xense_after_close(is_save=False)
        self.record_xense_grasp_debug(
            "xense_after_close_upper_drawer_handle",
            self.upper_drawer,
        )

        # 记录把手 y 坐标和抓取姿态，便于复现实验或诊断失败数据。
        self.metadata['pull_drawer_mode'] = PULL_DRAWER_MODE
        self.metadata['grasp_tilt_deg'] = float(GRASP_TILT_DEG if PULL_DRAWER_MODE == "tilted_segmented" else 0.0)
        self.metadata['handle_grasp_from'] = HANDLE_GRASP_FROM.tolist()
        self.metadata['handle_y'] = float(HANDLE_Y)
        self.metadata['handle_contact_pose'] = cpose.tolist()
        self.metadata['grasp_z_bias'] = float(grasp_z_bias)
        self.metadata['gripper_close_percent'] = float(close_percent)

    def _play_once(self):
        self._grasp_drawer_handle()
        is_xense = getattr(self.cfg, "tactile_sensor_type", "") in (
            "xensews",
            "xensews_robotiq",
        )
        pull_distance = (
            float(getattr(self.cfg, "xense_drawer_pull_distance", 0.11))
            if is_xense
            else PULL_DISTANCE
        )
        self.metadata['requested_pull_distance'] = float(pull_distance)
        if PULL_DRAWER_MODE == "vertical_single":
            # 原始演示动作：保持竖直抓取姿态，一次性沿世界坐标 x 负方向拉出上层抽屉。
            self.metadata['pull_steps'] = 1
            self.metadata['pull_step_distance'] = float(pull_distance)
            self.move(self.atom.move_by_displacement(
                x=-pull_distance,
                xyz_coord='world'
            ), tag="pull_drawer_x_minus", time_dilation_factor=0.5)
        else:
            # 倾斜抓取动作：保持夹爪抓住把手，沿世界坐标 x 负方向分段拉出上层抽屉。
            # 单次 cuRobo 规划只约束终点，不保证中间轨迹严格水平；分段后轨迹更接近直线 -X。
            pull_step_distance = pull_distance / PULL_STEPS
            self.metadata['pull_step_distance'] = float(pull_step_distance)
            self.metadata['pull_steps'] = int(PULL_STEPS)
            for step_idx in range(PULL_STEPS):
                self.move(
                    self.atom.move_by_displacement(
                        x=-pull_step_distance,
                        xyz_coord='world'
                    ),
                    tag=f"pull_drawer_x_minus_step_{step_idx}",
                    time_dilation_factor=0.5,
                    delay=False
                )

        # 拉动后等待一段时间，让抽屉和夹爪接触状态稳定下来，同时保存尾段观测。
        self.delay(20, is_save=True)
        self.record_xense_grasp_debug(
            "xense_after_pull_upper_drawer_handle",
            self.upper_drawer,
        )

    def _get_success_diagnostics(self):
        # 用当前上层抽屉位姿和 reset 时记录的初始位姿比较，得到成功判定的中间量。
        upper_pose = self.upper_drawer.get_pose()
        init_pose = self.upper_drawer_init_pose

        # 因脚本沿世界 x 负方向拉动，所以 init_x - current_x 为正表示抽屉被拉出。
        pull_distance_x = float(init_pose.p[0] - upper_pose.p[0])
        z_delta = float(upper_pose.p[2] - init_pose.p[2])

        return {
            'upper_drawer_pose': upper_pose.tolist(),
            'upper_drawer_init_pose': init_pose.tolist(),
            'pull_distance_x': pull_distance_x,
            'z_delta': z_delta,
            'pull_ok': bool(pull_distance_x > SUCCESS_PULL_DISTANCE),
            'z_ok': bool(abs(z_delta) < SUCCESS_Z_THRESHOLD),
        }

    def check_success(self):
        # 成功条件包含两部分：
        # 1. 抽屉沿目标方向被拉出的距离足够大；
        # 2. 抽屉没有明显上抬或下沉，避免把“被夹爪抬飞”误判为成功拉开。
        diagnostics = self._get_success_diagnostics()
        self.metadata['success_diagnostics'] = diagnostics
        return diagnostics['pull_ok'] and diagnostics['z_ok']
