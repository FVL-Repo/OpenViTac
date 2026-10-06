from ._base_task import *
import numpy as np


TASK_INSTRUCTION = "Fold the square towel by grasping the lifted corner and placing it on the opposite corner."

TOWEL_LENGTH = 0.35
TOWEL_WIDTH = 0.35
TABLE_TOP_Z = 0.0025
TOWEL_POSE = Pose([0.42, 0.0, TABLE_TOP_Z + 0.006], [1, 0, 0, 0])
GRASP_CORNER_SIGN = np.array([1.0, -1.0])
TOWEL_SHELL_THICKNESS = 0.0015
TOWEL_YOUNGS_MODULUS = 0.12
TOWEL_POISSON_RATE = 0.35
TOWEL_BENDING_STIFFNESS = 3.0
TOWEL_DENSITY = 180
TOWEL_FRICTION_RATIO = 1.4
INIT_LIFT_HEIGHT = 0.070
INIT_CORNER_LIFT_RADIUS = 0.060
INIT_CORNER_HOLD_RADIUS = 0.032
INIT_CONTACT_Z = TABLE_TOP_Z + 0.0015
GRASP_CORNER_INSET = 0.022
GRASP_Z_BIAS = -0.018
GRASP_PRE_DIS = 0.060
GRIPPER_CLOSE_PERCENT = -0.1
POST_GRASP_HOLD_DELAY = 30
LIFT_HEIGHT = 0.035
FOLD_MID_FRACTION = 0.55
FOLD_MID_Z = TABLE_TOP_Z + 0.075
FOLD_TARGET_Z = TABLE_TOP_Z + 0.050
FOLD_RELEASE_Z = TABLE_TOP_Z + 0.018
FOLD_TARGET_INSET = GRASP_CORNER_INSET
PRE_RELEASE_PRESS_DELAY = 12
POST_RELEASE_RETRACT_HEIGHT = 0.040
SETTLE_DELAY = 40
FLATNESS_HEIGHT_ABOVE_TABLE_THRESHOLD = 0.035

TASK_INITIAL_JOINT_POS = {
    "panda_joint1": -0.010809095,
    "panda_joint2": 0.096037410,
    "panda_joint3": 0.000734462,
    "panda_joint4": -2.433035851,
    "panda_joint5": 0.035354517,
    "panda_joint6": 2.500859022,
    "panda_joint7": 0.741,
    "panda_finger.*": 0.039,
}


@configclass
class TaskCfg(BaseTaskCfg):
    step_lim = 340
    use_adaptive_grasp = False
    reset_time_limit = 500.0
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(
                pos=(1.08, 0.0, 0.38),
                rot=(0.579228, 0.405580, 0.405580, 0.579228),
                convention="opengl",
            ),
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.0,
                focus_distance=1.0,
                horizontal_aperture=2.8,
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


class Task(BaseTask):
    def __init__(self, cfg: TaskCfg, mode: Literal["collect", "eval"] = "collect", render_mode: str | None = None, **kwargs):
        cfg.use_adaptive_grasp = False
        cfg.adaptive_grasp_depth_threshold = None
        cfg.sim.physics_material.dynamic_friction = TOWEL_FRICTION_RATIO
        cfg.sim.physics_material.static_friction = TOWEL_FRICTION_RATIO
        cfg.uipc_sim.contact.default_friction_ratio = TOWEL_FRICTION_RATIO
        cfg.uipc_sim.contact.d_hat = 0.0005
        cfg.uipc_sim.ground_height = TABLE_TOP_Z
        cfg.uipc_sim.newton.max_iter = 1024
        super().__init__(cfg, mode, render_mode, **kwargs)

    def load_robot_and_sensors(self, cfg: BaseTaskCfg):
        cfg = super().load_robot_and_sensors(cfg)
        cfg.robot.robot.init_state.joint_pos.update(TASK_INITIAL_JOINT_POS)
        return cfg

    def create_actors(self):
        self.towel = self._actor_manager.add_from_usd_file(
            name="patterned_towel",
            asset_path="cloth_task/patterned_towel.usda",
            pose=TOWEL_POSE,
            constitution_cfg=UipcObjectCfg.NeoHookeanShellCfg(
                youngs_modulus=TOWEL_YOUNGS_MODULUS,
                poisson_rate=TOWEL_POISSON_RATE,
                thickness=TOWEL_SHELL_THICKNESS,
                enable_bending=True,
                bending_stiffness=TOWEL_BENDING_STIFFNESS,
                render_offset=(0.0, 0.0, 0.0),
            ),
            density=TOWEL_DENSITY,
        )
        self.metadata["towel_material"] = {
            "youngs_modulus": TOWEL_YOUNGS_MODULUS,
            "poisson_rate": TOWEL_POISSON_RATE,
            "bending_stiffness": TOWEL_BENDING_STIFFNESS,
            "density": TOWEL_DENSITY,
            "friction_ratio": TOWEL_FRICTION_RATIO,
            "thickness": TOWEL_SHELL_THICKNESS,
        }

    def _reset_actors(self):
        self.towel.remove_animate()
        self.initial_vertices = None
        self.pre_release_vertices = None
        self.release_done = False
        self._flat_reset_vertices = self._get_towel_vertices().copy()
        self._raised_grasp_point = None
        self._raised_vertex_mask = None
        self._fold_target_point = None
        self._apply_raised_towel_state(hold=False)
        self.metadata["towel_initial_pose"] = TOWEL_POSE.tolist()
        self.metadata["towel_init_mode"] = "corner_lift"

    def _get_towel_vertices(self):
        vertices = self.towel.vertices
        if vertices.shape[0] == 0:
            vertices = self.towel.init_vertex_pos.detach().cpu().numpy()
        return np.asarray(vertices, dtype=np.float64)

    def _make_raised_towel_vertices(self):
        vertices = self._flat_reset_vertices.copy()
        vertices[:, 2] = np.maximum(vertices[:, 2], INIT_CONTACT_Z)

        x = vertices[:, 0]
        y = vertices[:, 1]
        x_min, x_max = float(x.min()), float(x.max())
        y_min, y_max = float(y.min()), float(y.max())
        grasp_x = x_max if GRASP_CORNER_SIGN[0] > 0 else x_min
        grasp_y = y_max if GRASP_CORNER_SIGN[1] > 0 else y_min
        target_x = x_min if GRASP_CORNER_SIGN[0] > 0 else x_max
        target_y = y_min if GRASP_CORNER_SIGN[1] > 0 else y_max
        grasp_inner_point = np.array(
            [
                grasp_x - GRASP_CORNER_INSET * GRASP_CORNER_SIGN[0],
                grasp_y - GRASP_CORNER_INSET * GRASP_CORNER_SIGN[1],
            ],
            dtype=np.float64,
        )

        dx = (x - grasp_x) / INIT_CORNER_LIFT_RADIUS
        dy = (y - grasp_y) / INIT_CORNER_LIFT_RADIUS
        corner_distance = np.sqrt(dx * dx + dy * dy)
        lift_weight = np.exp(-0.5 * corner_distance * corner_distance)
        lift_weight[corner_distance > 1.35] = 0.0
        vertices[:, 2] += INIT_LIFT_HEIGHT * lift_weight

        mask = corner_distance <= (INIT_CORNER_HOLD_RADIUS / INIT_CORNER_LIFT_RADIUS)
        if int(mask.sum()) < 4:
            top_ids = np.argsort(corner_distance)[:4]
            mask = np.zeros(vertices.shape[0], dtype=bool)
            mask[top_ids] = True

        lifted_vertices = vertices[mask]
        grasp_dist = np.linalg.norm(lifted_vertices[:, :2] - grasp_inner_point.reshape(1, 2), axis=1)
        top_count = min(8, lifted_vertices.shape[0])
        top_vertices = lifted_vertices[np.argsort(grasp_dist)[:top_count]]
        grasp_point = top_vertices.mean(axis=0)
        grasp_point[2] += GRASP_Z_BIAS
        target_point = np.array(
            [
                target_x + FOLD_TARGET_INSET * GRASP_CORNER_SIGN[0],
                target_y + FOLD_TARGET_INSET * GRASP_CORNER_SIGN[1],
                max(float(vertices[:, 2].min()), INIT_CONTACT_Z),
            ],
            dtype=np.float64,
        )
        self._fold_target_point = target_point
        return vertices, mask, grasp_point

    def _apply_raised_towel_state(self, hold: bool):
        vertices, mask, grasp_point = self._make_raised_towel_vertices()
        vertex_tensor = torch.tensor(vertices, dtype=torch.float64, device=self.towel.init_vertex_pos.device)
        self.towel.write_vertex_positions_to_sim(vertex_tensor)
        if hold:
            self.towel.set_vertex_targets(vertices, mask=mask)

        self._raised_grasp_point = grasp_point
        self._raised_vertex_mask = mask
        self.metadata["raised_towel"] = {
            "lift_height": INIT_LIFT_HEIGHT,
            "grasp_corner_inset": GRASP_CORNER_INSET,
            "held_vertex_count": int(mask.sum()),
            "grasp_point": grasp_point.tolist(),
            "target_corner_point": self._fold_target_point.tolist(),
        }

    def _make_lifted_grasp_pose(self) -> Pose:
        world_point = self._raised_grasp_point
        if world_point is None:
            _, _, world_point = self._make_raised_towel_vertices()
        return construct_grasp_pose(world_point, grasp_from=[0, 0, 1], camera_up=[1, 0, 0])

    def _make_fold_pose(self, fraction: float, z: float) -> Pose:
        if self._fold_target_point is None:
            self._make_raised_towel_vertices()
        start_point = self._raised_grasp_point
        if start_point is None:
            _, _, start_point = self._make_raised_towel_vertices()
        fraction = float(np.clip(fraction, 0.0, 1.0))
        world_point = (1.0 - fraction) * start_point + fraction * self._fold_target_point
        world_point[2] = z
        return construct_grasp_pose(world_point, grasp_from=[0, 0, 1], camera_up=[1, 0, 0])

    def pre_move(self):
        self._apply_raised_towel_state(hold=True)
        self.delay(5, is_save=False)
        self.initial_vertices = self.towel.vertices.copy()

        grasp_pose = self._make_lifted_grasp_pose()
        pre_grasp_pose = grasp_pose.add_bias([0, 0, -GRASP_PRE_DIS])
        self.move(
            self.atom.move_to_pose(self.atom.robot.gripper_center_to_ee(pre_grasp_pose)),
            tag="approach_lifted_towel_corner",
        )
        self.move(self.atom.move_to_pose(self.atom.robot.gripper_center_to_ee(grasp_pose)), tag="touch_lifted_towel_corner")
        self.move(
            self.atom.close_gripper(GRIPPER_CLOSE_PERCENT, depth_threshold=None),
            tag="pinch_towel_corner",
        )
        self.towel.remove_animate()
        self.delay(POST_GRASP_HOLD_DELAY, is_save=False)

        self.metadata["grasp_pose"] = grasp_pose.tolist()

    def _play_once(self):
        self.move(
            self.atom.move_by_displacement(z=LIFT_HEIGHT, xyz_coord="world"),
            tag="lift_towel_corner",
            time_dilation_factor=0.5,
        )
        fold_mid_pose = self._make_fold_pose(FOLD_MID_FRACTION, FOLD_MID_Z)
        self.move(
            self.atom.move_to_pose(self.atom.robot.gripper_center_to_ee(fold_mid_pose)),
            tag="sweep_towel_corner_over_diagonal",
            time_dilation_factor=0.5,
        )
        fold_target_pose = self._make_fold_pose(1.0, FOLD_TARGET_Z)
        self.move(
            self.atom.move_to_pose(self.atom.robot.gripper_center_to_ee(fold_target_pose)),
            tag="place_towel_corner_above_opposite",
            time_dilation_factor=0.5,
        )
        release_pose = self._make_fold_pose(1.0, FOLD_RELEASE_Z)
        self.move(
            self.atom.move_to_pose(self.atom.robot.gripper_center_to_ee(release_pose)),
            tag="press_corner_on_opposite_corner",
            time_dilation_factor=0.5,
        )
        self.delay(PRE_RELEASE_PRESS_DELAY, is_save=False)
        self.pre_release_vertices = self.towel.vertices.copy()
        self.move(self.atom.open_gripper(0.8), tag="release_towel")
        self.move(
            self.atom.move_by_displacement(z=POST_RELEASE_RETRACT_HEIGHT, xyz_coord="world"),
            tag="retract_after_release",
            time_dilation_factor=0.5,
        )
        self.release_done = True
        self.delay(SETTLE_DELAY, is_save=False)

    def _get_success_diagnostics(self):
        vertices = self.towel.vertices
        initial_vertices = self.initial_vertices if self.initial_vertices is not None else vertices

        initial_min_x = float(initial_vertices[:, 0].min())
        initial_max_x = float(initial_vertices[:, 0].max())
        initial_min_y = float(initial_vertices[:, 1].min())
        initial_max_y = float(initial_vertices[:, 1].max())
        current_min_x = float(vertices[:, 0].min())
        current_max_x = float(vertices[:, 0].max())
        current_min_y = float(vertices[:, 1].min())
        current_max_y = float(vertices[:, 1].max())
        initial_width = initial_max_x - initial_min_x
        initial_depth = initial_max_y - initial_min_y
        current_width = current_max_x - current_min_x
        current_depth = current_max_y - current_min_y
        width_ratio = current_width / initial_width if initial_width > 1e-6 else 1.0
        depth_ratio = current_depth / initial_depth if initial_depth > 1e-6 else 1.0
        area_ratio = width_ratio * depth_ratio
        height_above_table = np.maximum(vertices[:, 2] - TABLE_TOP_Z, 0.0)
        height_p90 = float(np.percentile(height_above_table, 90))
        height_p95 = float(np.percentile(height_above_table, 95))
        height_p99 = float(np.percentile(height_above_table, 99))
        flat_vertex_ratio = float(np.mean(height_above_table < FLATNESS_HEIGHT_ABOVE_TABLE_THRESHOLD))
        if self._raised_vertex_mask is not None and self._fold_target_point is not None:
            folded_corner = vertices[self._raised_vertex_mask].mean(axis=0)
            corner_to_target_distance = float(
                np.linalg.norm(folded_corner[:2] - self._fold_target_point[:2])
            )
        else:
            folded_corner = np.full(3, np.nan)
            corner_to_target_distance = float("inf")

        return {
            "release_done": bool(self.release_done),
            "initial_x_range": [initial_min_x, initial_max_x],
            "initial_y_range": [initial_min_y, initial_max_y],
            "current_x_range": [current_min_x, current_max_x],
            "current_y_range": [current_min_y, current_max_y],
            "initial_width": float(initial_width),
            "initial_depth": float(initial_depth),
            "current_width": float(current_width),
            "current_depth": float(current_depth),
            "width_ratio": float(width_ratio),
            "depth_ratio": float(depth_ratio),
            "area_ratio": float(area_ratio),
            "folded_corner": folded_corner.tolist(),
            "target_corner": self._fold_target_point.tolist() if self._fold_target_point is not None else None,
            "corner_to_target_distance": corner_to_target_distance,
            "max_height": float(vertices[:, 2].max()),
            "height_p90_above_table": height_p90,
            "height_p95_above_table": height_p95,
            "height_p99_above_table": height_p99,
            "flat_vertex_ratio": flat_vertex_ratio,
            "flatness_height_threshold": FLATNESS_HEIGHT_ABOVE_TABLE_THRESHOLD,
            "flatness_ok": bool(height_p95 < FLATNESS_HEIGHT_ABOVE_TABLE_THRESHOLD),
        }

    def check_success(self):
        diagnostics = self._get_success_diagnostics()
        self.metadata["success_diagnostics"] = diagnostics
        return (
            diagnostics["release_done"]
            and diagnostics["corner_to_target_distance"] < 0.090
        )
