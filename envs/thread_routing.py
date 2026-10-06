from ._base_task import *
import numpy as np
import torch


TASK_INSTRUCTION = "Route the black cable through the three white cable clips from left to right."

TABLE_TOP_Z = 0.0025
CABLE_ASSET_POSE = Pose([-0.39, 0.055, 0.0], [1, 0, 0, 0])
CABLE_THREAD_Z = 0.0245
CABLE_DRAG_Z = 0.034
CABLE_PRESS_Z = 0.016
CABLE_GRASP_Z_BIAS = 0.010
CABLE_PRE_DIS = 0.055
CABLE_TIP_HOLD_COUNT = 2
CABLE_ANCHOR_COUNT = 0
CABLE_DRAG_MAX_STEP = 0.025
CABLE_PRESS_MAX_STEP = 0.006
CABLE_CLIP_APPROACH_OFFSET = 0.040
CABLE_CLIP_EXIT_OFFSET = 0.045
CABLE_PRESS_FORWARD_OFFSET = 0.024

CLIP_XS = [0.42, 0.53, 0.64]
CLIP_YS = [0.055, -0.020, 0.055]
CLIP_GROUND_CLEARANCE = 0.002
CLIP_POSE_Z = TABLE_TOP_Z + CLIP_GROUND_CLEARANCE
CLIP_ASSET_PATH = "task_assets/cable_routing/cable_clip_white.usd"

FINAL_TIP_TARGET = np.array([0.70, 0.055, CABLE_DRAG_Z])


@configclass
class TaskCfg(BaseTaskCfg):
    step_lim = 360
    reset_time_limit = 500.0
    use_adaptive_grasp = False
    planner_ignore_actors: tuple[str, ...] = (
        "cable",
        "clip_0",
        "clip_1",
        "clip_2",
    )
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(
                pos=(1.12, 0.04, 0.50),
                rot=(0.579228, 0.405580, 0.405580, 0.579228),
                convention="opengl",
            ),
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=1.7,
                focus_distance=1.0,
                horizontal_aperture=4.2,
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
        cfg.sim.physics_material.dynamic_friction = 2.0
        cfg.sim.physics_material.static_friction = 2.0
        cfg.uipc_sim.contact.default_friction_ratio = 2.0
        cfg.uipc_sim.contact.d_hat = 0.0008
        cfg.uipc_sim.ground_height = TABLE_TOP_Z
        cfg.uipc_sim.newton.max_iter = 1024
        super().__init__(cfg, mode, render_mode, **kwargs)

    def create_actors(self):
        self.clips = []
        fixed_cfg = UipcObjectCfg.AffineBodyConstitutionCfg(m_kappa=200.0, kinematic=True)
        for clip_id, (x, y) in enumerate(zip(CLIP_XS, CLIP_YS)):
            clip = self._actor_manager.add_from_usd_file(
                name=f"clip_{clip_id}",
                asset_path=CLIP_ASSET_PATH,
                visual_asset_path=CLIP_ASSET_PATH,
                pose=Pose([x, y, CLIP_POSE_Z], [1, 0, 0, 0]),
                constitution_cfg=fixed_cfg,
                density=5.0e3,
                show_physics_mesh=False,
            )
            self.clips.append(clip)

        self.cable = self._actor_manager.add_from_usd_file(
            name="cable",
            asset_path="thread_task/cable_centerline.usda",
            pose=CABLE_ASSET_POSE,
            constitution_cfg=UipcObjectCfg.HookeanSpringCfg(
                kappa=4.0e4,
                thickness=0.006,
                enable_bending=True,
                bending_stiffness=1.0e5,
                render_radius=0.004,
                render_sides=8,
            ),
            density=250.0,
            show_physics_mesh=True,
            keep_constrained=True,
        )

    def _reset_actors(self):
        self.cable.remove_animate(force=True)
        self.initial_cable_vertices = self.cable.vertices.copy()
        self.route_done = False
        self.current_tip_target = self.initial_cable_vertices[-1].copy()
        self._set_cable_targets(self.current_tip_target, hold_tip=False)
        self.metadata["clip_centers"] = [
            [float(x), float(y), float(CABLE_THREAD_Z)] for x, y in zip(CLIP_XS, CLIP_YS)
        ]
        self.metadata["initial_tip"] = self.initial_cable_vertices[-1].tolist()

    def _set_cable_targets(self, tip_point: np.ndarray, hold_tip: bool = True):
        vertices = self.cable.vertices.copy()
        if vertices.shape[0] == 0:
            vertices = self.cable.init_vertex_pos.detach().cpu().numpy()

        targets = vertices.copy()
        mask = np.zeros(vertices.shape[0], dtype=bool)
        mask[:CABLE_ANCHOR_COUNT] = True
        targets[:CABLE_ANCHOR_COUNT] = self.initial_cable_vertices[:CABLE_ANCHOR_COUNT]

        if hold_tip:
            tip_point = np.asarray(tip_point, dtype=np.float64).reshape(3)
            mask[-CABLE_TIP_HOLD_COUNT:] = True
            targets[-1] = tip_point
            if CABLE_TIP_HOLD_COUNT > 1:
                prev = vertices[-2]
                direction = tip_point - prev
                norm = np.linalg.norm(direction)
                if norm < 1e-6:
                    direction = np.array([1.0, 0.0, 0.0])
                else:
                    direction = direction / norm
                targets[-2] = tip_point - 0.025 * direction

        self.cable.set_vertex_targets(targets, mask=mask)
        self.current_tip_target = targets[-1].copy()

    def _tip_grasp_pose(self, point: np.ndarray) -> Pose:
        point = np.asarray(point, dtype=np.float64).copy()
        point[2] += CABLE_GRASP_Z_BIAS
        return construct_grasp_pose(point, grasp_from=[0, 0, 1], camera_up=[1, 0, 0])

    def _move_held_tip_to(
        self,
        tip_point: np.ndarray,
        tag: str,
        max_step: float = CABLE_DRAG_MAX_STEP,
        time_dilation_factor: float = 0.65,
        settle_steps: int = 1,
        is_save: bool = True,
    ):
        target = np.asarray(tip_point, dtype=np.float64).reshape(3)
        start = np.asarray(self.current_tip_target, dtype=np.float64).reshape(3)
        distance = float(np.linalg.norm(target - start))
        segments = max(1, int(np.ceil(distance / max(max_step, 1e-6))))
        self.metadata.setdefault("drag_segments", []).append(
            {
                "tag": tag,
                "start": start.tolist(),
                "target": target.tolist(),
                "segments": int(segments),
            }
        )

        for segment_id in range(segments):
            alpha = float(segment_id + 1) / float(segments)
            segment_tip = start + (target - start) * alpha
            self._set_cable_targets(segment_tip, hold_tip=True)
            pose = self._tip_grasp_pose(segment_tip)
            ok = self.move(
                [Action("move", target_pose=self.atom.robot.gripper_center_to_ee(pose))],
                tag=tag if segments == 1 else f"{tag}_{segment_id}",
                delay=False,
                is_save=is_save,
                time_dilation_factor=time_dilation_factor,
            )
            if not ok:
                return False
            if settle_steps > 0:
                self.delay(settle_steps, is_save=is_save)
        return True

    def _press_cable_into_clip(self, clip_id: int, x: float, y: float):
        approach = np.array([x - CABLE_CLIP_APPROACH_OFFSET, y, CABLE_DRAG_Z], dtype=np.float64)
        over_clip = np.array([x, y, CABLE_DRAG_Z], dtype=np.float64)
        press_ready = np.array([x + CABLE_PRESS_FORWARD_OFFSET, y, CABLE_DRAG_Z], dtype=np.float64)
        pressed = np.array([x + CABLE_PRESS_FORWARD_OFFSET, y, CABLE_PRESS_Z], dtype=np.float64)
        lifted = np.array([x + CABLE_PRESS_FORWARD_OFFSET, y, CABLE_DRAG_Z], dtype=np.float64)
        exit_point = np.array([x + CABLE_CLIP_EXIT_OFFSET, y, CABLE_DRAG_Z], dtype=np.float64)

        stages = [
            (approach, f"drag_to_clip_{clip_id}_approach", CABLE_DRAG_MAX_STEP, 0.65, 1),
            (over_clip, f"drag_over_clip_{clip_id}", CABLE_DRAG_MAX_STEP, 0.70, 1),
            (press_ready, f"move_forward_before_press_{clip_id}", CABLE_PRESS_MAX_STEP, 0.75, 2),
            (pressed, f"press_cable_into_clip_{clip_id}", CABLE_PRESS_MAX_STEP, 0.85, 8),
            (lifted, f"lift_after_clip_{clip_id}", CABLE_PRESS_MAX_STEP, 0.80, 2),
            (exit_point, f"drag_out_of_clip_{clip_id}", CABLE_DRAG_MAX_STEP, 0.70, 1),
        ]
        for target, tag, max_step, dilation, settle in stages:
            if not self._move_held_tip_to(
                target,
                tag=tag,
                max_step=max_step,
                time_dilation_factor=dilation,
                settle_steps=settle,
                is_save=True,
            ):
                return False
        return True

    def pre_move(self):
        self._set_cable_targets(self.initial_cable_vertices[-1], hold_tip=True)
        self.delay(8, is_save=False)

        grasp_pose = self._tip_grasp_pose(self.initial_cable_vertices[-1])
        pre_grasp_pose = grasp_pose.add_bias([0, 0, -CABLE_PRE_DIS])
        self.move(
            self.atom.move_to_pose(self.atom.robot.gripper_center_to_ee(pre_grasp_pose)),
            tag="approach_cable_tip",
            time_dilation_factor=0.5,
        )
        self.move(
            self.atom.move_to_pose(self.atom.robot.gripper_center_to_ee(grasp_pose)),
            tag="touch_cable_tip",
            time_dilation_factor=0.5,
        )
        self.move(self.atom.close_gripper(-0.05, depth_threshold=None), tag="pinch_cable_tip")
        self.delay(12, is_save=False)
        self.metadata["grasp_pose"] = grasp_pose.tolist()

    def _play_once(self):
        routed_waypoints = []
        for clip_id, (x, y) in enumerate(zip(CLIP_XS, CLIP_YS)):
            if not self._press_cable_into_clip(clip_id, x, y):
                return
            routed_waypoints.append([float(x), float(y), float(CABLE_PRESS_Z)])

        if not self._move_held_tip_to(
            FINAL_TIP_TARGET,
            tag="drag_to_final_exit",
            max_step=CABLE_DRAG_MAX_STEP,
            time_dilation_factor=0.70,
            settle_steps=2,
            is_save=True,
        ):
            return
        self.route_done = True
        self.metadata["routed_waypoints"] = routed_waypoints
        self.metadata["final_tip_target"] = FINAL_TIP_TARGET.tolist()
        self.delay(25, is_save=False)

    def _get_success_diagnostics(self):
        vertices = self.cable.vertices
        clip_distances = []
        for x, y in zip(CLIP_XS, CLIP_YS):
            center_xy = np.array([x, y], dtype=np.float64)
            distances = np.linalg.norm(vertices[:, :2] - center_xy.reshape(1, 2), axis=1)
            clip_distances.append(float(distances.min()))

        tip = vertices[-1]
        return {
            "route_done": bool(self.route_done),
            "clip_distances": clip_distances,
            "tip": tip.tolist(),
            "tip_goal_distance": float(np.linalg.norm(tip[:2] - FINAL_TIP_TARGET[:2])),
            "max_clip_distance": float(max(clip_distances)),
        }

    def check_success(self):
        diagnostics = self._get_success_diagnostics()
        self.metadata["success_diagnostics"] = diagnostics
        return (
            diagnostics["route_done"]
            and diagnostics["max_clip_distance"] < 0.038
            and diagnostics["tip_goal_distance"] < 0.060
        )
