from ._base_task import *
import numpy as np
import torch


TASK_INSTRUCTION = "Find the rough block by touch and place it on the yellow plate."

FILTER_DARK_CAMERA_PIXELS = True
DARK_CAMERA_PIXEL_THRESHOLD = 10
GRIPPER_CLOSE_QPOS_RANGE = (0.0035, 0.004)
BLOCK_RESET_XY_NOISE = 0.01
LIFT_HEIGHT = 0.05
LIFT_HEIGHT_NOISE = 0.01
TARGET_XY_NOISE = 0.01
GRASP_PRE_DISTANCE = 0.04
WRONG_BLOCK_HOLD_DELAY_STEPS = 20

LEFT_BLOCK_POSE = Pose([0.40, 0.05, 0.002], [1, 0, 0, 0])
RIGHT_BLOCK_POSE = Pose([0.40, -0.05, 0.002], [1, 0, 0, 0])
BLOCK_SIDES = ("left", "right")


def _zero_dark_camera_pixels(obs):
    if not FILTER_DARK_CAMERA_PIXELS:
        return obs

    for camera_data in obs.get("observation", {}).values():
        image = camera_data.get("rgb")
        if image is None or image.ndim != 3 or image.shape[-1] < 3:
            continue
        if isinstance(image, np.ndarray):
            filtered = image.copy()
            rgb = filtered[..., :3]
            mask = np.max(rgb, axis=-1) <= DARK_CAMERA_PIXEL_THRESHOLD
            rgb[mask] = 0
        elif torch.is_tensor(image):
            filtered = image.clone()
            rgb = filtered[..., :3]
            mask = torch.amax(rgb, dim=-1) <= DARK_CAMERA_PIXEL_THRESHOLD
            filtered[..., :3] = torch.where(mask[..., None], torch.zeros_like(rgb), rgb)
        else:
            continue
        camera_data["rgb"] = filtered
    return obs


@configclass
class TaskCfg(BaseTaskCfg):
    rough_block_side: Literal["random", "left", "right"] = "right"
    initial_grasp_side: Literal["random", "left", "right"] = "right"
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(pos=(0.9, 0.0, 0.15), rot=(0.5, 0.5, 0.5, 0.5), convention="opengl"),
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
    step_lim = 500


class Task(BaseTask):
    def __init__(self, cfg: TaskCfg, mode: Literal["collect", "eval"] = "collect", render_mode: str | None = None, **kwargs):
        if cfg.rough_block_side not in ("random", *BLOCK_SIDES):
            raise ValueError("rough_block_side must be 'random', 'left', or 'right'")
        if cfg.initial_grasp_side not in ("random", *BLOCK_SIDES):
            raise ValueError("initial_grasp_side must be 'random', 'left', or 'right'")

        cfg.sim.physics_material.dynamic_friction = 2.5
        cfg.sim.physics_material.static_friction = 2.5
        cfg.uipc_sim.contact.default_friction_ratio = 2.5
        super().__init__(cfg, mode, render_mode, **kwargs)

    def create_actors(self):
        initial_rough_side = (
            "left" if self.cfg.rough_block_side == "random" else self.cfg.rough_block_side
        )
        initial_smooth_side = self._other_side(initial_rough_side)
        initial_block_poses = {
            "left": LEFT_BLOCK_POSE,
            "right": RIGHT_BLOCK_POSE,
        }

        self.blue_plate = self._actor_manager.add_from_usd_file(
            name="blue_plate",
            asset_path="roughness_task/blue_square_plate.usd",
            pose=Pose([0.48, 0.08, 0.002], [1, 0, 0, 0]),
        )
        self.yellow_plate = self._actor_manager.add_from_usd_file(
            name="yellow_plate",
            asset_path="roughness_task/yellow_square_plate.usd",
            pose=Pose([0.48, -0.08, 0.002], [1, 0, 0, 0]),
        )

        self.smooth_block = self._actor_manager.add_from_usd_file(
            name="smooth_block",
            asset_path="roughness_task/black_smooth_cuboid.usd",
            pose=initial_block_poses[initial_smooth_side],
        )
        self.rough_block = self._actor_manager.add_from_usd_file(
            name="rough_block",
            asset_path="roughness_task/black_rough_cuboid.usd",
            pose=initial_block_poses[initial_rough_side],
        )

    def _reset_actors(self):
        self.rough_block_side = self._resolve_side(self.cfg.rough_block_side)
        self.smooth_block_side = self._other_side(self.rough_block_side)
        self.initial_grasp_side = self._resolve_side(self.cfg.initial_grasp_side)

        shared_noise = Pose([
            self.rng.uniform(-BLOCK_RESET_XY_NOISE, BLOCK_RESET_XY_NOISE),
            self.rng.uniform(-BLOCK_RESET_XY_NOISE, BLOCK_RESET_XY_NOISE),
            0.0,
        ], [1, 0, 0, 0])
        self.block_base_poses = {
            "left": LEFT_BLOCK_POSE.add_offset(shared_noise),
            "right": RIGHT_BLOCK_POSE.add_offset(shared_noise),
        }

        self.side_to_label = {
            self.rough_block_side: "rough",
            self.smooth_block_side: "smooth",
        }
        self.label_to_side = {
            label: side for side, label in self.side_to_label.items()
        }
        self.blocks = {
            "rough": self.rough_block,
            "smooth": self.smooth_block,
        }
        self.side_to_block = {
            side: self.blocks[label] for side, label in self.side_to_label.items()
        }

        self.rough_block.set_pose(self.block_base_poses[self.rough_block_side])
        self.smooth_block.set_pose(self.block_base_poses[self.smooth_block_side])
        self.target_block = self.rough_block
        self.target = self.yellow_plate
        self.other_target = self.blue_plate
        self.target_pose = self.yellow_plate.get_pose().add_bias([
            self.rng.uniform(-TARGET_XY_NOISE, TARGET_XY_NOISE),
            self.rng.uniform(-TARGET_XY_NOISE, TARGET_XY_NOISE),
            0.01,
        ])

        self.metadata["rough_block_side"] = self.rough_block_side
        self.metadata["smooth_block_side"] = self.smooth_block_side
        self.metadata["initial_grasp_side"] = self.initial_grasp_side
        self.metadata["shared_reset_noise"] = shared_noise.p.tolist()
        self.metadata["rough_block_pose"] = self.block_base_poses[self.rough_block_side].tolist()
        self.metadata["smooth_block_pose"] = self.block_base_poses[self.smooth_block_side].tolist()
        self.metadata["target_block"] = "rough"
        self.metadata["target_plate"] = self.yellow_plate.cfg.name
        self.metadata["target_pose"] = self.target_pose.tolist()
        self.metadata["block_plate_mapping"] = {
            "rough": self.yellow_plate.cfg.name,
            "smooth": self.blue_plate.cfg.name,
        }

    def _release_reset_constraints(self):
        self._actor_manager.remove_animate(force=True)

    def _resolve_side(self, side_cfg):
        if side_cfg == "random":
            return str(self.rng.choice(BLOCK_SIDES))
        return str(side_cfg)

    @staticmethod
    def _other_side(side):
        return "right" if side == "left" else "left"

    def _get_observations(self):
        return _zero_dark_camera_pixels(super()._get_observations())

    def build_instruction(self) -> str:
        return "Find the rough block by touch and place it on the yellow plate."

    def pre_move(self):
        self.delay(10)
        self.move(self.atom.open_gripper(0.5), tag="open_gripper_for_roughness_sort")

    def _grasp_block(self, block, label, side):
        grasp_rotate = self.rng.uniform(-np.pi / 36, np.pi / 36)
        block_pose = block.get_pose()
        target_pose = block.get_pose().add_bias([0.0, 0.0, 0.035 + 0.01 * self.rng.random()])\
            .add_rotation([0, grasp_rotate, 0])
        target_mat = target_pose.to_transformation_matrix()
        cpose = construct_grasp_pose(
            target_pose.p,
            target_mat[:3, 2],
            target_mat[:3, 0],
        )
        cid = block.register_point(cpose, type="contact")
        self.move(self.atom.grasp_actor(
            block,
            contact_point_id=cid,
            pre_dis=GRASP_PRE_DISTANCE,
            dis=0.0,
            is_close=False,
        ), tag=f"approach_{side}_{label}_block")

        gripper_qpos = self.rng.uniform(*GRIPPER_CLOSE_QPOS_RANGE) / 0.039
        self.move(self.atom.close_gripper(gripper_qpos), tag=f"close_{side}_{label}_block")
        return {
            "side": side,
            "label": label,
            "block_pose_before_grasp": block_pose.tolist(),
            "grasp_rotate_rad": float(grasp_rotate),
            "grasp_rotate_deg": float(np.rad2deg(grasp_rotate)),
            "gripper_qpos": float(gripper_qpos),
            "gripper_qpos_ratio": float(gripper_qpos / self._robot_manager.gripper_max_qpos),
            "grasp_pose": cpose.tolist(),
        }

    def _release_wrong_block(self, label, return_pose):
        self.metadata["wrong_block_hold_delay_steps"] = int(WRONG_BLOCK_HOLD_DELAY_STEPS)
        self.delay(WRONG_BLOCK_HOLD_DELAY_STEPS, is_save=True)
        self.move(self.atom.open_gripper(0.5), tag=f"release_{label}_block")
        self.move(
            self.atom.move_to_pose(return_pose),
            tag=f"return_to_initial_pose_after_{label}_block",
        )
        self.delay(5, is_save=False)

    def _place_rough_block_on_yellow_plate(self):
        lift_height = LIFT_HEIGHT + self.rng.uniform(-LIFT_HEIGHT_NOISE, LIFT_HEIGHT_NOISE)
        self.move(self.atom.move_by_displacement(z=lift_height), tag="lift_rough_block")

        self.metadata["lift_height"] = float(lift_height)

        self.move(self.atom.place_actor(
            self.rough_block,
            target_pose=self.target_pose,
            pre_dis=0.0,
            dis=0.0,
            is_open=False,
        ), tag="place_rough_block_on_yellow_plate", time_dilation_factor=0.5)
        self.delay(20, is_save=False)

    def _play_once(self):
        initial_ee_pose = self._robot_manager.get_ee_pose()
        first_label = self.side_to_label[self.initial_grasp_side]
        second_side = self._other_side(self.initial_grasp_side)
        second_label = self.side_to_label[second_side]

        self.metadata["initial_ee_pose_before_first_grasp"] = initial_ee_pose.tolist()
        self.metadata["first_grasp_side"] = self.initial_grasp_side
        self.metadata["first_grasp_block"] = first_label
        self.metadata["second_grasp_side"] = second_side
        self.metadata["second_grasp_block"] = second_label
        self.metadata["first_grasp_block_pose"] = (
            self.side_to_block[self.initial_grasp_side].get_pose().tolist()
        )
        self.metadata["second_grasp_block_pose"] = (
            self.side_to_block[second_side].get_pose().tolist()
        )

        first_grasp = self._grasp_block(
            self.side_to_block[self.initial_grasp_side],
            first_label,
            self.initial_grasp_side,
        )
        self.metadata["first_grasp"] = first_grasp

        if first_label == "rough":
            self.metadata["released_first_block"] = False
            self._place_rough_block_on_yellow_plate()
            return

        self.metadata["released_first_block"] = True
        self._release_wrong_block(first_label, initial_ee_pose)

        second_grasp = self._grasp_block(
            self.side_to_block[second_side],
            second_label,
            second_side,
        )
        self.metadata["second_grasp"] = second_grasp
        if second_label != "rough":
            raise RuntimeError("Expected the second grasped block to be rough")
        self._place_rough_block_on_yellow_plate()

    def check_success(self):
        if self.target_pose is None:
            return False

        block_pose = self.rough_block.get_pose().rebase(self.target_pose)
        xy_threshold = 0.04
        z_threshold = 0.01
        upright_score = np.dot(
            block_pose.to_transformation_matrix()[:3, 2],
            np.array([0, 0, 1]),
        )
        success = np.all(np.abs(block_pose.p) < np.array([xy_threshold, xy_threshold, z_threshold])) and \
            upright_score > 0.965

        self.metadata["success_diagnostics"] = {
            "target_block": "rough",
            "target_plate": self.yellow_plate.cfg.name,
            "rough_block_pose_in_target": block_pose.tolist(),
            "xy_threshold": float(xy_threshold),
            "z_threshold": float(z_threshold),
            "upright_score": float(upright_score),
            "success": bool(success),
        }
        return bool(success)
