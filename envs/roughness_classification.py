from ._base_task import *
import numpy as np
import torch

TASK_INSTRUCTION = "Classify the block surface by touch: place the smooth block on the blue plate and the rough block on the yellow plate."

FILTER_DARK_CAMERA_PIXELS = True
DARK_CAMERA_PIXEL_THRESHOLD = 10
GRIPPER_CLOSE_QPOS_RANGE = (0.0035, 0.004)
BLOCK_RESET_XY_NOISE = 0.01
LIFT_HEIGHT = 0.05
LIFT_HEIGHT_NOISE = 0.01
TARGET_XY_NOISE = 0.01

START_POSE = Pose([0.38, 0.0, 0.002], [1, 0, 0, 0])
SMOOTH_STANDBY_POSE = Pose([0.38, 1.0, 0.002], [1, 0, 0, 0])
ROUGH_STANDBY_POSE = Pose([0.38, -1.0, 0.002], [1, 0, 0, 0])


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
    roughness_label: Literal["random", "smooth", "rough"] = "random"
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


class Task(BaseTask):
    def __init__(self, cfg: TaskCfg, mode: Literal['collect', 'eval'] = 'collect', render_mode: str | None = None, **kwargs):
        if cfg.roughness_label not in ("random", "smooth", "rough"):
            raise ValueError("roughness_label must be 'random', 'smooth', or 'rough'")

        cfg.sim.physics_material.dynamic_friction = 2.5
        cfg.sim.physics_material.static_friction = 2.5
        cfg.uipc_sim.contact.default_friction_ratio = 2.5
        super().__init__(cfg, mode, render_mode, **kwargs)

    def create_actors(self):
        self.blue_plate = self._actor_manager.add_from_usd_file(
            name='blue_plate',
            asset_path="roughness_task/blue_square_plate.usd",
            pose=Pose([0.45, 0.08, 0.002], [1, 0, 0, 0]),
        )
        self.yellow_plate = self._actor_manager.add_from_usd_file(
            name='yellow_plate',
            asset_path="roughness_task/yellow_square_plate.usd",
            pose=Pose([0.45, -0.08, 0.002], [1, 0, 0, 0]),
        )

        self.smooth_block = self._actor_manager.add_from_usd_file(
            name='smooth_block',
            asset_path="roughness_task/black_smooth_cuboid.usd",
            pose=SMOOTH_STANDBY_POSE,
        )
        self.rough_block = self._actor_manager.add_from_usd_file(
            name='rough_block',
            asset_path="roughness_task/black_rough_cuboid.usd",
            pose=ROUGH_STANDBY_POSE,
        )

    def _reset_actors(self):
        self.choice = self._resolve_roughness_label(self.cfg.roughness_label)
        start_pose = START_POSE.add_offset(Pose([
            self.rng.uniform(-BLOCK_RESET_XY_NOISE, BLOCK_RESET_XY_NOISE),
            self.rng.uniform(-BLOCK_RESET_XY_NOISE, BLOCK_RESET_XY_NOISE),
            0.0,
        ], [1, 0, 0, 0]))

        self.smooth_block.set_pose(SMOOTH_STANDBY_POSE)
        self.rough_block.set_pose(ROUGH_STANDBY_POSE)

        if self.choice == 'smooth':
            self.block = self.smooth_block
            self.target = self.blue_plate
            self.other_target = self.yellow_plate
        else:
            self.block = self.rough_block
            self.target = self.yellow_plate
            self.other_target = self.blue_plate
        self.block.set_pose(start_pose)
        self.target_pose = self.target.get_pose().add_bias([
            self.rng.uniform(-TARGET_XY_NOISE, TARGET_XY_NOISE),
            self.rng.uniform(-TARGET_XY_NOISE, TARGET_XY_NOISE),
            0.01
        ])
        self.metadata["roughness_label"] = self.choice
        self.metadata["roughness_label_cfg"] = self.cfg.roughness_label
        self.metadata["target_plate"] = self.target.cfg.name
        self.metadata["start_pose"] = start_pose.tolist()
        self.metadata["target_pose"] = self.target_pose.tolist()

    def _release_reset_constraints(self):
        self._actor_manager.remove_animate(force=True)

    def _resolve_roughness_label(self, roughness_label):
        if roughness_label == "random":
            return str(self.rng.choice(['smooth', 'rough']))
        return str(roughness_label)

    def _get_observations(self):
        return _zero_dark_camera_pixels(super()._get_observations())

    def pre_move(self):
        self.delay(10)

        self.move(self.atom.open_gripper(0.5))

    def _play_once(self):
        grasp_rotate = self.rng.uniform(-np.pi / 36, np.pi / 36)
        target_pose = self.block.get_pose().add_bias([0.0, 0.0, 0.035 + 0.01 * self.rng.random()])\
            .add_rotation([0, grasp_rotate, 0])
        target_mat = target_pose.to_transformation_matrix()
        cpose = construct_grasp_pose(
            target_pose.p,
            target_mat[:3, 2],
            target_mat[:3, 0]
        )
        cid = self.block.register_point(cpose, type='contact')
        self.move(self.atom.grasp_actor(
            self.block, contact_point_id=cid, pre_dis=0.04, dis=0.0, is_close=False
        ))
        gripper_qpos = self.rng.uniform(*GRIPPER_CLOSE_QPOS_RANGE) / 0.039
        self.move(self.atom.close_gripper(gripper_qpos))
        lift_height = LIFT_HEIGHT + self.rng.uniform(-LIFT_HEIGHT_NOISE, LIFT_HEIGHT_NOISE)
        self.move(self.atom.move_by_displacement(z=lift_height))

        self.move(self.atom.place_actor(
            self.block,
            target_pose=self.target_pose,
            pre_dis=0.0,
            dis=0.0,
            is_open=False
        ), time_dilation_factor=0.5)
        self.delay(20, is_save=False)

    def check_success(self):
        block_pose = self.block.get_pose().rebase(self.target_pose)
        xy_threshold = 0.04
        z_threshold = 0.01
        return np.all(np.abs(block_pose.p) < np.array([xy_threshold, xy_threshold, z_threshold])) and \
            np.dot(block_pose.to_transformation_matrix()[:3, 2], np.array([0, 0, 1])) > 0.965
