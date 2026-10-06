from ._base_task import *
import numpy as np


TASK_INSTRUCTION = "Classify the block hardness by touch: place the soft red block on the blue plate and the hard red block on the yellow plate."

GRIPPER_CLOSE_QPOS_RANGE = (0.0035, 0.004)
SOFT_BLOCK_YOUNGS_MODULUS = 0.1
BLOCK_RESET_XY_NOISE = 0.01
LIFT_HEIGHT = 0.05
LIFT_HEIGHT_NOISE = 0.01
TARGET_XY_NOISE = 0.01

START_POSE = Pose([0.38, 0.0, 0.002], [1, 0, 0, 0])
HARD_STANDBY_POSE = Pose([0.38, 1.0, 0.002], [1, 0, 0, 0])
SOFT_STANDBY_POSE = Pose([0.38, -1.0, 0.002], [1, 0, 0, 0])


@configclass
class TaskCfg(BaseTaskCfg):
    hardness_label: Literal["random", "soft", "hard"] = "random"
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
        if cfg.hardness_label not in ("random", "soft", "hard"):
            raise ValueError("hardness_label must be 'random', 'soft', or 'hard'")

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

        self.hard_block = self._actor_manager.add_from_usd_file(
            name='hard_red_block',
            asset_path="roughness_task/red_smooth_cuboid.usd",
            pose=HARD_STANDBY_POSE,
            density=1e3,
        )
        self.soft_block = self._actor_manager.add_from_usd_file(
            name='soft_red_block',
            asset_path="roughness_task/red_smooth_cuboid.usd",
            pose=SOFT_STANDBY_POSE,
            constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(
                youngs_modulus=SOFT_BLOCK_YOUNGS_MODULUS,
                poisson_rate=0.45,
            ),
            density=300,
        )

    def _reset_actors(self):
        self.choice = self._resolve_hardness_label(self.cfg.hardness_label)
        start_pose = START_POSE.add_offset(Pose([
            self.rng.uniform(-BLOCK_RESET_XY_NOISE, BLOCK_RESET_XY_NOISE),
            self.rng.uniform(-BLOCK_RESET_XY_NOISE, BLOCK_RESET_XY_NOISE),
            0.0,
        ], [1, 0, 0, 0]))

        self.hard_block.set_pose(HARD_STANDBY_POSE)
        self.soft_block.set_pose(SOFT_STANDBY_POSE)

        if self.choice == 'soft':
            self.block = self.soft_block
            self.target = self.blue_plate
            self.other_target = self.yellow_plate
        else:
            self.block = self.hard_block
            self.target = self.yellow_plate
            self.other_target = self.blue_plate

        self.block.set_pose(start_pose)
        self.target_pose = self.target.get_pose().add_bias([
            self.rng.uniform(-TARGET_XY_NOISE, TARGET_XY_NOISE),
            self.rng.uniform(-TARGET_XY_NOISE, TARGET_XY_NOISE),
            0.01
        ])
        self.metadata["hardness_label"] = self.choice
        self.metadata["hardness_label_cfg"] = self.cfg.hardness_label
        self.metadata["target_plate"] = self.target.cfg.name
        self.metadata["start_pose"] = start_pose.tolist()
        self.metadata["target_pose"] = self.target_pose.tolist()

    def _release_reset_constraints(self):
        self._actor_manager.remove_animate(force=True)

    def _resolve_hardness_label(self, hardness_label):
        if hardness_label == "random":
            return str(self.rng.choice(['soft', 'hard']))
        return str(hardness_label)

    def pre_move(self):
        self.delay(10)

        self.move(self.atom.open_gripper(0.5), tag="open_gripper_for_hardness_block")

    def _play_once(self):
        grasp_rotate = self.rng.uniform(-np.pi / 36, np.pi / 36)
        target_pose = self.block.get_pose().add_bias([0.0, 0.0, 0.04 + 0.01 * self.rng.random()])\
            .add_rotation([0, grasp_rotate, 0])
        target_mat = target_pose.to_transformation_matrix()
        cpose = construct_grasp_pose(
            target_pose.p,
            target_mat[:3, 2],
            target_mat[:3, 0]
        )
        cid = self.block.register_point(cpose, type='contact')
        self.move(self.atom.grasp_actor(
            self.block,
            contact_point_id=cid,
            pre_dis=0.04,
            dis=0.0,
            is_close=False,
        ), tag=f"approach_{self.choice}_block")

        gripper_qpos = self.rng.uniform(*GRIPPER_CLOSE_QPOS_RANGE) / 0.039
        self.move(self.atom.close_gripper(gripper_qpos), tag=f"close_{self.choice}_block")
        lift_height = LIFT_HEIGHT + self.rng.uniform(-LIFT_HEIGHT_NOISE, LIFT_HEIGHT_NOISE)
        self.move(self.atom.move_by_displacement(z=lift_height), tag=f"lift_{self.choice}_block")

        self.metadata["grasp_rotate_rad"] = float(grasp_rotate)
        self.metadata["grasp_rotate_deg"] = float(np.rad2deg(grasp_rotate))
        self.metadata["soft_block_youngs_modulus"] = float(SOFT_BLOCK_YOUNGS_MODULUS)
        self.metadata["gripper_qpos"] = float(gripper_qpos)
        self.metadata["gripper_qpos_ratio"] = float(gripper_qpos / self._robot_manager.gripper_max_qpos)
        self.metadata["lift_height"] = float(lift_height)
        self.metadata["target_pose"] = self.target_pose.tolist()

        self.move(self.atom.place_actor(
            self.block,
            target_pose=self.target_pose,
            pre_dis=0.0,
            dis=0.0,
            is_open=False
        ), tag=f"place_{self.choice}_block", time_dilation_factor=0.5)
        self.delay(20, is_save=False)

    def check_success(self):
        block_pose = self.block.get_pose().rebase(self.target_pose)
        xy_threshold = 0.04
        z_threshold = 0.01
        success = np.all(np.abs(block_pose.p) < np.array([xy_threshold, xy_threshold, z_threshold])) and \
            np.dot(block_pose.to_transformation_matrix()[:3, 2], np.array([0, 0, 1])) > 0.965
        self.metadata["success_diagnostics"] = {
            "hardness_label": self.choice,
            "target_plate": self.target.cfg.name,
            "block_pose_in_target": block_pose.tolist(),
            "xy_threshold": float(xy_threshold),
            "z_threshold": float(z_threshold),
            "success": bool(success),
        }
        return success
