import sys
import json
import time
import os
import torch
import pickle
import torchvision
import h5py
import transforms3d as t3d

from envs.utils.data import HDF5Handler, VideoHandler
from warp import Function
import isaaclab
import numpy as np
from pathlib import Path
from typing import Generator, Literal
import decorator

import carb
import omni.ui
import logging
from contextlib import suppress
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.prims import XFormPrim
with suppress(ImportError):
    # isaacsim.gui is not available when running in headless mode.
    import isaacsim.gui.components.ui_utils as ui_utils

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    quat_error_magnitude,
    sample_uniform,
    wrap_to_pi,
)
from isaaclab.utils.noise import (
    GaussianNoiseCfg,
    NoiseModelCfg,
    UniformNoiseCfg,
    gaussian_noise,
)

from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg
from tacex_uipc import (
    UipcRLEnv,
    UipcIsaacAttachments,
    UipcIsaacAttachmentsCfg,
    UipcObject,
    UipcObjectCfg,
    UipcSimCfg,
)
from tacex_uipc.utils import TetMeshCfg

from typing import Any

from ._global import *
from .utils import *
from .robot.robot import RobotManager
from .robot.robot_cfg import *
from .sensors.camera import CameraManager, CameraCfg
from .sensors.tactile import TactileManager, TactileCfg, create_tactile_cfg


def _action_to_numpy(action):
    """Convert policy action tensors to a flat CPU numpy array."""

    if isinstance(action, torch.Tensor):
        return action.detach().cpu().numpy().reshape(-1)
    return np.asarray(action).reshape(-1)


def _apply_world_rotvec_delta(pose: Pose, delta_xyz, delta_rotvec):
    """Apply a world/base-frame xyz + rotation-vector delta to an EEF pose."""

    delta_xyz = np.asarray(delta_xyz, dtype=np.float64).reshape(3)
    delta_rotvec = np.asarray(delta_rotvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(delta_rotvec))
    if angle < 1e-12:
        delta_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        delta_q = t3d.quaternions.axangle2quat(delta_rotvec / angle, angle)

    target_p = pose.p + delta_xyz
    target_q = t3d.quaternions.qmult(delta_q, pose.q)
    target_q = target_q / np.linalg.norm(target_q)
    return Pose(target_p, target_q)


XENSE_WRIST_Y_ALIGNMENT_OFFSET = -np.pi / 4


def resolve_scene_background_texture(background: str | os.PathLike | None) -> Path:
    text = "base5" if background is None else str(background).strip()
    if not text:
        text = "base5"

    candidate = Path(text)
    if candidate.suffix != ".exr":
        candidate = SCENE_ASSETS_ROOT / f"{text}.exr"
    elif not candidate.is_absolute():
        candidate = SCENE_ASSETS_ROOT / candidate

    if not candidate.exists():
        available = ", ".join(sorted(path.stem for path in SCENE_ASSETS_ROOT.glob("base*.exr")))
        raise FileNotFoundError(
            f"Scene background {text!r} not found at {candidate}. Available backgrounds: {available}"
        )
    return candidate


def apply_xense_wrist_y_alignment(joint_pos: dict[str, float]) -> dict[str, float]:
    joint_pos = dict(joint_pos)
    if "panda_joint7" in joint_pos:
        joint_pos["panda_joint7"] = (
            float(joint_pos["panda_joint7"]) + XENSE_WRIST_Y_ALIGNMENT_OFFSET
        )
    return joint_pos


SENSOR_RUNTIME_DEFAULTS: dict[str, dict[str, Any]] = {
    "gsmini": {
        "video_size": (960, 320),
        "reset_time_limit": 120.0,
        "reset_first_frame_steps": 5,
        "reset_after_actor_steps": 20,
        "reset_final_steps": 5,
    },
    "neote": {
        "video_size": (960, 320),
        "reset_time_limit": 120.0,
        "reset_first_frame_steps": 5,
        "reset_after_actor_steps": 20,
        "reset_final_steps": 5,
    },
    "xensews": {
        "video_size": (1760, 700),
        "reset_time_limit": 600.0,
        "reset_first_frame_steps": 1,
        "reset_after_actor_steps": 20,
        "reset_final_steps": 5,
    },
}


@configclass
class BaseTaskCfg(DirectRLEnvCfg):
    logger_level = "error"
    debug_vis = False

    # viewer settings
    viewer: ViewerCfg = ViewerCfg()
    viewer.eye = (0.6, 0.15, 0.05)
    viewer.lookat = (-3.0, -4.5, -0.6)

    step_lim = 300
    planner_ignore_actors: tuple[str, ...] = ()

    save_dir = "auto"
    obs_data_type = {}

    save_frequency = 2
    video_frequency = 2
    render_frequency = 0
    video_size: tuple[int, int] | None = None

    ui_window_class_type = BaseEnvWindow

    # Video Save Config.
    save_pre_move = False
    skip_pre_move = False
    uniform_policy_recording = False
    final_acceptance_contract = False
    tactile_video_key = "rgb_marker"

    # Resolved from SENSOR_RUNTIME_DEFAULTS after selecting the tactile sensor.
    reset_first_frame_steps: int | None = None
    reset_after_actor_steps: int | None = None
    reset_final_steps: int | None = None
    reset_render_warmup_steps: int = 32
    eval_step_timeout_seconds: float = 0.0
    xense_marker_reference_max_settle_steps = 180
    xense_marker_reference_stable_steps = 8
    eval_start_delay_steps = 20

    decimation = 1
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1/120,
        render_interval=decimation,
        # device="cpu",
        physx=PhysxCfg(
            enable_ccd=True,  # needed for more stable ball_rolling
            # bounce_threshold_velocity=10000,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            restitution=0.0,
        )
    )

    uipc_sim = UipcSimCfg(
        # logger_level="Info"
        dt=sim.dt,
        ground_height=0.001,
        contact=UipcSimCfg.Contact(
            d_hat=0.0005,
            enable_friction=True,
            eps_velocity=0.1
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=1.5,
        replicate_physics=True,
        lazy_sensor_update=True,
    )

    # light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75), intensity=1500.0,
            texture_file=str(SCENE_ASSETS_ROOT / 'base5.exr')
        ),
    )
    background = "base5"

    # plate
    plate = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ground_plate",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0, 0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(SCENE_ASSETS_ROOT / "plate01.usda"),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                kinematic_enabled=True,
            ),
        ),
    )

    use_adaptive_grasp: bool = True
    adaptive_grasp_depth_threshold = None # in mm
    # Optional per-task Xense adaptive thresholds.  If unset, each close uses
    # adaptive_grasp_depth_threshold.  Larger values stop earlier / press less.
    xense_usb_adaptive_grasp_depth_threshold: float | None = None
    xense_half_cylinder_adaptive_grasp_depth_threshold: float | None = 25.8
    xense_insert_half_cylinder_adaptive_grasp_depth_threshold: float | None = 25.8
    xense_cube_adaptive_grasp_depth_threshold: float | None = 26.0
    xense_cup_adaptive_grasp_depth_threshold: float | None = 26.6
    xense_pour_cup_adaptive_grasp_depth_threshold: float | None = 26.6
    xense_drawer_adaptive_grasp_depth_threshold: float | None = 26.0
    xense_gear_adaptive_grasp_depth_threshold: float | None = 26.0
    # Optional per-task Xense contact stop policies.  If unset, each close uses
    # xense_adaptive_grasp_require_both_contacts.  Rigid/centered grasps usually
    # benefit from both pads contacting; soft cups often need an earlier any-pad stop.
    xense_usb_adaptive_grasp_require_both_contacts: bool | None = None
    xense_half_cylinder_adaptive_grasp_require_both_contacts: bool | None = None
    xense_insert_half_cylinder_adaptive_grasp_require_both_contacts: bool | None = None
    xense_cube_adaptive_grasp_require_both_contacts: bool | None = None
    xense_cup_adaptive_grasp_require_both_contacts: bool | None = False
    xense_pour_cup_adaptive_grasp_require_both_contacts: bool | None = None
    xense_drawer_adaptive_grasp_require_both_contacts: bool | None = None
    xense_gear_adaptive_grasp_require_both_contacts: bool | None = None
    # Xense-specific grasp tuning. These are ignored by GelSight/Neote branches.
    xense_usb_close_percent: float = 0.10
    xense_half_cylinder_close_percent: float = 0.185
    xense_insert_half_cylinder_close_percent: float = 0.185
    xense_cube_close_percent: float = 0.30
    xense_cup_close_percent: float = 0.76
    xense_cup_min_principal_ratio: float = 0.90
    xense_cup_max_nonrigid_error: float = 0.08
    xense_pour_cup_close_percent: float = 0.86
    xense_pour_ball_friction_ratio: float = 0.05
    xense_pour_grip_friction_ratio: float = 4.0
    xense_pour_wrist_angle_deg: float = 100.0
    xense_pour_wrist_steps: int = 160
    xense_pour_wrist_translation_x: float = 0.0
    xense_pour_wrist_translation_y: float = -0.030
    xense_pour_wrist_translation_z: float = 0.0
    xense_pour_actor_tilt_deg: float = 0.0
    xense_pour_actor_tilt_axis_x: float = 1.0
    xense_pour_actor_tilt_axis_y: float = 0.0
    xense_pour_actor_tilt_axis_z: float = 0.0
    xense_pour_carry_segments: int = 8
    xense_pour_carry_settle_steps: int = 0
    xense_pour_hold_actor_during_carry: bool = False
    xense_pour_target_y_offset: float = 0.035
    xense_pour_target_z_offset: float = 0.106
    xense_pour_cup_grasp_height_bias: float = -0.020
    xense_pour_release_lift: float = 0.0
    xense_pour_release_snap_angle_deg: float = 25.0
    xense_pour_release_snap_steps: int = 30
    xense_pour_release_snap_cycles: int = 0
    xense_pour_fix_cup_during_release: bool = True
    xense_pour_release_retract_x: float = 0.0
    xense_pour_release_carry_y: float = 0.100
    xense_drawer_close_percent: float = 0.15
    xense_gear_close_percent: float = 0.30
    xense_half_cylinder_grasp_height_bias: float = 0.010
    xense_insert_half_cylinder_grasp_height_bias: float = 0.010
    xense_cube_grasp_height_bias: float = 0.008
    xense_cup_grasp_height_bias: float = -0.010
    xense_pour_cup_grasp_world_x_bias: float = 0.0
    xense_drawer_grasp_z_bias: float = 0.0
    xense_half_cylinder_grasp_world_y_bias: float = 0.0
    xense_insert_half_cylinder_grasp_world_y_bias: float = 0.0
    xense_cube_grasp_world_y_bias: float = 0.0
    xense_gear_grasp_world_y_bias: float = 0.0
    xense_initial_settle_steps: int = 1
    xense_half_cylinder_initial_settle_steps: int = 1
    xense_cup_initial_settle_steps: int = 1
    xense_pour_initial_settle_steps: int = 1
    xense_drawer_initial_settle_steps: int = 1
    xense_insert_half_cylinder_initial_settle_steps: int = 1
    xense_cube_initial_settle_steps: int = 1
    xense_carry_time_dilation: float = 0.8
    xense_carry_segments: int = 6
    xense_carry_max_step: float = 0.04
    xense_post_close_settle_steps: int = 20
    xense_usb_post_close_settle_steps: int = 80
    xense_adaptive_grasp_max_steps: int = 180
    # Extra Robotiq tracking steps after the nominal plan_gripper path.
    # Xense/Robotiq physical qpos lags the commanded target, so contact often
    # appears during this short monitored tail rather than during the original
    # plan.  Keeping this bounded preserves runtime while avoiding blind
    # post-close squeezing.
    xense_adaptive_grasp_tail_steps: int = 80
    xense_adaptive_grasp_check_interval: int = 2
    xense_adaptive_grasp_target_tolerance: float = 0.006
    # Robotiq needs a tiny sustained closing command after tactile stop;
    # otherwise the physical finger joint can relax and drop the object even
    # though contact was detected.  The hold target is capped by the requested
    # close qpos, so the per-task close_percent remains the hard maximum.
    xense_adaptive_grasp_hold_margin: float = 0.012
    xense_adaptive_grasp_hold_velocity: float = 0.12
    xense_adaptive_grasp_min_steps_before_contact: int = 2
    xense_adaptive_grasp_min_travel: float = 0.0
    xense_adaptive_grasp_require_both_contacts: bool = True
    reset_time_limit: float | None = None  # in seconds

    cameras: list[CameraCfg] = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(pos=(0.554, 1.0, 0.150), rot=(0, 0, 0.707, 0.707), convention="opengl"),
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=1.94, focus_distance=1.0, horizontal_aperture=2.688, clipping_range=(0.01, 100.0)
            ),
            width=480,
            height=270,
            update_period=1/120
        ),
        CameraCfg(
            name="wrist",
            prim_path="/World/envs/env_.*/Robot/WristCamera/Camera",
            data_types=["rgb", "depth"],
            spawn=None, # use existing camera
            width=480,
            height=270,
            update_period=1/120,
        )
    ]

    robot: RobotCfg = None
    tactile_sensor_type:Literal['gsmini', 'xensews', 'neote'] = 'gsmini'
    dense_gelpad: bool = False
    force_field_grid: tuple[int, int] = (64, 48)

    planner_time_dilation_factor: float = 1.0

    gaussian_noise_cfg: GaussianNoiseCfg = GaussianNoiseCfg(mean=0.0, std=0.002, operation="add")
    
    random_texture: bool = False
    keep_contact: bool = False
    max_save_frames: int = 1000

    # some filler values, needed for DirectRLEnv
    episode_length_s = 0
    action_space = 0
    observation_space = 0
    state_space = 0

class BaseTask(UipcRLEnv):
    cfg: BaseTaskCfg

    PHASE_PRE_MOVE = 0
    PHASE_POLICY = 1
    PHASE_TERMINAL = 2
    PHASE_NAMES = {
        PHASE_PRE_MOVE: 'pre_move',
        PHASE_POLICY: 'policy',
        PHASE_TERMINAL: 'terminal',
    }

    @staticmethod
    def apply_scene_background(cfg: BaseTaskCfg) -> None:
        texture_file = resolve_scene_background_texture(getattr(cfg, "background", "base5"))
        cfg.background = texture_file.stem
        cfg.light.spawn.texture_file = str(texture_file)

    def __init__(self, cfg: BaseTaskCfg, mode:Literal['collect', 'eval'] = 'collect', render_mode=None, **kwargs):
        self.apply_scene_background(cfg)
        cfg = self.load_robot_and_sensors(cfg)
        
        self.cfg = cfg
        self.render_outdated = True

        self._setup_save()
        self.rng = np.random.default_rng()
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs) # Full Render

        self.logger = logging.getLogger(name=self.__class__.__name__)
        self.logger.setLevel(getattr(logging, self.cfg.logger_level.upper(), logging.ERROR))

        self.mode = mode
        self.first_frame = None
        
        self.start_time = 0.0
        self.step_count = 0
        self.save_count = 0
        self.last_render = -1
        self.step_cost = np.zeros(20)
        self.last_step = time.perf_counter()
        self.mean_steps = 0
        self.take_action_cnt = 0
        self.plan_success = True
        self.eval_success = False
        self.phase_id = self.PHASE_PRE_MOVE
        self.in_pre_move = True
        self.policy_start_step = None
        self.policy_start_saved_index = None
        self.policy_step_count = 0
        self.terminal_reason = None
        self.last_saved_phase_id = None
        self.phase_saved_counts = {
            self.PHASE_PRE_MOVE: 0,
            self.PHASE_POLICY: 0,
            self.PHASE_TERMINAL: 0,
        }
        self.last_qpos = None
        self.keep_still_times = 0
        self.atom_tag = ''
        self.atom_id = 0
        self.log = ''
        self.metadata = {}
 
        self.instruction = ""
        self.video_handler = VideoHandler()
        if getattr(self.cfg, "final_acceptance_contract", False) and self.cfg.video_frequency > 0:
            self.video_handler.fps = 1.0 / (
                self.cfg.sim.dt * self.cfg.decimation * self.cfg.video_frequency)

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self._robot_manager.setup()
        self._camera_manager.setup()
        self._tactile_manager.setup()
        self._tactile_manager.set_debug_vis(self.cfg.debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)
        self.save_pre_move = getattr(self.cfg, "save_pre_move", False)

    def load_robot_and_sensors(self, cfg:BaseTaskCfg):
        data_type = ["camera_depth", "tactile_rgb", "marker_rgb", "marker_motion"]
        if cfg.tactile_sensor_type == 'gsmini':
            cfg.robot = create_franka_gsmini_gripper(data_type=data_type, dense_gelpad=cfg.dense_gelpad)
        elif cfg.tactile_sensor_type == 'neote':
            cfg.robot = create_franka_neote_gripper(data_type=data_type, dense_gelpad=cfg.dense_gelpad)
        elif cfg.tactile_sensor_type == 'xensews':
            cfg.robot = create_franka_xensews_gripper(data_type=data_type)
        else:
            raise ValueError(f'Unknown tactile sensor type: {cfg.tactile_sensor_type}')

        runtime_defaults = SENSOR_RUNTIME_DEFAULTS[cfg.tactile_sensor_type]
        for key, value in runtime_defaults.items():
            if getattr(cfg, key, None) is None:
                setattr(cfg, key, value)
        
        if cfg.adaptive_grasp_depth_threshold is None:
            cfg.adaptive_grasp_depth_threshold = cfg.robot.adaptive_grasp_depth_threshold
        return cfg
 
    def _setup_save(self):
        if self.cfg.save_dir == "auto":
            module_name = self.__class__.__module__
            module = sys.modules[module_name]
            file_name = Path(module.__file__).stem
            if self.mode == 'collect':
                save_dir = Path('./data') / file_name
            else:
                save_dir = Path('./eval_result') / file_name
        else:
            save_dir = self.cfg.save_dir

        self.save_root = Path(save_dir)
        self.save_root.mkdir(parents=True, exist_ok=True)
        self.tmp_save_dir = self.save_root / '.cache' / str(self.cfg.seed)
        self.save_path = self.save_root / 'hdf5' / f'{self.cfg.seed}.hdf5'
        self.save_video_path = self.save_root / 'video' / f'{self.cfg.seed}.mp4'
        self.metadata_path = self.save_root / 'metadata.json'

        self.cfg.uipc_sim.workspace = str(self.save_root / 'scene')

    def _setup_scene(self):
        '''
            call once when initializing the environment
        '''
        self._setup_base_scene()
        self.scene.clone_environments(copy_from_source=False)
        
        self._actor_manager = ActorManager(self)
        self.create_actors()

        # add sensors
        self._camera_manager = CameraManager(self.cfg.cameras, self)
        self._tactile_manager = TactileManager(self.cfg.robot.tactiles, self)

    def _setup_base_scene(self):
        # add robot
        self._robot_manager:RobotManager = RobotManager(
            robot_cfg=self.cfg.robot,
            task=self,
            planner_time_dilation_factor=self.cfg.planner_time_dilation_factor
        )
        self.atom:Atom = Atom(self)

        self.plate = RigidObject(self.cfg.plate)

        # add lights
        self.cfg.light.spawn.func(self.cfg.light.prim_path, self.cfg.light.spawn)
    
    def create_noise(self, vec=[0.0, 0.0, 0.0], euler=[0.0, 0.0, 0.0]) -> Pose:
        '''Create random noise pose'''
        return Pose.create_noise(vec=vec, euler=euler, rng=self.rng)

    def timer(self, name):
        def log(*msg):
            # with open('log.log', 'a') as f:
            #     f.write(' '.join(str(m) for m in msg) + '\n')
            pass
        if not hasattr(self, '_timers'):
            self._timers = {}
        if name not in self._timers:
            self._timers[name] = time.perf_counter()
        else:
            log(f'[{self.step_count:>3d}][{name:^20}] cost: {(time.perf_counter() - self._timers[name])*1000:.2f} ms')
            self._timers.pop(name)

    def pre_move(self):
        pass

    def prepare_initial_state(self):
        pass

    def build_instruction(self) -> str:
        """Build an instruction after episode randomization has selected its goal."""
        return str(getattr(sys.modules[self.__class__.__module__], 'TASK_INSTRUCTION', ''))

    def set_instruction(self, instruction: str | None):
        self.instruction = '' if instruction is None else str(instruction)
        if self.instruction:
            self.metadata['instruction'] = self.instruction
        else:
            self.metadata.pop('instruction', None)

    def get_episode_context(self) -> dict[str, Any]:
        """Return optional task context without forcing text into every observation."""
        context = {
            'instruction': self.instruction,
            'task': Path(sys.modules[self.__class__.__module__].__file__).stem,
            'seed': int(self.cfg.seed),
        }
        if getattr(self.cfg, "final_acceptance_contract", False):
            from ._force_task_utils import timing_context
            context['timing_contract'] = timing_context(self)
        return context

    def _set_phase(self, phase_id: int, terminal_reason: str | None = None):
        if phase_id not in self.PHASE_NAMES:
            raise ValueError(f'Unknown task phase id: {phase_id}')
        self.phase_id = int(phase_id)
        self.in_pre_move = self.phase_id == self.PHASE_PRE_MOVE
        if self.phase_id == self.PHASE_TERMINAL:
            self.terminal_reason = terminal_reason or self.terminal_reason or 'terminal'

    def create_actors(self):
        pass

    def seed(self, seed:int = -1):
        seed = super().seed(seed)
        self.cfg.seed = seed
        self.rng = np.random.default_rng(seed)
        self._setup_save()
    
    def show_scene(self, actor_names:list[str]=None, show_next:bool=True):
        import trimesh
        geos = []
        
        if actor_names is None:
            actor_names = list(self._actor_manager.actors.keys())

        for actor_name in actor_names:
            if actor_name not in self._actor_manager.actors:
                continue
            actor = self._actor_manager.actors[actor_name]
            p1, p2 = actor.vertices, actor.next_pts
            geos.append(trimesh.PointCloud(
                p1, colors=[0, 0, 0]
            ))
            if show_next and p2 is not None:
                geos.append(trimesh.PointCloud(
                    p2, colors=[255, 0, 0]
                ))
        trimesh.Scene(geos).show()

    def reset(self, seed:int=-1, instructions:list[str]|None=None, options:dict[str, Any]|None=None):
        self.seed(seed)
        ret = super().reset()
        
        if self.first_frame is not None:
            self.uipc_sim.replay_frame(self.first_frame)

        total_cost = time.perf_counter() - self.start_time
        if total_cost > self.cfg.reset_time_limit:
            raise RuntimeError(
                f'Timeout: reset exceed time limit of {self.cfg.reset_time_limit} s, cost {total_cost} s.'
            )

        if self.cfg.video_frequency > 0:
            self.video_handler.reset(self.save_video_path, self.cfg.video_size)
        self._set_phase(self.PHASE_PRE_MOVE)
        # Marker motion is relative to the post-reset FEM reference. Do not
        # persist settling frames captured before that reference exists.
        save_reset_steps = False
        if self.first_frame is None:
            reset_test_start = time.perf_counter()
            for _ in range(int(getattr(self.cfg, 'reset_first_frame_steps', 5))):
                self._step(is_save=False)
                reset_test_cost = time.perf_counter() - reset_test_start
                if reset_test_cost > self.cfg.reset_time_limit:
                    raise RuntimeError(
                        f'Timeout: reset exceed time limit of {self.cfg.reset_time_limit} s, cost {reset_test_cost} s.'
                    )
            self._update_render()

            self.first_frame = self.uipc_sim.world.frame()
            self.uipc_sim.save_frame()

        if hasattr(self, '_reset_actors'):
            self._reset_actors()
            # Actor.set_pose() only stages UIPC constraint data. Apply it
            # before stepping so the reset settling loop runs at the target.
            self._actor_manager.update(dt=0.0)

            reset_test_start = time.perf_counter()
            for _ in range(int(getattr(self.cfg, 'reset_after_actor_steps', 20))):
                self._step(is_save=save_reset_steps)
                reset_test_cost = time.perf_counter() - reset_test_start
                if reset_test_cost > self.cfg.reset_time_limit:
                    raise RuntimeError(
                        f'Timeout: reset exceed time limit of {self.cfg.reset_time_limit} s, cost {reset_test_cost} s.'
                    )
            self._update_render()

        if instructions is not None:
            self.set_instruction(self.rng.choice(instructions))
        else:
            self.set_instruction(self.build_instruction())
        self.metadata['episode_context'] = self.get_episode_context()
        
        reset_test_start = time.perf_counter()
        for _ in range(int(getattr(self.cfg, 'reset_final_steps', 5))):
            self._step(is_save=save_reset_steps)
            reset_test_cost = time.perf_counter() - reset_test_start
            if reset_test_cost > self.cfg.reset_time_limit:
                raise RuntimeError(
                    f'Timeout: reset exceed time limit of {self.cfg.reset_time_limit} s, cost {reset_test_cost} s.'
                )
        for _ in range(int(getattr(self.cfg, 'reset_render_warmup_steps', 32))):
            self._update_render()

        is_xsense = str(self.cfg.tactile_sensor_type).startswith('xense')
        marker_reference_settle_steps = 0
        if is_xsense:
            max_settle_steps = max(
                0,
                int(getattr(self.cfg, 'xense_marker_reference_max_settle_steps', 180)),
            )
            required_stable_steps = max(
                1,
                int(getattr(self.cfg, 'xense_marker_reference_stable_steps', 8)),
            )
            marker_reference_stable_steps = 0
            marker_reference_reset_result = {
                name: False for name in self._tactile_manager.tactiles
            }
            while marker_reference_settle_steps <= max_settle_steps:
                marker_reference_reset_result = (
                    self._tactile_manager.reset_marker_reference()
                )
                if all(marker_reference_reset_result.values()):
                    marker_reference_stable_steps += 1
                else:
                    marker_reference_stable_steps = 0
                if marker_reference_stable_steps >= required_stable_steps:
                    break
                if marker_reference_settle_steps >= max_settle_steps:
                    break
                self._step(is_save=False)
                self._update_render()
                marker_reference_settle_steps += 1
            if marker_reference_stable_steps < required_stable_steps:
                raise RuntimeError(
                    'XSense FEM attachments did not settle into a stable Marker '
                    f'reference within {max_settle_steps} simulation steps: '
                    f'{marker_reference_reset_result}, consecutive_valid='
                    f'{marker_reference_stable_steps}/{required_stable_steps}'
                )
            self.metadata['xense_marker_reference_reset_result'] = (
                marker_reference_reset_result
            )
            self.metadata['xense_marker_reference_reset_step'] = int(self.step_count)
            self.metadata['xense_marker_reference_settle_steps'] = int(
                marker_reference_settle_steps
            )
            self.metadata['xense_marker_reference_stable_steps'] = int(
                marker_reference_stable_steps
            )
            self.metadata['xense_marker_reference_required_stable_steps'] = int(
                required_stable_steps
            )
        else:
            marker_reference_reset_result = self._tactile_manager.reset_marker_reference()
            self.metadata['marker_reference_reset_result'] = marker_reference_reset_result
            self.metadata['marker_reference_reset_step'] = int(self.step_count)

        self.prepare_initial_state()

        if hasattr(self, '_release_reset_constraints'):
            self._release_reset_constraints()
            self._actor_manager.update(dt=0.0)
            reset_test_start = time.perf_counter()
            for _ in range(int(getattr(self.cfg, 'reset_final_steps', 5))):
                self._step(is_save=save_reset_steps)
                reset_test_cost = time.perf_counter() - reset_test_start
                if reset_test_cost > self.cfg.reset_time_limit:
                    raise RuntimeError(
                        f'Timeout: reset exceed time limit of {self.cfg.reset_time_limit} s, cost {reset_test_cost} s.'
                    )

        run_pre_move = not bool(getattr(self.cfg, 'skip_pre_move', False))
        if options is not None and 'run_pre_move' in options:
            run_pre_move = bool(options['run_pre_move'])
        if run_pre_move:
            self.pre_move()

        # A short preparation phase can fall entirely between periodic save
        # steps. Preserve one real handoff observation so every episode keeps
        # an explicit pre_move -> policy boundary for BC/RL consumers.
        if (
            self.mode == 'collect'
            and self.save_pre_move
            and self.phase_saved_counts[self.PHASE_PRE_MOVE] == 0
        ):
            self._update_render()
            boundary_observation = self._get_observations()
            self.save_observations(boundary_observation)
            if getattr(self.cfg, "final_acceptance_contract", False) and self.cfg.video_frequency > 0:
                self.video_handler.write(self.get_frame_shot(boundary_observation))

        # update render to avoid artifacts
        for _ in range(5):
            self._update_render()

        if self.mode == 'eval':
            eval_start_delay_steps = int(getattr(self.cfg, 'eval_start_delay_steps', 20))
            if eval_start_delay_steps > 0:
                self.delay(eval_start_delay_steps)

        # The policy boundary starts only after all scripted preparation and
        # evaluation warm-up steps have completed.
        if getattr(self.cfg, "final_acceptance_contract", False):
            for _ in range((-self.step_count) % self.cfg.policy_action_repeat):
                self._step(is_save=False)
        self._set_phase(self.PHASE_POLICY)
        self.policy_start_step = int(self.step_count)
        self.policy_step_count = 0

        # ToBeCheck.
        if not self.save_pre_move:
            self.atom_id = 0
            self.atom_tag = ''

        handoff_obs = self._get_observations()
        handoff_info = {
            'phase': self.PHASE_NAMES[self.phase_id],
            'policy_start_sim_step': int(self.policy_start_step),
            'episode_context': self.get_episode_context(),
        }
        if isinstance(ret, tuple) and len(ret) == 2:
            if isinstance(ret[1], dict):
                handoff_info = {**ret[1], **handoff_info}
            return handoff_obs, handoff_info
        return handoff_obs

    # def _reset_actors(self):
    #     pass

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        if self.cfg.random_texture:
            Actor._set_texture('/World/envs/env_0/ground_plate', 'random', self.rng)
        self._actor_manager._reset_idx(self.rng)
        self._robot_manager._reset_idx()
        self._tactile_manager._reset_idx()

        self.plan_success = True
        self.eval_success = False
        self.step_count = 0
        self.save_count = 0
        self.step_cost = np.zeros(20)
        self.last_step = time.perf_counter()
        self.last_render = -1
        self.take_action_cnt = 0
        self.current_goal_idx = 0
        self.render_outdated = True
        self.start_time = time.perf_counter()
        self.last_qpos = None
        self.keep_still_times = 0
        self.policy_start_step = None
        self.policy_start_saved_index = None
        self.policy_step_count = 0
        self.terminal_reason = None
        self._set_phase(self.PHASE_PRE_MOVE)
        self.last_saved_phase_id = None
        self.phase_saved_counts = {
            self.PHASE_PRE_MOVE: 0,
            self.PHASE_POLICY: 0,
            self.PHASE_TERMINAL: 0,
        }
        self.metadata = {}
        self.log = ''

    def pause(self):
        self.sim.pause()

    def _update_render(self):
        self.uipc_sim.update_render_meshes()
        self._actor_manager.sync_visuals()
        self.sim.render()
        
        dt = self.physics_dt * self.cfg.decimation * max(1, self.step_count - self.last_render)
        self.scene.update(dt=dt)
        self._actor_manager.update(dt=dt)
        self._camera_manager.update(dt=dt, force_recompute=True)
        self._tactile_manager.update(dt=dt, force_recompute=True)
 
        self.last_render = self.step_count
    
    def get_frame_shot(self, obs):
        head_obs = obs['observation']['head']['rgb'].clone()
        wrist_obs = obs['observation']['wrist']['rgb'].clone()
        tac_size = 160
        tactile_key = getattr(self.cfg, "tactile_video_key", "rgb_marker")

        def to_hwc3(image):
            image = image.clone().to(device=head_obs.device)
            if image.dim() == 2:
                image = image.unsqueeze(-1)
            if image.dim() != 3:
                raise RuntimeError(f"Expected image tensor with 2 or 3 dims, got {tuple(image.shape)}")
            if image.shape[-1] not in (1, 3, 4) and image.shape[0] in (1, 3, 4):
                image = image.permute(1, 2, 0)
            if image.shape[-1] == 4:
                image = image[..., :3]
            if image.shape[-1] == 1:
                image = image.expand(-1, -1, 3)
            if image.shape[-1] != 3:
                raise RuntimeError(f"Expected HWC image with 1, 3 or 4 channels, got {tuple(image.shape)}")
            if image.dtype != head_obs.dtype:
                image = image.to(dtype=head_obs.dtype)
            return image

        def resize_hwc(image, size):
            image = to_hwc3(image)
            return torchvision.transforms.Resize(size)(image.permute(2, 0, 1)).permute(1, 2, 0)

        def resize_hwc_if_needed(image, size):
            image = to_hwc3(image)
            if tuple(image.shape[:2]) == tuple(size):
                return image
            return torchvision.transforms.Resize(size)(image.permute(2, 0, 1)).permute(1, 2, 0)

        def pad_height(image, target_height):
            image = to_hwc3(image)
            if image.shape[0] == target_height:
                return image
            if image.shape[0] > target_height:
                return resize_hwc(image, (target_height, image.shape[1]))
            pad_top = (target_height - image.shape[0]) // 2
            padded = torch.zeros(
                (target_height, image.shape[1], 3),
                dtype=image.dtype,
                device=image.device,
            )
            padded[pad_top:pad_top + image.shape[0], :, :] = image
            return padded

        def depth_to_rgb(depth):
            depth = depth.clone().to(device=head_obs.device)
            if depth.dim() == 3:
                if depth.shape[-1] == 1:
                    depth = depth[..., 0]
                elif depth.shape[0] == 1:
                    depth = depth.squeeze(0)
                else:
                    raise RuntimeError(f"Expected single-channel depth tensor, got {tuple(depth.shape)}")
            if depth.dim() != 2:
                raise RuntimeError(f"Expected depth tensor with 2 dims, got {tuple(depth.shape)}")

            depth = depth.to(dtype=torch.float32)
            finite = torch.isfinite(depth)
            if finite.any():
                valid = depth[finite]
                min_depth = valid.min()
                max_depth = valid.max()
                if (max_depth - min_depth).abs().item() > 1e-12:
                    depth = (depth - min_depth) / (max_depth - min_depth)
                    depth = torch.where(finite, depth, torch.zeros_like(depth))
                else:
                    depth = torch.zeros_like(depth)
            else:
                depth = torch.zeros_like(depth)

            if head_obs.dtype == torch.uint8:
                depth = (depth * 255.0).clamp(0, 255).to(dtype=head_obs.dtype)
            else:
                depth = depth.to(dtype=head_obs.dtype)
            return depth.unsqueeze(-1).expand(-1, -1, 3)

        is_xense = getattr(self.cfg, "tactile_sensor_type", "") in ("xensews", "xensews_robotiq")
        if is_xense:
            xense_tac_size = (700, 400)
            rgb_size = (320, 480)
            row_height = xense_tac_size[0]
            if tactile_key in ("rgb_marker", "rgb", "marker_force_img", "force_field_img", "gel_particle"):
                left_tac = resize_hwc_if_needed(obs['tactile']['left_tactile'][tactile_key], xense_tac_size)
                right_tac = resize_hwc_if_needed(obs['tactile']['right_tactile'][tactile_key], xense_tac_size)
            elif tactile_key == "depth":
                left_tac = resize_hwc_if_needed(depth_to_rgb(obs['tactile']['left_tactile'][tactile_key]), xense_tac_size)
                right_tac = resize_hwc_if_needed(depth_to_rgb(obs['tactile']['right_tactile'][tactile_key]), xense_tac_size)
            else:
                raise RuntimeError(f"Unknown tactile key: {tactile_key}")

            head_tile = pad_height(resize_hwc(head_obs, rgb_size), row_height)
            wrist_tile = pad_height(resize_hwc(wrist_obs, rgb_size), row_height)
            return torch.cat((head_tile, wrist_tile, left_tac, right_tac), dim=1)

        if tactile_key in ("rgb_marker", "rgb", "marker_force_img", "force_field_img", "gel_particle"):
            left_tac = resize_hwc(obs['tactile']['left_tactile'][tactile_key], (tac_size, tac_size))
            right_tac = resize_hwc(obs['tactile']['right_tactile'][tactile_key], (tac_size, tac_size))
        elif tactile_key == "depth":
            left_tac = resize_hwc(depth_to_rgb(obs['tactile']['left_tactile'][tactile_key]), (tac_size, tac_size))
            right_tac = resize_hwc(depth_to_rgb(obs['tactile']['right_tactile'][tactile_key]), (tac_size, tac_size))
        else:
            raise RuntimeError(f"Unknown tactile key: {tactile_key}")

        img = torch.zeros((320, 480*2+160, 3), dtype=head_obs.dtype, device=head_obs.device)
        img[:, :480, :] = resize_hwc(head_obs, (320, 480))
        img[:, 480:480*2, :] = resize_hwc(wrist_obs, (320, 480))
        img[:tac_size, 480*2:, :] = left_tac
        img[tac_size:, 480*2:, :] = right_tac
        return img

    @staticmethod
    def _step_callback(status:dict):
        mode = status['mode']
        is_save = status['is_save']
        atom_id = status['atom_id']
        atom_tag = status['atom_tag']
        step_count = status['step_count']
        total_cost = status['total_cost']
        mean_steps = status['mean_steps']
        step_mean_cost = status['step_mean_cost']
        take_action_cnt = status['take_action_cnt']
        step_status = f'FPS {1/step_mean_cost:6.2f}, Running {total_cost:7.2f}s'

        if mean_steps > 0.0:
            if mode == 'eval':
                step_percent = f'({take_action_cnt / mean_steps * 100:6.2f}%)'
            else:
                step_percent = f'({step_count / mean_steps * 100:6.2f}%)'
        else:
            step_percent = '(   N/A%)'

        log = ''
        if mode == 'collect':
            atom_status = f'Atom ID: {atom_id:>2d}, Tag: {atom_tag:<15}'
            log = (f'Step {step_count:>5d}{step_percent}'
                    f', save {is_save}, {step_status}, {atom_status}')
        elif mode == 'eval_test':
            log = (f'Step {step_count:>5d}{step_percent}, testing     '
                  f', {step_status}')
        else:
            if not is_save:
                log = (f'Step {step_count:>5d}{step_percent}, pre moving  '
                      f', {step_status}')
            else:
                log = (f'Step {step_count:>5d}{step_percent}, action {take_action_cnt:>5d}'
                      f', {step_status}')
        return log

    def _step(self, is_save:bool=True):
        if self.plan_success is False:
            return 
        
        self.step_count += 1

        # is_save = is_save and (not self.in_pre_move) and (not self.mode == 'eval_test')
        # Modify for Save the Whole Trajectory
        is_save = is_save and (self.save_pre_move or not self.in_pre_move) and (not self.mode == 'eval_test')
        save_freq = (self.cfg.save_frequency > 0 and self.step_count % self.cfg.save_frequency == 0)
        video_freq = (self.cfg.video_frequency > 0 and self.step_count % self.cfg.video_frequency == 0)
        render_freq = (self.cfg.render_frequency > 0 and self.step_count % self.cfg.render_frequency == 0)

        self.scene.write_data_to_sim()
        for _ in range(self.cfg.decimation):
            self.sim.step(render=False)

        if render_freq or (self.mode == 'collect' and is_save and save_freq) or (is_save and video_freq) \
            or self.mode == 'eval':
            self._update_render()

        obs = None
        if self.mode == 'collect' and is_save and save_freq:
            obs = self._get_observations()
            self.save_observations(obs)

            def check(d):
                depth = d[50:-50, 50:-50]
                if depth.min() == depth.max():
                    return False
                return True
            if self.cfg.keep_contact:
                if not check(obs['tactile']['left_tactile']['depth']) \
                    or not check(obs['tactile']['right_tactile']['depth']):
                        self.plan_success = False
            if self.save_count > self.cfg.max_save_frames-1:
                self.plan_success = False
 
        if is_save and video_freq:
            if obs is None:
                obs = self._get_observations()
            self.video_handler.write(self.get_frame_shot(obs))

        step_mean_cost = 0.0
        step_cost = time.perf_counter() - self.last_step
        self.last_step = time.perf_counter()
        self.step_cost[(self.step_count-1) % 20] = step_cost
        if self.step_count <= len(self.step_cost):
            step_mean_cost = np.mean(self.step_cost[:self.step_count])
        else:
            step_mean_cost = np.mean(self.step_cost)
        total_cost = time.perf_counter() - self.start_time

        status_dict = {
            'mode': self.mode,
            'is_save': is_save,
            'mean_steps': self.mean_steps,
            'step_count': self.step_count,
            'take_action_cnt': self.take_action_cnt,
            'atom_id': self.atom_id,
            'atom_tag': self.atom_tag,
            'step_mean_cost': step_mean_cost,
            'step_cost': step_cost,
            'total_cost': total_cost
        }
        self.log = self._step_callback(status_dict)
        print(self.log+' '*5, end='\r')
    
    def _play_once(self):
        pass

    def play_once(self):
        ret = self._play_once()
        if ret is not None:
            self.metadata.update(ret)
        self._save_metadata()

    def _get_observations(self):
        phase_id = int(self.phase_id)
        policy_step = int(self.policy_step_count) if phase_id != self.PHASE_PRE_MOVE else -1
        obs = {
            'observation': {},
            'embodiment': {},
            'tactile': {},
            'actor': {},
            'step': self.step_count,
            'atom': {
                'id': self.atom_id,
                'tag': self.atom_tag
            },
            'phase': {
                'id': phase_id,
                'name': self.PHASE_NAMES[phase_id],
                'sim_step': int(self.step_count),
                'policy_step': policy_step,
                'is_boundary': int(self.last_saved_phase_id != phase_id),
            }
        }

        if 'embodiment' in self.cfg.obs_data_type:
            obs['embodiment'] = self._robot_manager.get_observations(self.cfg.obs_data_type['embodiment'])
        if 'camera' in self.cfg.obs_data_type:
            obs['observation'] = self._camera_manager.get_observations(self.cfg.obs_data_type['camera'])
        if 'tactile' in self.cfg.obs_data_type:
            obs['tactile'] = self._tactile_manager.get_observations(self.cfg.obs_data_type['tactile'])
        if 'actor' in self.cfg.obs_data_type:
            obs['actor'] = self._actor_manager.get_observations()
        return obs
    
    def clean_cache(self, mean_steps:float=0.0, result:str=None):
        self.mean_steps = mean_steps
        if self.tmp_save_dir.exists():
            for f in self.tmp_save_dir.iterdir():
                f.unlink()
            self.tmp_save_dir.rmdir()
        if self.cfg.video_frequency > 0:
            self.video_handler.close(result)
        if result is not None:
            self.metadata['cost_step'] = self.step_count
            self.metadata['cost_time'] = time.perf_counter() - self.start_time
            self.metadata['result'] = result
            self._save_metadata()
 
    def save_to_hdf5(self):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        if getattr(self.cfg, "final_acceptance_contract", False):
            handler = HDF5Handler(allow_empty_strings=True)
        else:
            handler = HDF5Handler()
        handler.pkls_to_hdf5(self.tmp_save_dir, self.save_path)
        with h5py.File(self.save_path, 'a') as hdf5_file:
            phase_group = hdf5_file.require_group('phase')
            phase_group.attrs['schema_version'] = 2
            phase_group.attrs['pre_move_id'] = self.PHASE_PRE_MOVE
            phase_group.attrs['policy_id'] = self.PHASE_POLICY
            phase_group.attrs['action_id'] = self.PHASE_POLICY
            phase_group.attrs['terminal_id'] = self.PHASE_TERMINAL
            phase_group.attrs['policy_start_sim_step'] = int(self.policy_start_step or 0)
            phase_group.attrs['policy_start_saved_index'] = int(
                self.policy_start_saved_index
                if self.policy_start_saved_index is not None
                else -1
            )
            phase_group.attrs['pre_move_saved_frames'] = int(
                self.phase_saved_counts[self.PHASE_PRE_MOVE]
            )
            phase_group.attrs['policy_saved_frames'] = int(
                self.phase_saved_counts[self.PHASE_POLICY]
            )
            phase_group.attrs['action_saved_frames'] = int(
                self.phase_saved_counts[self.PHASE_POLICY]
            )
            phase_group.attrs['terminal_saved_frames'] = int(
                self.phase_saved_counts[self.PHASE_TERMINAL]
            )
            phase_group.attrs['save_frequency'] = int(self.cfg.save_frequency)
            phase_group.attrs['terminal_reason'] = self.terminal_reason or ''
            hdf5_file.attrs['instruction'] = self.instruction
            hdf5_file.attrs['episode_context_json'] = json.dumps(
                self.get_episode_context(), ensure_ascii=False
            )
        if 'vertex_force' in self.cfg.obs_data_type.get('tactile', []):
            try:
                self._tactile_manager.dump_force_field_meta(self.save_root)
            except Exception as exc:
                print(f"[force_field_meta] dump failed: {exc}")
    
    def _save_metadata(self):
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    all_metadata = json.load(f)
            except Exception as _:
                all_metadata = {}
        else:
            all_metadata = {}
        all_metadata[str(self.cfg.seed)] = self.metadata
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(all_metadata, f, ensure_ascii=False, indent=4)

    def save_observations(self, obs: dict):
        def to_cpu(data):
            if isinstance(data, dict):
                return {k: to_cpu(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [to_cpu(v) for v in data]
            elif isinstance(data, torch.Tensor):
                return data.cpu().numpy()
            else:
                return data

        self.tmp_save_dir.mkdir(parents=True, exist_ok=True)
        with open(self.tmp_save_dir / f'{self.save_count}.pkl', 'wb') as f:
            pickle.dump(to_cpu(obs), f)
        phase_id = int(obs['phase']['id'])
        if phase_id == 1 and self.policy_start_saved_index is None:
            self.policy_start_saved_index = int(self.save_count)
        self.phase_saved_counts[phase_id] += 1
        self.last_saved_phase_id = phase_id
        self.save_count += 1
 
    def check_success(self):
        return False
    
    def move(
        self,
        actions: list[Action],
        tag:str = 'move',
        is_save: bool = True,
        delay: bool = True,
        constraint_pose = None,
        time_dilation_factor = None,
        gripper_depth_threshold = None,
        gripper_require_both_contacts = None,
    ):
        """
        Take action for the robot.
        """
        if self.plan_success is False:
            return False
        
        self.atom_id += 1
        self.atom_tag = tag

        for idx, action in enumerate(actions):
            control_seq = {
                "arm": None,
                "gripper": None,
            }
            if action.action == 'move' or action.action == 'all':
                action.args['constraint_pose'] = action.args.get(
                    'constraint_pose', constraint_pose)
                action.args['time_dilation_factor'] = action.args.get(
                    'time_dilation_factor', time_dilation_factor)
                control_seq['arm'] = self._robot_manager.plan_arm(
                    action.target_pose,
                    pre_dis=action.args.get('pre_dis'),
                    constraint_pose=action.args['constraint_pose'],
                    time_dilation_factor=action.args['time_dilation_factor'],
                )
                if control_seq['arm']['status'] == 'Fail':
                    self.logger.error(f'Arm motion planning failed on action {idx}: {action.__str__()}')
                    if self.cfg.debug_vis:
                        add_visual_box(action.target_pose, 'failed_target')
                        self.delay(100)
                    self.plan_success = False
                    return False

                if self.cfg.debug_vis:
                    add_visual_box(action.target_pose, 'target')

            if action.action == 'gripper' or action.action == 'all':
                if self.mode in ['collect', 'eval_test'] or (self.mode == 'eval' and self.in_pre_move):
                    if self.cfg.use_adaptive_grasp:
                        target_pos = self._robot_manager.gripper_percent2qpos(action.target_gripper_pos)
                        action_depth_threshold = action.args.get('gripper_depth_threshold', None)
                        # A task-level move(..., gripper_depth_threshold=...) is the
                        # most specific tuning knob; otherwise use the action arg and
                        # finally the sensor default.  This keeps per-task YAML values
                        # from being shadowed by atom.close_gripper()'s auto default.
                        depth_threshold = (
                            gripper_depth_threshold
                            if gripper_depth_threshold is not None
                            else action_depth_threshold
                        )
                        is_opening = self._robot_manager.is_gripper_opening(target_pos)
                        if depth_threshold is None and not is_opening:
                            depth_threshold = self.cfg.adaptive_grasp_depth_threshold
                        action_require_both_contacts = action.args.get('gripper_require_both_contacts', None)
                        require_both_contacts = (
                            gripper_require_both_contacts
                            if gripper_require_both_contacts is not None
                            else action_require_both_contacts
                        )
                        control_seq['gripper'] = {
                            'status': 'success',
                            'num_steps': -1,
                            'target': target_pos,
                            'threshold': depth_threshold,
                            'require_both_contacts': require_both_contacts,
                        }
                    else:
                        control_seq['gripper'] = self._robot_manager.plan_gripper(
                            action.target_gripper_pos, type='percent'
                        )
                else:
                    control_seq['gripper'] = self._robot_manager.plan_gripper(
                        action.target_gripper_pos, type='qpos'
                    )
                if control_seq['gripper']['status'] == 'Fail':
                    self.logger.error(f'Gripper motion planning failed on action {idx}: {action.__str__()}')
                    self.plan_success = False
                    return False

            self.take_dense_action(control_seq, is_save)
            if delay:
                self.delay(10, is_save)
        self._update_render()
        return True

    def move_gripper_center_path(
        self,
        poses: list[Pose],
        tag: str,
        is_save: bool = True,
        delay: bool = False,
        settle_steps: int = 0,
        time_dilation_factor: float | None = None,
        metadata_prefix: str | None = None,
    ):
        """Move through gripper-center waypoints by converting each one to an EE target."""
        ee_poses = [self._robot_manager.gripper_center_to_ee(pose) for pose in poses]
        if metadata_prefix is not None:
            self.metadata[f"{metadata_prefix}_gripper_center_poses"] = [pose.tolist() for pose in poses]
            self.metadata[f"{metadata_prefix}_ee_poses"] = [pose.tolist() for pose in ee_poses]
            self.metadata[f"{metadata_prefix}_actual_gripper_center_poses"] = []
            self.metadata[f"{metadata_prefix}_actual_position_errors"] = []

        for idx, ee_pose in enumerate(ee_poses):
            move_tag = tag if len(ee_poses) == 1 else f"{tag}_{idx}"
            ok = self.move(
                [Action("move", target_pose=ee_pose)],
                tag=move_tag,
                is_save=is_save,
                delay=delay,
                time_dilation_factor=time_dilation_factor,
            )
            if not ok:
                return ee_poses
            if settle_steps > 0:
                self.delay(settle_steps, is_save=is_save)
            if metadata_prefix is not None:
                actual_pose = self._robot_manager.get_gripper_center_pose()
                self.metadata[f"{metadata_prefix}_actual_gripper_center_poses"].append(actual_pose.tolist())
                self.metadata[f"{metadata_prefix}_actual_position_errors"].append(
                    float(np.linalg.norm(actual_pose.p - poses[idx].p))
                )
        return ee_poses

    def gripper_center_pose_for_actor_target(self, actor, target_pose: Pose) -> Pose:
        """Compute the gripper-center pose that would place an in-hand actor at target_pose."""
        inhand_pose = actor.get_pose().rebase(to_coord=self._robot_manager.get_gripper_center_pose())
        target_gripper_center_mat = (
            target_pose.to_transformation_matrix()
            @ np.linalg.inv(inhand_pose.to_transformation_matrix())
        )
        return Pose.from_matrix(target_gripper_center_mat)

    def gripper_center_pose_for_actor_position_target(self, actor, target_position) -> Pose:
        """Translate the current gripper-center pose so the actor center reaches target_position."""
        current_gripper_center_pose = self._robot_manager.get_gripper_center_pose()
        target_position = np.asarray(target_position, dtype=float).reshape(3)
        actor_delta = target_position - actor.get_pose().p
        return current_gripper_center_pose.add_bias(actor_delta, coord="world")

    def move_actor_with_gripper_center_to_position(
        self,
        actor,
        target_position,
        tag: str,
        segments: int | None = None,
        settle_steps: int = 5,
        time_dilation_factor: float | None = None,
        metadata_prefix: str | None = None,
    ):
        current_pose = self._robot_manager.get_gripper_center_pose()
        target_pose = self.gripper_center_pose_for_actor_position_target(actor, target_position)
        segments = max(1, int(segments or getattr(self.cfg, "xense_carry_segments", 4)))
        if time_dilation_factor is None:
            time_dilation_factor = float(getattr(self.cfg, "xense_carry_time_dilation", 0.5))
        poses = []
        for idx in range(1, segments + 1):
            alpha = idx / segments
            p = current_pose.p + (target_pose.p - current_pose.p) * alpha
            poses.append(Pose(p, current_pose.q))
        if metadata_prefix is not None:
            self.metadata[f"{metadata_prefix}_actor_start_pose"] = actor.get_pose().tolist()
            self.metadata[f"{metadata_prefix}_actor_target_position"] = np.asarray(target_position, dtype=float).reshape(3).tolist()
        return self.move_gripper_center_path(
            poses,
            tag=tag,
            delay=False,
            settle_steps=settle_steps,
            time_dilation_factor=time_dilation_factor,
            metadata_prefix=metadata_prefix,
        )

    def move_actor_by_world_displacement_to_position(
        self,
        actor,
        target_position,
        tag: str,
        segments: int | None = None,
        settle_steps: int = 5,
        time_dilation_factor: float | None = None,
        metadata_prefix: str | None = None,
        actor_pose_hold: bool = False,
    ):
        """Move an in-hand actor by commanding bounded world-frame EE displacements."""
        target_position = np.asarray(target_position, dtype=float).reshape(3)
        requested_segments = max(1, int(segments or getattr(self.cfg, "xense_carry_segments", 4)))
        max_step = max(1e-6, float(getattr(self.cfg, "xense_carry_max_step", 0.04)))
        start_actor_position = actor.get_pose().p.copy()
        total_delta = target_position - start_actor_position
        auto_segments = max(1, int(np.ceil(float(np.linalg.norm(total_delta)) / max_step)))
        segments = max(requested_segments, auto_segments)
        if time_dilation_factor is None:
            time_dilation_factor = float(getattr(self.cfg, "xense_carry_time_dilation", 0.5))

        if metadata_prefix is not None:
            self.metadata[f"{metadata_prefix}_actor_start_pose"] = actor.get_pose().tolist()
            self.metadata[f"{metadata_prefix}_actor_target_position"] = target_position.tolist()
            self.metadata[f"{metadata_prefix}_start_actor_position"] = start_actor_position.tolist()
            self.metadata[f"{metadata_prefix}_total_delta"] = total_delta.tolist()
            self.metadata[f"{metadata_prefix}_requested_segments"] = int(requested_segments)
            self.metadata[f"{metadata_prefix}_segments"] = int(segments)
            self.metadata[f"{metadata_prefix}_max_step"] = float(max_step)
            self.metadata[f"{metadata_prefix}_commanded_step_deltas"] = []
            self.metadata[f"{metadata_prefix}_actual_actor_poses"] = []
            self.metadata[f"{metadata_prefix}_actual_ee_poses"] = []
            self.metadata[f"{metadata_prefix}_actual_position_errors"] = []
            self.metadata[f"{metadata_prefix}_gripper_center_poses"] = []
            self.metadata[f"{metadata_prefix}_actual_gripper_center_poses"] = []
            self.metadata[f"{metadata_prefix}_actor_pose_hold"] = bool(actor_pose_hold)

        start_actor_pose_for_hold = actor.get_pose()
        if np.linalg.norm(total_delta) < 1e-8:
            if metadata_prefix is not None:
                self.metadata[f"{metadata_prefix}_skipped"] = True
            return True

        current_gripper_center_pose = self._robot_manager.get_gripper_center_pose()
        target_gripper_center_pose = self.gripper_center_pose_for_actor_position_target(actor, target_position)
        step_delta = total_delta / float(segments)

        for idx in range(segments):
            alpha = (idx + 1) / float(segments)
            gripper_center_pos = (
                current_gripper_center_pose.p
                + (target_gripper_center_pose.p - current_gripper_center_pose.p) * alpha
            )
            gripper_center_pose = Pose(gripper_center_pos, current_gripper_center_pose.q)
            ee_pose = self._robot_manager.gripper_center_to_ee(gripper_center_pose)
            move_tag = tag if segments == 1 else f"{tag}_{idx}"

            if metadata_prefix is not None:
                self.metadata[f"{metadata_prefix}_commanded_step_deltas"].append(step_delta.tolist())
                self.metadata[f"{metadata_prefix}_gripper_center_poses"].append(gripper_center_pose.tolist())

            if actor_pose_hold:
                held_actor_pose = Pose(
                    start_actor_position + total_delta * alpha,
                    start_actor_pose_for_hold.q,
                )
                actor.set_pose(held_actor_pose)
                self._actor_manager.update(dt=0.0)
                if metadata_prefix is not None:
                    self.metadata.setdefault(f"{metadata_prefix}_held_actor_target_poses", []).append(
                        held_actor_pose.tolist()
                    )

            ok = self.move(
                [Action("move", target_pose=ee_pose)],
                tag=move_tag,
                delay=False,
                time_dilation_factor=time_dilation_factor,
            )
            if not ok:
                return False
            if settle_steps > 0:
                self.delay(settle_steps, is_save=True)

            if metadata_prefix is not None:
                actual_actor_pose = actor.get_pose()
                self.metadata[f"{metadata_prefix}_actual_actor_poses"].append(actual_actor_pose.tolist())
                self.metadata[f"{metadata_prefix}_actual_ee_poses"].append(self._robot_manager.get_ee_pose().tolist())
                self.metadata[f"{metadata_prefix}_actual_gripper_center_poses"].append(
                    self._robot_manager.get_gripper_center_pose().tolist()
                )
                self.metadata[f"{metadata_prefix}_actual_position_errors"].append(
                    float(np.linalg.norm(actual_actor_pose.p - target_position))
                )
        return True

    def get_xense_close_percent(self, key: str, fallback_key: str = "xense_usb_close_percent") -> float:
        if getattr(self.cfg, "tactile_sensor_type", "") not in ("xensews", "xensews_robotiq"):
            return 0.0
        if hasattr(self.cfg, key):
            return float(getattr(self.cfg, key))
        return float(getattr(self.cfg, fallback_key, 0.0))

    def get_xense_grasp_height_bias(self, key: str, default: float = 0.0) -> float:
        if getattr(self.cfg, "tactile_sensor_type", "") not in ("xensews", "xensews_robotiq"):
            return 0.0
        return float(getattr(self.cfg, key, default))

    def get_xense_adaptive_grasp_depth_threshold(
        self,
        key: str,
        fallback_key: str = "adaptive_grasp_depth_threshold",
    ) -> float | None:
        is_xense = getattr(self.cfg, "tactile_sensor_type", "") in (
            "xensews",
            "xensews_robotiq",
        )
        threshold = getattr(self.cfg, key, None) if is_xense else None
        if threshold is None:
            threshold = getattr(self.cfg, fallback_key, None)
        return None if threshold is None else float(threshold)

    def get_xense_adaptive_grasp_require_both_contacts(
        self,
        key: str,
        fallback_key: str = "xense_adaptive_grasp_require_both_contacts",
    ) -> bool | None:
        if getattr(self.cfg, "tactile_sensor_type", "") not in ("xensews", "xensews_robotiq"):
            return None
        value = getattr(self.cfg, key, None)
        if value is None:
            value = getattr(self.cfg, fallback_key, None)
        return None if value is None else bool(value)

    def record_xense_grasp_debug(self, prefix: str, actor: Actor | None = None):
        if getattr(self.cfg, "tactile_sensor_type", "") not in ("xensews", "xensews_robotiq"):
            return
        self.metadata[f"{prefix}_gripper_center_pose"] = self._robot_manager.get_gripper_center_pose().tolist()
        self.metadata[f"{prefix}_ee_pose"] = self._robot_manager.get_ee_pose().tolist()
        self.metadata[f"{prefix}_gripper_qpos"] = float(self._robot_manager.get_gripper_qpos())
        self.metadata[f"{prefix}_gripper_percent"] = float(self._robot_manager.get_gripper_percentage())
        if actor is not None:
            self.metadata[f"{prefix}_actor_pose"] = actor.get_pose().tolist()

    def settle_xense_after_close(self, is_save: bool = False):
        if getattr(self.cfg, "tactile_sensor_type", "") not in ("xensews", "xensews_robotiq"):
            return
        steps = max(0, int(getattr(self.cfg, "xense_post_close_settle_steps", 80)))
        if steps > 0:
            self.delay(steps, is_save=is_save)
 
    def delay(self, steps=20, is_save:bool=False, force:bool=False):
        if getattr(self.cfg, "final_acceptance_contract", False) and self.mode == "collect" and self.phase_id == self.PHASE_POLICY:
            is_save = True
        if not force and not self.plan_success:
            return False
        self.logger.info(f"Delaying for {steps} steps")
        self.atom_tag = 'delay'
        self.atom_id += 1
        for _ in range(steps):
            self._step(is_save)
        self._update_render()
        return True
 
    def take_dense_action(self, control_seq, is_save:bool=True):
        """
        control_seq:
            arm, gripper
        """
        arm_seq, gripper_seq = (
            control_seq['arm'],
            control_seq['gripper'],
        )

        arm_steps = arm_seq['num_steps'] if arm_seq is not None else 0
        gripper_steps = gripper_seq['num_steps'] if gripper_seq is not None else 0

        if gripper_steps == -1: # adaptive grasp
            idx, gripper_active = 0, True
            gripper_planner = self.adaptive_set_gripper(
                gripper_seq['target'],
                gripper_seq['threshold'],
                gripper_seq.get('require_both_contacts'),
            )
            while True:
                if idx >= arm_steps and not gripper_active:
                    break
                if arm_seq is not None and idx < arm_steps:
                    self._robot_manager.set_arm(
                        arm_seq['position'][idx],
                        arm_seq['velocity'][idx]
                    )
                if gripper_active:
                    pos, vel, gripper_active = next(gripper_planner)
                    self._robot_manager.set_gripper(pos, vel)
                self._step(is_save)
                idx += 1
        else:
            max_control_len = max(arm_steps, gripper_steps)
            for idx in range(max_control_len):
                if arm_seq is not None and idx < arm_steps:
                    self._robot_manager.set_arm(
                        arm_seq['position'][idx],
                        arm_seq['velocity'][idx]
                    )
                if gripper_steps is not None and idx < gripper_steps:
                    self._robot_manager.set_gripper(
                        gripper_seq['position'][idx],
                        gripper_seq['velocity'][idx]
                    )
                self._step(is_save)
        return True

    def check_early_stop(self):
        return False

    def get_rl_metrics(self):
        return {}

    def compute_rl_reward(self, previous_metrics, current_metrics, action, success):
        action_array = np.asarray(action, dtype=np.float32)
        control_penalty = 1e-3 * float(np.mean(np.square(action_array)))
        return (10.0 if success else 0.0) - control_penalty

    def check_rl_early_stop(self, metrics):
        return False

    def env_step(
        self,
        action,
        action_type: Literal['qpos', 'ee', 'delta_ee'] = 'qpos',
        force: bool = True,
        action_repeat: int | None = None,
        joint_velocity=None,
    ):
        if self.phase_id == self.PHASE_TERMINAL:
            raise RuntimeError('env_step() called after the episode reached TERMINAL')
        if self.phase_id != self.PHASE_POLICY:
            raise RuntimeError('env_step() is only valid after the POLICY handoff')
        if action_repeat is None:
            action_repeat = self.cfg.policy_action_repeat if getattr(self.cfg, "final_acceptance_contract", False) else 1
        if action_repeat < 1:
            raise ValueError('action_repeat must be at least 1')
        if joint_velocity is not None:
            joint_velocity = torch.as_tensor(
                joint_velocity, dtype=torch.float32, device=self.device
            )
        previous_metrics = self.get_rl_metrics()
        action_tensor = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if getattr(self.cfg, "final_acceptance_contract", False):
            exec_success, success = self.take_action(
                action_tensor, action_type=action_type, force=force,
                joint_velocity=joint_velocity, action_repeat=action_repeat)
        else:
            first_action = action_tensor
            initial_qpos = None
            if action_type == 'qpos' and action_repeat > 1:
                initial_qpos = self._robot_manager.get_observations(['joint'])['joint'][:len(action_tensor)]
                first_action = initial_qpos + (action_tensor - initial_qpos) / action_repeat
            exec_success, success = self.take_action(
                first_action,
                action_type=action_type,
                force=force,
                joint_velocity=joint_velocity,
            )
            for repeat_index in range(1, action_repeat):
                if not exec_success or success or action_type != 'qpos':
                    break
                interpolation = (repeat_index + 1) / action_repeat
                repeated_action = initial_qpos + (action_tensor - initial_qpos) * interpolation
                self._robot_manager.set_arm(
                    repeated_action[:-1],
                    vel=joint_velocity,
                    force=force,
                )
                self._robot_manager.set_gripper(repeated_action[-1], force=force)
                self._step()
                if self.check_success():
                    self.eval_success = True
                    success = True
                    break
        current_metrics = self.get_rl_metrics()
        reward = self.compute_rl_reward(
            previous_metrics,
            current_metrics,
            action_tensor.detach().cpu().numpy(),
            bool(success),
        )
        rl_early_stop = bool(self.check_rl_early_stop(current_metrics))
        task_early_stop = bool(self.check_early_stop())
        terminated = bool(success)
        truncated = bool(
            not exec_success
            or self.take_action_cnt >= self.cfg.step_lim
            or rl_early_stop
            or task_early_stop
        )
        self.policy_step_count += 1
        if not (getattr(self.cfg, "final_acceptance_contract", False) and self.phase_id == self.PHASE_TERMINAL):
            if terminated:
                self._set_phase(self.PHASE_TERMINAL, terminal_reason='success')
            elif truncated:
                if not exec_success:
                    terminal_reason = 'execution_failure'
                elif self.take_action_cnt >= self.cfg.step_lim:
                    terminal_reason = 'step_limit'
                elif rl_early_stop:
                    terminal_reason = 'rl_early_stop'
                else:
                    terminal_reason = 'task_early_stop'
                self._set_phase(self.PHASE_TERMINAL, terminal_reason=terminal_reason)
        observation = self._get_observations()
        info = {
            'exec_success': bool(exec_success),
            'success': bool(success),
            'rl_early_stop': rl_early_stop,
            'task_early_stop': task_early_stop,
            'take_action_count': int(self.take_action_cnt),
            'action_repeat': int(action_repeat),
            'metrics': current_metrics,
        }
        return observation, float(reward), terminated, truncated, info

    def take_action(
        self,
        action: torch.Tensor,
        action_type: Literal['qpos', 'ee', 'delta_ee', 'delta_ee_rotvec', 'delta_ee_rotvec_ik'] = 'qpos',
        force: bool = True,
        joint_velocity=None,
        action_repeat: int | None = None,
    ):
        '''
            qpos     : actions is Tensor([8]), qpos (7 DOFS + gripper)
            ee       : actions is Tensor([7]), position (3), orientation (4)
            delta_ee : actions is Tensor([6]), delta_position (3), delta_orientation (3)
            delta_ee_rotvec : actions is Tensor([7]), delta_position (3), delta_rotvec (3), delta_gripper (1)
            delta_ee_rotvec_ik : actions is Tensor([7]), Differential IK single-step servo, no planner
        '''
        if getattr(self.cfg, "final_acceptance_contract", False):
            from ._force_task_utils import take_force_task_action
            return take_force_task_action(
                self, action, action_type=action_type, force=force,
                joint_velocity=joint_velocity, action_repeat=action_repeat)
        if action_repeat not in (None, 1):
            raise ValueError("Use env_step(action_repeat=...) for legacy task repetition")
        if self.take_action_cnt >= self.cfg.step_lim or self.eval_success:
            return True, self.eval_success

        exec_success = True
        self.take_action_cnt += 1
        self.logger.info(f"step: {self.take_action_cnt} / {self.cfg.step_lim}")

        if action_type == 'ee':
            action_np = _action_to_numpy(action)
            target_pose = Pose(p=action_np[:3], q=action_np[3:7])
            target_gripper_pos = float(action_np[7])
            exec_success = self.move([
                Action(action='all', target_pose=target_pose, target_gripper_pos=target_gripper_pos)
            ], delay=False)
        elif action_type == 'delta_ee':
            action_np = _action_to_numpy(action)
            ee_pose = self._robot_manager.get_ee_pose()
            ee_next_pose = ee_pose.add_bias(action_np[:3], coord='world')\
                .add_rotation(euler=action_np[3:6].tolist(), coord='world')
            gripper_pos = self._robot_manager.get_gripper_qpos()
            gripper_next_pos = float(gripper_pos + action_np[6])
            exec_success = self.move([
                Action(action='all', target_pose=ee_next_pose, target_gripper_pos=gripper_next_pos)
            ], delay=False)
        elif action_type == 'delta_ee_rotvec':
            action_np = _action_to_numpy(action)
            ee_pose = self._robot_manager.get_ee_pose()
            ee_next_pose = _apply_world_rotvec_delta(
                ee_pose,
                delta_xyz=action_np[:3],
                delta_rotvec=action_np[3:6],
            )
            gripper_pos = self._robot_manager.get_gripper_qpos()
            gripper_next_pos = float(gripper_pos + action_np[6])
            exec_success = self.move([
                Action(action='all', target_pose=ee_next_pose, target_gripper_pos=gripper_next_pos)
            ], delay=False)
        elif action_type == 'delta_ee_rotvec_ik':
            exec_success = self._robot_manager.servo_delta_ee_rotvec(action, force=force)
            self._step()
        else:
            self._robot_manager.set_arm(
                action[:-1], vel=joint_velocity, force=force
            )
            self._robot_manager.set_gripper(action[-1], force=force)
            self._step()
        
        if self.check_success():
            self.eval_success = True

        return exec_success, self.eval_success

    def adaptive_set_gripper(self, qpos, depth_threshold:float=None, require_both_contacts: bool | None = None):
        if self.cfg.tactile_sensor_type in ("xensews", "xensews_robotiq"):
            yield from self._adaptive_set_xense_gripper(qpos, depth_threshold, require_both_contacts)
            return

        max_steps = 1000

        default_step, contact_step = 0.0005, 0.00005

        last_qpos = self._robot_manager.get_gripper_qpos()
        max_depth = (
            self.cfg.robot.tactile_far_plane
            * torch.ones_like(self._tactile_manager.get_min_depth())
        ) # mm
        if depth_threshold is not None:
            depth_threshold = depth_threshold * torch.ones_like(max_depth)
        qpos = float(qpos)
        direct = 'open' if self._robot_manager.is_gripper_opening(qpos) else 'close'
        step_sign = 1.0 if qpos > self._robot_manager.get_gripper_qpos() else -1.0
        step_size = step_sign * (contact_step if direct == 'open' else default_step)

        stop_reason = 'max_steps'
        for i in range(max_steps):
            current_qpos = self._robot_manager.get_gripper_qpos()
            tactile_depth = self._tactile_manager.get_min_depth()

            if direct == 'close':
                if torch.allclose(max_depth, tactile_depth, atol=1e-5):
                    step_size = step_sign * default_step
                elif depth_threshold is not None:
                    if torch.all(tactile_depth < depth_threshold):
                        stop_reason = 'depth_threshold'
                        break
                    else:
                        step_size = step_sign * min(
                            torch.min(torch.abs(tactile_depth - depth_threshold)).item()/1000,
                            contact_step
                        )
                else:
                    step_size = step_sign * default_step
            else:
                if torch.allclose(max_depth, tactile_depth, atol=1e-5):
                    step_size = step_sign * default_step
                if depth_threshold is not None:
                    if torch.all(tactile_depth > depth_threshold):
                        stop_reason = 'depth_threshold'
                        break
                    else:
                        step_size = step_sign * min(
                            torch.min(torch.abs(depth_threshold - tactile_depth)).item()/1000,
                            contact_step
                        )
                else:
                    step_size = step_sign * default_step

            if np.allclose(current_qpos, qpos, atol=1e-5):
                stop_reason = 'target'
                break
            elif np.abs(current_qpos - qpos) < np.abs(step_size):
                target_qpos = qpos
            else:
                target_qpos = current_qpos + step_size
            cmd_dim = len(self._robot_manager._gripper_ids)
            position = torch.full((cmd_dim,), target_qpos, device=self._robot_manager.device)
            velocity = torch.full_like(position, (target_qpos - current_qpos) / self.cfg.sim.dt)
            last_qpos = current_qpos
            yield position, velocity, True

        final_qpos = self._robot_manager.get_gripper_qpos()
        final_target_qpos = qpos if stop_reason == 'max_steps' else final_qpos
        cmd_dim = len(self._robot_manager._gripper_ids)
        final_position = torch.full((cmd_dim,), final_target_qpos, device=self._robot_manager.device)
        yield final_position, torch.zeros_like(final_position), False

    def _adaptive_set_xense_gripper(
        self,
        qpos,
        depth_threshold: float = None,
        require_both_contacts: bool | None = None,
    ):
        """Xense/Robotiq adaptive close.

        The important compatibility rule is that, when tactile contact does not
        reach the threshold, this follows RobotManager.plan_gripper(qpos) rather
        than inventing a different gripper trajectory.  Therefore enabling
        use_adaptive_grasp cannot slow down or otherwise perturb tasks that do
        not actually trigger tactile early-stop.  If threshold contact is seen,
        the plan is truncated at the current physical qpos, matching GelSight's
        "close target is a maximum, tactile contact may stop earlier" semantics.
        """
        qpos = float(qpos)
        max_steps = max(1, int(getattr(self.cfg, "xense_adaptive_grasp_max_steps", 180)))
        tail_steps = max(0, int(getattr(self.cfg, "xense_adaptive_grasp_tail_steps", 80)))
        check_interval = max(1, int(getattr(self.cfg, "xense_adaptive_grasp_check_interval", 2)))
        target_tolerance = max(0.0, float(getattr(self.cfg, "xense_adaptive_grasp_target_tolerance", 0.006)))
        hold_margin = max(0.0, float(getattr(self.cfg, "xense_adaptive_grasp_hold_margin", 0.0)))
        hold_velocity = max(0.0, float(getattr(self.cfg, "xense_adaptive_grasp_hold_velocity", 0.0)))
        min_steps_before_contact = max(0, int(getattr(self.cfg, "xense_adaptive_grasp_min_steps_before_contact", 2)))
        min_travel = max(0.0, float(getattr(self.cfg, "xense_adaptive_grasp_min_travel", 0.0)))
        if require_both_contacts is None:
            require_both_contacts = bool(getattr(self.cfg, "xense_adaptive_grasp_require_both_contacts", False))
        else:
            require_both_contacts = bool(require_both_contacts)

        direct = 'open' if self._robot_manager.is_gripper_opening(qpos) else 'close'
        cmd_dim = len(self._robot_manager._gripper_ids)
        threshold = None
        if depth_threshold is not None:
            threshold = torch.as_tensor(depth_threshold, dtype=torch.float32, device=self.device)

        start_qpos = float(self._robot_manager.get_gripper_qpos())
        target_direction = 1.0 if qpos >= start_qpos else -1.0
        last_depth = None
        last_contact_mask = None
        stop_reason = 'target'
        steps_run = 0
        plan_steps_run = 0
        tail_steps_run = 0

        gripper_plan = self._robot_manager.plan_gripper(qpos, type='qpos')
        plan_steps = int(gripper_plan.get('num_steps', 0) or 0)
        positions = gripper_plan.get('position')
        velocities = gripper_plan.get('velocity')
        steps_to_run = min(plan_steps, max_steps)
        plan_step = float(getattr(self._robot_manager, "gripper_plan_step", 0.0))

        def has_moved_enough_for_contact_stop(current_qpos: float, step_idx: int) -> bool:
            if direct != 'close':
                return True
            if step_idx < min_steps_before_contact:
                return False
            return abs(current_qpos - start_qpos) >= min_travel

        def contact_stop_reached(contact_mask: torch.Tensor) -> bool:
            return bool(torch.all(contact_mask)) if require_both_contacts else bool(torch.any(contact_mask))

        def should_check_depth(step_idx: int) -> bool:
            return (
                direct == 'close'
                and threshold is not None
                and (step_idx == 0 or step_idx % check_interval == 0 or step_idx == steps_to_run - 1)
            )

        def should_check_tail_depth(step_idx: int, tail_idx: int, tail_limit: int) -> bool:
            return (
                direct == 'close'
                and threshold is not None
                and (tail_idx == 0 or step_idx % check_interval == 0 or tail_idx == tail_limit - 1)
            )

        def physical_target_reached(current_qpos: float) -> bool:
            return target_direction * (current_qpos - qpos) >= -target_tolerance

        def plan_terminal_velocity() -> torch.Tensor:
            if velocities is None or plan_steps <= 0 or plan_steps_run <= 0:
                return torch.zeros((cmd_dim,), dtype=torch.float32, device=self._robot_manager.device)
            velocity = torch.as_tensor(
                velocities[min(plan_steps_run - 1, plan_steps - 1)],
                dtype=torch.float32,
                device=self._robot_manager.device,
            ).flatten()
            if velocity.numel() == 1 and cmd_dim != 1:
                velocity = velocity.repeat(cmd_dim)
            return velocity

        for i in range(steps_to_run):
            current_qpos = float(self._robot_manager.get_gripper_qpos())
            if should_check_depth(i):
                tactile_depth = self._tactile_manager.get_min_depth()
                contact_mask = tactile_depth < threshold
                last_depth = tactile_depth.detach().cpu().tolist()
                last_contact_mask = contact_mask.detach().cpu().tolist()
                if contact_stop_reached(contact_mask) and has_moved_enough_for_contact_stop(current_qpos, i):
                    stop_reason = 'depth_threshold'
                    break

            plan_steps_run = i + 1
            steps_run = plan_steps_run
            yield positions[i], velocities[i], True

        if plan_steps > max_steps and stop_reason != 'depth_threshold':
            stop_reason = 'max_steps'

        # Robotiq closes physically slower than plan_gripper's target sequence.
        # Contact often appears while the joint is still chasing the final
        # target, i.e. in the post-plan settle window.  Monitor a bounded tail
        # here so adaptive grasp can stop on tactile contact before the later
        # task-level settle blindly squeezes the object.
        if stop_reason != 'depth_threshold' and direct == 'close' and threshold is not None:
            tail_limit = min(tail_steps, max(0, max_steps - steps_run))
            tail_position = torch.full((cmd_dim,), qpos, device=self._robot_manager.device)
            tail_velocity = plan_terminal_velocity()
            for tail_idx in range(tail_limit):
                current_qpos = float(self._robot_manager.get_gripper_qpos())
                global_step_idx = plan_steps_run + tail_idx
                if should_check_tail_depth(global_step_idx, tail_idx, tail_limit):
                    tactile_depth = self._tactile_manager.get_min_depth()
                    contact_mask = tactile_depth < threshold
                    last_depth = tactile_depth.detach().cpu().tolist()
                    last_contact_mask = contact_mask.detach().cpu().tolist()
                    if contact_stop_reached(contact_mask) and has_moved_enough_for_contact_stop(current_qpos, global_step_idx):
                        stop_reason = 'depth_threshold'
                        break
                if physical_target_reached(current_qpos):
                    stop_reason = 'target_reached_physical'
                    break

                tail_steps_run = tail_idx + 1
                steps_run += 1
                yield tail_position, tail_velocity, True

        final_qpos = float(self._robot_manager.get_gripper_qpos())
        if stop_reason == 'depth_threshold':
            if direct == 'close':
                final_target_qpos = min(qpos, final_qpos + hold_margin)
            else:
                final_target_qpos = max(qpos, final_qpos - hold_margin)
        else:
            final_target_qpos = qpos
        final_position = torch.full((cmd_dim,), final_target_qpos, device=self._robot_manager.device)
        if stop_reason == 'depth_threshold':
            if direct == 'close' and hold_velocity > 0.0:
                final_velocity = torch.full_like(final_position, min(hold_velocity, self._robot_manager.gripper_velocity_limit))
            elif direct == 'open' and hold_velocity > 0.0:
                final_velocity = torch.full_like(final_position, -min(hold_velocity, self._robot_manager.gripper_velocity_limit))
            else:
                final_velocity = torch.zeros_like(final_position)
        elif steps_run <= 0 or velocities is None:
            final_velocity = torch.zeros_like(final_position)
        else:
            # Preserve the original plan_gripper terminal velocity when no
            # tactile early-stop happened.  Robotiq joints physically lag the
            # target, so replacing this with zero velocity makes adaptive mode
            # weaker than the fixed gripper plan even when it should be a no-op.
            final_velocity = plan_terminal_velocity()

        def _debug_threshold(value):
            if value is None:
                return None
            if torch.is_tensor(value):
                v = value.detach().cpu()
                return float(v.item()) if v.numel() == 1 else v.tolist()
            try:
                return float(value)
            except (TypeError, ValueError):
                return value

        debug = {
            'direct': direct,
            'start_qpos': start_qpos,
            'target_qpos': qpos,
            'final_qpos': final_qpos,
            'final_target_qpos': final_target_qpos,
            'stop_reason': stop_reason,
            'steps': int(steps_run),
            'plan_steps_run': int(plan_steps_run),
            'tail_steps_run': int(tail_steps_run),
            'plan_steps': int(plan_steps),
            'max_steps': int(max_steps),
            'tail_steps': int(tail_steps),
            'check_interval': int(check_interval),
            'qpos_step': float(plan_step),
            'depth_threshold': _debug_threshold(depth_threshold),
            'last_depth': last_depth,
            'last_contact_mask': last_contact_mask,
            'target_tolerance': float(target_tolerance),
            'hold_margin': float(hold_margin),
            'hold_velocity': float(hold_velocity),
            'min_steps_before_contact': int(min_steps_before_contact),
            'min_travel': float(min_travel),
            'require_both_contacts': bool(require_both_contacts),
            'follow_plan_gripper': True,
        }
        self.metadata.setdefault('xense_adaptive_grasp', []).append(debug)
        if os.environ.get("XENSE_ADAPTIVE_GRASP_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}:
            print(
                f"[xense-debug] adaptive_gripper done reason={stop_reason} "
                f"steps={steps_run}/{plan_steps}+{tail_steps_run}/{tail_steps} "
                f"qpos={final_qpos:.6f} target={qpos:.6f} "
                f"threshold={_debug_threshold(depth_threshold)} contact={last_contact_mask} "
                f"last_depth={last_depth} follow_plan=True"
            )
        yield final_position, final_velocity, False

    def gravity_rotate(self, actor:Actor, target_vec, target_axis=[0, 0, 1], is_save=True):
        if self.plan_success is False:
            return False
        
        max_steps = 200
        omega_threshold = 0.05
        contact_threshold = self.cfg.robot.contact_threshold # [min, max]
        target_axis = np.array(target_axis).reshape(3, 1)

        def get_axis():
            nonlocal actor, target_axis
            axis = (actor.get_pose().to_transformation_matrix()[:3, :3] @ target_axis).reshape(-1)
            axis /= np.linalg.norm(axis)
            return axis

        target_vec = np.array(target_vec) / np.linalg.norm(target_vec) 
        last_z = get_axis()
        last_theta = np.arccos(np.dot(last_z, target_vec))
        for _ in range(max_steps):
            curr_z = get_axis()
            curr_qpos = self._robot_manager.get_gripper_qpos()
            curr_depth = torch.min(self._tactile_manager.get_min_depth()).item()

            theta = np.arccos(np.dot(curr_z, target_vec))
            if theta < 0.05 or theta > last_theta:
                break
            omega = theta - last_theta
            last_theta = theta

            if np.abs(omega) < omega_threshold:
                if curr_depth < contact_threshold[1]:
                    curr_qpos += 0.0001
            elif curr_depth > contact_threshold[0]:
                curr_qpos -= 0.0001

            position = torch.tensor([curr_qpos, curr_qpos],
                                    dtype=torch.float32, device=self._robot_manager.device)
            velocity = torch.clip((position - curr_qpos)/self.cfg.sim.dt, -0.0001, 0.0001)
            self._robot_manager.set_gripper(position, velocity)

            for _ in range(5):
                self._step(is_save)
            last_z = curr_z
        self.move(self.atom.close_gripper())
    
    def gripper_rotate(self, actor:Actor, theta, steps:int=6, is_save=True):
        if self.plan_success is False:
            return False

        for i in range(steps):
            rpy = [0, theta/steps, 0]
            actor_pose = actor.get_pose()
            gripper_center_pose = self._robot_manager.get_gripper_center_pose()
            new_gripper_center = gripper_center_pose.add_rotation(rpy, coord=actor_pose)
            new_gripper_center.q = gripper_center_pose.q.copy()
            new_target_pose = self._robot_manager.gripper_center_to_ee(new_gripper_center)
            self.move([Action(
                action='move', target_pose=new_target_pose
            )], tag='rotate', is_save=is_save, delay=False, time_dilation_factor=0.5)
    
    def try_forward(self, actor:Actor, dis=0.01, delta_d=0.004, is_save=True):
        if self.plan_success is False:
            return False

        actor_last_pose = actor.get_pose()
        max_trials = int(np.ceil(np.abs(dis/delta_d)))
        delta = np.sign(dis) * delta_d
        for i in range(max_trials):
            success = self.move(self.atom.move_by_displacement(
                z=delta, xyz_coord='local'
            ), tag='try_forward', is_save=is_save, delay=False, cosntraint_pose=[1, 1, 1, 1, 1, 0])
            actor_pose = actor.get_pose()
            if np.linalg.norm(actor_pose.p - actor_last_pose.p) < np.abs(delta):
                return False
            actor_last_pose = actor_pose
        return True

    def try_forward(self, actor:Actor, dis=0.01, delta_d=0.004, is_save=True):
        if self.plan_success is False:
            return False

        actor_last_pose = actor.get_pose()
        max_trials = int(np.ceil(np.abs(dis/delta_d)))
        delta = np.sign(dis) * delta_d
        for i in range(max_trials):
            success = self.move(self.atom.move_by_displacement(
                z=delta, xyz_coord='local'
            ), tag='try_forward', is_save=is_save, delay=False)
            actor_pose = actor.get_pose()
            if np.linalg.norm(actor_pose.p - actor_last_pose.p) < np.abs(delta):
                return False
            actor_last_pose = actor_pose
        return True
