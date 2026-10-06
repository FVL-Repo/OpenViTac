import torch
import torch.nn.functional as torch_F
from envs.utils import data
import numpy as np
import os
from pathlib import Path
from tacex import GelSightSensor, GelSightSensorCfg
from tacex_assets import TACEX_ASSETS_DATA_DIR
from tacex.simulation_approaches.fem_based import ManiSkillSimulatorCfg
from tacex.simulation_approaches.fots import FOTSMarkerSimulatorCfg

from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import FrameTransformer, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg

from tacex_uipc import (
    UipcRLEnv,
    UipcIsaacAttachments,
    UipcIsaacAttachmentsCfg,
    UipcObject,
    UipcObjectCfg,
    UipcSimCfg
)

from ..utils.transforms import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .._base_task import BaseTask
    from tacex_uipc.sim import UipcIsaacAttachmentsCfg, UipcSim
    from tacex_uipc import UipcInteractiveScene


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


XSENSE_USE_REAL_BLUR_BG = _env_bool("XSENSE_USE_REAL_BLUR_BG", True)
XSENSE_BLUR_BG_PATH = os.environ.get("XSENSE_BLUR_BG_PATH", "").strip()
XSENSE_BACKGROUND_OVERRIDE_RAW = _env_bool("XSENSE_BACKGROUND_OVERRIDE_RAW", True)
XSENSE_OPTICAL_RESPONSE_GAIN = 1.0
XSENSE_MARKER_MOTION_GAIN = 1.0
XSENSE_MARKER_VISUAL_NOISE = 8.0
XSENSE_GEL_MASS_DENSITY = 1e3
XSENSE_MARKER_GRID_BOUNDS = (0.05, 0.12, 0.95, 0.88)
XSENSE_OUTPUT_ORIENTATION = os.environ.get("XSENSE_OUTPUT_ORIENTATION", "none").strip().lower()
TACTILE_ATTACHMENT_DEBUG = _env_bool("TACTILE_ATTACHMENT_DEBUG", False)


def _xsense_blur_bg_override_path(sensor_name: str) -> str:
    if not XSENSE_USE_REAL_BLUR_BG:
        return ""

    candidates: list[Path] = []
    if XSENSE_BLUR_BG_PATH:
        candidates.append(Path(XSENSE_BLUR_BG_PATH))

    side = None
    sensor_name_lower = sensor_name.lower()
    if "left" in sensor_name_lower:
        side = "left"
    elif "right" in sensor_name_lower:
        side = "right"
    if side is not None:
        calib_dir = Path(TACEX_ASSETS_DATA_DIR) / "Sensors" / "XenseWS" / "calibs" / "400x700"
        candidates.append(calib_dir / f"real_{side}_blur_bg.png")

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return ""


@configclass
class TactileCfg:
    name: str = 'tactile_sensor'
    sensor_type: str = 'gsmini'
    sensor_cfg = None
    gelpad_cfg: UipcObjectCfg = None
    gelpad_attachment_cfg: UipcIsaacAttachmentsCfg = None

def create_gelsight_mini_cfg(
    prim_path: str,
    gelpad_prim_path: str,
    gelpad_attachment_body_name: str,
    name: str = "tactile_sensor",
    resolution = (320, 240),
    update_period = 1/120,
    data_type:list[str] = ["camera_depth", "tactile_rgb"],
    sensor_type: str = "gsmini",
    dense: bool = False,
):
    from tacex_assets.sensors.gelsight_mini.gsmini_cfg import GelSightMiniCfg
    marker_sensor_type = 'gsmini_dyn' if dense else 'gsmini'
    sensor_cfg = GelSightMiniCfg(
        prim_path=prim_path,
        sensor_camera_cfg=GelSightMiniCfg.SensorCameraCfg(
            prim_path_appendix="/Camera",
            resolution=resolution,
            update_period=update_period,
            data_types=["depth", "rgb"],
            clipping_range=(0.024, 0.034),
        ),
        device="cuda",
        debug_vis=False,  # for rendering sensor output in the gui
        update_period=1/120,
        marker_motion_sim_cfg=ManiSkillSimulatorCfg(
            tactile_img_res=resolution,
            marker_shape=(9, 7),
            marker_interval=(2.40625, 2.45833),
            sub_marker_num=0,
            marker_radius=6,
            camera_to_surface=0.0283,
            real_size=(0.0266, 0.0209),
            # Neote reuses the GelSight Mini physical gelpad/camera geometry.
            # Keep the FEM surface tables on gsmini and use TactileCfg.sensor_type
            # below to select Neote-only visualization outputs.
            sensor_type=marker_sensor_type,
        ),
        data_types=data_type
    )
    sensor_cfg.marker_motion_sim_cfg.marker_params.num_markers = 64
    sensor_cfg.optical_sim_cfg = sensor_cfg.optical_sim_cfg.replace(
        with_shadow=False,
        tactile_img_res=resolution,
        device="cuda",
    )

    cfg = TactileCfg(
        name=name,
        sensor_type=sensor_type,
        sensor_cfg=sensor_cfg,
        gelpad_cfg=UipcObjectCfg(
            prim_path=gelpad_prim_path,
            constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(youngs_modulus=0.1),
            mass_density=1e4
        ),
        gelpad_attachment_cfg=UipcIsaacAttachmentsCfg(
            constraint_strength_ratio=1e4,
            body_name=gelpad_attachment_body_name,
            debug_vis=False,
        ),
    )
    return cfg


def create_xensews_cfg(
    prim_path: str,
    gelpad_prim_path: str,
    gelpad_attachment_body_name: str,
    gelpad_attachment_prim_path: str = None,
    name: str = "tactile_sensor",
    # resolution = (320, 240),
    resolution = (400, 700),
    update_period = 1/120,
    data_type:list[str] = ["camera_depth", "tactile_rgb"],
) -> TactileCfg:
    from tacex_assets.sensors.xensews.xensews_cfg import XenseWSCfg

    sensor_cfg = XenseWSCfg(
        prim_path=prim_path,
        sensor_camera_cfg=XenseWSCfg.SensorCameraCfg(
            prim_path_appendix="/Camera",
            update_period=update_period,
            resolution=resolution,
            data_types=["depth", "rgb"],
            clipping_range=(0.024, 0.030),  # GelSight-style tactile optical window; excludes static Xense case from no-contact depth.
        ),
        device="cuda",
        debug_vis=False,  # for rendering sensor output in the gui
        update_period=update_period,
        # marker_motion_sim_cfg=ManiSkillSimulatorCfg(
        #     tactile_img_res=resolution,
        #     sub_marker_num=0,
        #     sensor_type='xensews',
        # ),
        # data_types=data_type
        marker_motion_sim_cfg=ManiSkillSimulatorCfg(
            tactile_img_res=resolution,
            marker_shape=(11, 20),
            marker_interval=(
                16.42 * (XSENSE_MARKER_GRID_BOUNDS[2] - XSENSE_MARKER_GRID_BOUNDS[0]) / 10,
                27.84 * (XSENSE_MARKER_GRID_BOUNDS[3] - XSENSE_MARKER_GRID_BOUNDS[1]) / 19,
            ),
            sub_marker_num=0,
            marker_radius=3,
            sensor_type='xensews',
            marker_motion_gain=XSENSE_MARKER_MOTION_GAIN,
            marker_visual_noise=XSENSE_MARKER_VISUAL_NOISE,
            # Bind both XSense lattices to the camera optical x-axis. The
            # runtime left gel surface is offset by about 1.1 mm, while the
            # right surface is centered; surface-centering made the two FEM
            # sampling frames asymmetric.
            marker_binding_center_x_m=0.0,
            # XSense images are 400x700 portrait: width is the 16.42mm short axis,
            # height is the 27.84mm long axis.
            camera_to_surface=0.0280,
            real_size=(0.01642, 0.02784),
        ),
        data_types=data_type,
    )
    # sensor_cfg.marker_motion_sim_cfg.marker_params.num_markers = 1200
    sensor_cfg.marker_motion_sim_cfg.marker_params.num_markers = 11 * 20
    sensor_cfg.optical_sim_cfg = sensor_cfg.optical_sim_cfg.replace(
        calib_folder_path=f"{TACEX_ASSETS_DATA_DIR}/Sensors/XenseWS/calibs/640x480",
        with_shadow=False,
        tactile_img_res=resolution,
        device="cuda",
        background_img_override_path=_xsense_blur_bg_override_path(name),
        background_img_override_raw=XSENSE_BACKGROUND_OVERRIDE_RAW,
        response_gain=XSENSE_OPTICAL_RESPONSE_GAIN,
        subtract_zero_indentation_baseline=True,
        use_physical_indentation_map=True,
        gel_surface_depth=0.0280,
        # The calibrated Xense gel occupies 24~28 mm from the optical center.
        gelpad_to_camera_min_distance=0.024,
    )

    cfg = TactileCfg(
        name=name,
        sensor_type="xensews",
        sensor_cfg=sensor_cfg,
        gelpad_cfg=UipcObjectCfg(
            prim_path=gelpad_prim_path,
            # Xense uses raw depth, so keep the gelpad close to rigid to avoid
            # whole-pad bending/tilting dominating the tactile camera image.
            constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(youngs_modulus=15.0),
            mass_density=XSENSE_GEL_MASS_DENSITY
        ),
        gelpad_attachment_cfg=UipcIsaacAttachmentsCfg(
            constraint_strength_ratio=2e5,
            body_name=gelpad_attachment_body_name,
            isaac_rigid_prim_path=gelpad_attachment_prim_path,
            attachment_points_radius=0.0008,
            debug_vis=False,
        ),
    )
    return cfg

def create_tactile_cfg(
    prim_path: str,
    gelpad_prim_path: str,
    gelpad_attachment_body_name: str,
    gelpad_attachment_prim_path: str = None,
    name: str = "tactile_sensor",
    sensor_type:Literal['gsmini', 'xensews', 'neote'] = "gsmini",
    data_type:list[str] = ["camera_depth", "tactile_rgb"],
    dense: bool = False,
) -> TactileCfg:
    if sensor_type in ("gsmini", "neote"):
        return create_gelsight_mini_cfg(
            prim_path=prim_path,
            gelpad_prim_path=gelpad_prim_path,
            gelpad_attachment_body_name=gelpad_attachment_body_name,
            name=name,
            data_type=data_type,
            sensor_type=sensor_type,
            dense=dense,
        )
    elif sensor_type == "xensews":
        return create_xensews_cfg(
            prim_path=prim_path,
            gelpad_prim_path=gelpad_prim_path,
            gelpad_attachment_body_name=gelpad_attachment_body_name,
            gelpad_attachment_prim_path=gelpad_attachment_prim_path,
            name=name,
            data_type=data_type,
        )
    else:
        raise ValueError(f"Unknown sensor type: {sensor_type}")


class VisualTactileSensor:
    def __init__(self, name:str, cfg:TactileCfg, robot, scene: 'UipcInteractiveScene', uipc_sim:'UipcSim'):
        self.cfg = cfg
        self.name = name
        self.scene = scene
        self.robot = robot
        self.uipc_sim = uipc_sim

        self.gelpad = UipcObject(self.cfg.gelpad_cfg, self.uipc_sim)
        self.attachment = UipcIsaacAttachments(
            self.cfg.gelpad_attachment_cfg, self.gelpad, self.robot
        )
        # self.sensor = GelSightSensor(self.cfg.sensor_cfg, self.gelpad)
        sensor_cls = getattr(self.cfg.sensor_cfg, "class_type", GelSightSensor)
        self.sensor = sensor_cls(self.cfg.sensor_cfg, self.gelpad)
        self.force_field_grid = (64, 48)

    def _debug_log(self, message: str):
        if TACTILE_ATTACHMENT_DEBUG:
            print(f"[tactile-debug] {self.name} {message}")

    def setup(self):
        self.device = self.uipc_sim.cfg.device
        if not self._is_xsense_sensor():
            init_pts = self.gelpad._data.nodal_pos_w[
                self.attachment.attachment_points_idx
            ].cpu().numpy()
            init_world_trans = self.gelpad.init_world_transform.cpu().numpy()
            self.origin_pts = (
                init_pts - init_world_trans[:3, 3]
            ) @ (init_world_trans[:3, :3].T).T
            attach_pts = self.attachment.attachment_offsets
            init_trans = estimate_rigid_transform(self.origin_pts, attach_pts)
            self.attach_to_init = torch.tensor(
                np.linalg.inv(init_trans), dtype=torch.float64, device=self.device
            )
            self.sensor.marker_motion_simulator.marker_motion_sim.init_vertices()
            return

        attach_count = len(self.attachment.attachment_points_idx)
        self._debug_log(f"attachment_points={attach_count}")
        self._attachment_reference_offsets = np.array(
            self.attachment.attachment_offsets, dtype=np.float32, copy=True
        )
        target_points = self._get_gelpad_points_for_current_attachment()
        if target_points is not None:
            self._write_gelpad_vertices(target_points)
        else:
            self._reset_gelpad_vertices_to_initial()
        self._refresh_attachment_reference()

        self.sensor.marker_motion_simulator.marker_motion_sim.init_vertices()

    def _get_initial_gelpad_reference_points(self):
        init_vertex_pos = getattr(self.gelpad, "init_vertex_pos", None)
        if init_vertex_pos is None:
            return None
        return torch.as_tensor(init_vertex_pos, dtype=torch.float32, device=self.device)

    def _restore_attachment_reference_offsets(self):
        reference_offsets = getattr(self, "_attachment_reference_offsets", None)
        if reference_offsets is None:
            reference_offsets = np.array(
                self.attachment.attachment_offsets, dtype=np.float32, copy=True
            )
            self._attachment_reference_offsets = reference_offsets
        self.attachment.attachment_offsets = np.array(
            reference_offsets, dtype=np.float32, copy=True
        )
        return self.attachment.attachment_offsets

    def _get_attachment_aim_positions_from_reference_offsets(self):
        attach_idx = self.attachment.attachment_points_idx
        if len(attach_idx) == 0:
            return None
        offsets_np = self._restore_attachment_reference_offsets()
        try:
            attach_pose = self.get_attach_pose()
        except Exception as exc:
            self._debug_log(f"attachment aim unavailable: {exc!r}")
            return None

        offsets = torch.as_tensor(offsets_np, dtype=torch.float32, device=self.device)
        body_pos_w = torch.tensor(
            attach_pose.p, dtype=torch.float32, device=self.device
        ).reshape(1, 3).repeat(offsets.shape[0], 1)
        body_quat_w = torch.tensor(
            attach_pose.q, dtype=torch.float32, device=self.device
        ).reshape(1, 4).repeat(offsets.shape[0], 1)
        return math_utils.quat_apply(body_quat_w, offsets) + body_pos_w

    def _get_gelpad_points_for_current_attachment(self):
        init_points = self._get_initial_gelpad_reference_points()
        aim_points = self._get_attachment_aim_positions_from_reference_offsets()
        attach_idx = self.attachment.attachment_points_idx
        if init_points is None or aim_points is None or len(attach_idx) == 0:
            return init_points

        init_attach_points = init_points[attach_idx].detach().cpu().numpy()
        aim_points_np = aim_points.detach().cpu().numpy()
        if init_attach_points.shape != aim_points_np.shape:
            self._debug_log(
                "attachment reference mismatch: "
                f"{init_attach_points.shape} vs {aim_points_np.shape}"
            )
            return init_points

        init_to_current = estimate_rigid_transform(init_attach_points, aim_points_np)
        init_points_np = init_points.detach().cpu().numpy()
        target_points_np = init_points_np @ init_to_current[:3, :3].T + init_to_current[:3, 3]
        target_points = torch.tensor(target_points_np, dtype=torch.float32, device=self.device)

        fit_error = torch.linalg.norm(target_points[attach_idx] - aim_points, dim=1).mean()
        target_centroid = target_points.mean(dim=0).detach().cpu().numpy()
        aim_centroid = aim_points.mean(dim=0).detach().cpu().numpy()
        self._debug_log(
            "placed gelpad from USD attachment offsets, "
            f"fit_error={fit_error.item():.6e}m, "
            f"gel_centroid={target_centroid.tolist()}, "
            f"aim_centroid={aim_centroid.tolist()}"
        )
        return target_points

    def _invalidate_gelpad_nodal_cache(self):
        data_obj = getattr(self.gelpad, "_data", None)
        nodal_buffer = getattr(data_obj, "_nodal_pos_w", None)
        if nodal_buffer is not None:
            nodal_buffer.timestamp = -float("inf")

    def _write_gelpad_vertices(self, vertex_positions):
        self.gelpad.write_vertex_positions_to_sim(vertex_positions=vertex_positions)
        self._invalidate_gelpad_nodal_cache()
        return True

    def _reset_gelpad_vertices_to_initial(self):
        init_vertex_pos = getattr(self.gelpad, "init_vertex_pos", None)
        if init_vertex_pos is None:
            return False
        return self._write_gelpad_vertices(init_vertex_pos)

    def _refresh_attachment_reference(self):
        attach_idx = self.attachment.attachment_points_idx
        if len(attach_idx) == 0:
            self.origin_pts = np.zeros((0, 3), dtype=np.float64)
            self.attach_to_init = torch.eye(4, dtype=torch.float64, device=self.device)
            return
        points_w = self._get_initial_gelpad_reference_points()
        if points_w is None:
            points_w = self.gelpad._data.nodal_pos_w
        init_pts = points_w[self.attachment.attachment_points_idx].detach().cpu().numpy()
        init_world_trans = self.gelpad.init_world_transform.cpu().numpy()
        self.origin_pts = (init_pts - init_world_trans[:3, 3]) @ (init_world_trans[:3, :3].T).T
        attach_pts = self._restore_attachment_reference_offsets()
        init_trans = estimate_rigid_transform(self.origin_pts, attach_pts)
        self.attach_to_init = np.linalg.inv(init_trans)
        self.attach_to_init = torch.tensor(self.attach_to_init, dtype=torch.float64, device=self.device)

    def _realign_attachment_offsets_to_current_body(self, reference_points_w=None):
        attach_idx = self.attachment.attachment_points_idx
        if len(attach_idx) == 0:
            return
        try:
            attach_pose = self.get_attach_pose()
        except Exception as exc:
            self._debug_log(f"attachment offset realign skipped: {exc!r}")
            return

        if reference_points_w is None:
            points_src = self.gelpad._data.nodal_pos_w
        else:
            points_src = torch.as_tensor(reference_points_w, dtype=torch.float32, device=self.device)
        points_w = points_src[attach_idx].to(dtype=torch.float32, device=self.device)
        body_pos_w = torch.tensor(
            attach_pose.p, dtype=torch.float32, device=self.device
        ).reshape(1, 3)
        body_quat_w = torch.tensor(
            attach_pose.q, dtype=torch.float32, device=self.device
        ).reshape(1, 4)
        body_quat_w = body_quat_w.repeat(points_w.shape[0], 1)

        offsets = math_utils.quat_apply_inverse(body_quat_w, points_w - body_pos_w)
        self.attachment.attachment_offsets = offsets.detach().cpu().numpy()
        mean_error = torch.linalg.norm(
            math_utils.quat_apply(body_quat_w, offsets) + body_pos_w - points_w,
            dim=1,
        ).mean()
        self._debug_log(
            "realigned attachment_offsets "
            f"to current body pose, mean_reproj_error={mean_error.item():.6e}m"
        )

    def get_attach_pose(self):
        if type(self.attachment.isaaclab_rigid_object) is Articulation:
            # this only works when rigid body is an articulation
            # self.attachment.isaaclab_rigid_object._physics_sim_view.update_articulations_kinematic()
            # read data from simulation
            poses = self.attachment.isaaclab_rigid_object._root_physx_view.get_link_transforms().clone()
            poses[..., 3:7] = math_utils.convert_quat(poses[..., 3:7], to="wxyz")
            pose = poses[:, self.attachment.rigid_body_id, 0:7].clone()
        elif type(self.attachment.isaaclab_rigid_object) is RigidObject:
            # only works with rigid body
            pose = self.attachment.isaaclab_rigid_object._root_physx_view.root_state_w.view(-1, 1, 13)
            pose = pose[:, self.attachment.rigid_body_id, 0:7].clone()
        else:
            raise RuntimeError("Need an Articulation or a RigidBody object for the Isaac X UIPC attachment.")
        return Pose.from_list(pose.flatten().tolist())

    def get_init_pts(self):
        curr_attach_pose = self.get_attach_pose()
        trans_to_attach = np.linalg.inv(curr_attach_pose.to_transformation_matrix())
        trans_to_attach = torch.tensor(trans_to_attach, dtype=torch.float64, device=self.device)
        trans_to_init = self.attach_to_init @ trans_to_attach
        return self.gelpad.data.nodal_pos_w @ trans_to_init[:3, :3].T + trans_to_init[:3, 3]
 
    def update(self, dt, force_recompute=False):
        self.gelpad.update(dt=dt)
        self.sensor.update(dt=dt, force_recompute=force_recompute)
    
    def set_debug_vis(self):
        if not self.sensor.cfg.debug_vis:
            return 
        for data_type in ['marker_motion']:
            self.sensor._prim_view.prims[0].GetAttribute(f"debug_{data_type}").Set(True)

    def _is_xsense_sensor(self) -> bool:
        marker_cfg = getattr(getattr(self.sensor, "cfg", None), "marker_motion_sim_cfg", None)
        sensor_type = str(getattr(marker_cfg, "sensor_type", "")).lower()
        return sensor_type.startswith("xense")

    def _is_neote_sensor(self) -> bool:
        return str(getattr(self.cfg, "sensor_type", "")).lower() == "neote"

    def _resize_spatial(self, value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        dtype = value.dtype
        is_uint8 = dtype == torch.uint8
        work = value.to(dtype=torch.float32)

        if work.dim() == 2:
            work = work.unsqueeze(0).unsqueeze(0)
            out = torch_F.interpolate(work, size=size, mode="bilinear", align_corners=False)[0, 0]
        elif work.dim() == 3 and work.shape[-1] in (1, 3, 4):
            work = work.permute(2, 0, 1).unsqueeze(0)
            out = torch_F.interpolate(work, size=size, mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
        elif work.dim() == 3 and work.shape[0] in (1, 3, 4):
            work = work.unsqueeze(0)
            out = torch_F.interpolate(work, size=size, mode="bilinear", align_corners=False)[0]
        else:
            return value

        if is_uint8:
            return out.round().clamp(0, 255).to(dtype=dtype)
        return out.to(dtype=dtype)

    def _orient_xsense_image(self, value: torch.Tensor) -> torch.Tensor:
        if not self._is_xsense_sensor() or XSENSE_OUTPUT_ORIENTATION in {"", "none", "identity"}:
            return value

        if value.dim() == 2:
            target_size = tuple(value.shape[:2])
            if XSENSE_OUTPUT_ORIENTATION == "transpose":
                return self._resize_spatial(value.transpose(0, 1), target_size)
        elif value.dim() == 3 and value.shape[-1] in (1, 3, 4):
            target_size = tuple(value.shape[:2])
            if XSENSE_OUTPUT_ORIENTATION == "transpose":
                return self._resize_spatial(value.transpose(0, 1), target_size)
        elif value.dim() == 3 and value.shape[0] in (1, 3, 4):
            target_size = tuple(value.shape[1:3])
            if XSENSE_OUTPUT_ORIENTATION == "transpose":
                return self._resize_spatial(value.transpose(1, 2), target_size)
        return value

    def _orient_xsense_marker(self, value: torch.Tensor) -> torch.Tensor:
        if not self._is_xsense_sensor() or XSENSE_OUTPUT_ORIENTATION != "transpose":
            return value

        marker_cfg = getattr(getattr(self.sensor, "cfg", None), "marker_motion_sim_cfg", None)
        tactile_img_res = getattr(marker_cfg, "tactile_img_res", None)
        if tactile_img_res is None or value.shape[-1] != 2:
            return value

        width, height = float(tactile_img_res[0]), float(tactile_img_res[1])
        if width <= 1.0 or height <= 1.0:
            return value

        oriented = value.clone()
        x = value[..., 0]
        y = value[..., 1]
        oriented[..., 0] = y * ((width - 1.0) / (height - 1.0))
        oriented[..., 1] = x * ((height - 1.0) / (width - 1.0))
        return oriented
    
    def get_observations(self, data_types: list[str] = None):
        obs = {}
        if data_types is None:
            data_types = ['rgb', 'rgb_marker', 'depth', 'points', 'pose', 'flow']
        for data_type in data_types:
            if data_type == 'rgb':
                obs['rgb'] = self._orient_xsense_image(self.sensor.data.output['tactile_rgb'].squeeze(0))
            elif data_type == 'rgb_marker':
                obs['rgb_marker'] = self._orient_xsense_image(self.sensor.data.output['marker_rgb'].squeeze(0))
            elif data_type == 'depth':
                obs['depth'] = self._orient_xsense_image(self.sensor.data.output['height_map'].squeeze(0))
            elif data_type == 'indentation':
                optical_simulator = getattr(self.sensor, 'optical_simulator', None)
                indentation_map = getattr(optical_simulator, 'indentation_map', None)
                if indentation_map is None:
                    raise RuntimeError('Optical simulator does not expose an indentation map')
                obs['indentation'] = self._orient_xsense_image(indentation_map.squeeze(0))
            elif data_type == 'marker':
                obs['marker'] = self._orient_xsense_marker(self.sensor.data.output['marker_motion'].squeeze(0))
            elif data_type == 'points':
                obs['points'] = self.get_init_pts()
            elif data_type == 'pose':
                obs['pose'] = self.get_attach_pose().totensor()
            elif data_type == 'contact_force':
                obs['contact_force'] = self._get_contact_force()
            elif data_type == 'vertex_force':
                obs['vertex_force'] = self.get_vertex_force()
            elif data_type == 'marker_force':
                obs['marker_force'] = self.get_marker_force(mode='interp')
            elif data_type == 'marker_force_scatter':
                obs['marker_force_scatter'] = self.get_marker_force(mode='scatter')
            elif data_type == 'marker_force_img':
                obs['marker_force_img'] = self._orient_xsense_image(self.get_marker_force_image(mode='interp'))
            elif data_type == 'force_field':
                obs['force_field'] = self.get_force_field(grid=self.force_field_grid)
            elif data_type == 'force_field_img':
                obs['force_field_img'] = self._orient_xsense_image(self.get_force_field_image(grid=self.force_field_grid))
            elif data_type == 'gel_particle':
                obs['gel_particle'] = self._orient_xsense_image(self.get_gel_particle_image())
        return obs

    def _get_contact_force(self):
        idx, grad = self.uipc_sim.get_contact_gradient()
        offsets = self.uipc_sim._system_vertex_offsets["uipc::backend::cuda::GlobalVertexManager"]
        start = int(offsets[self.gelpad.global_system_id])
        num_v = self.gelpad.data.nodal_pos_w.shape[0]
        dense = torch.zeros((num_v, 3), dtype=torch.float32, device=self.device)
        if idx.shape[0] > 0:
            mask = (idx >= start) & (idx < start + num_v)
            if mask.any():
                loc = torch.as_tensor(idx[mask] - start, device=self.device, dtype=torch.long)
                dense[loc] = torch.as_tensor(-grad[mask], dtype=torch.float32, device=self.device)
        return dense

    def _world_to_sensor_rot(self):
        cam = self.sensor.camera
        cam._update_poses(cam._ALL_INDICES)
        return math_utils.matrix_from_quat(cam._data.quat_w_ros)[0]

    def get_vertex_force(self, in_sensor_frame: bool = True):
        force = self._get_contact_force()
        if in_sensor_frame:
            force = force @ self._world_to_sensor_rot().to(force.dtype)
        return force

    def _precompute_marker_force_maps(self):
        mm = self.sensor.marker_motion_simulator.marker_motion_sim
        if not hasattr(mm, "marker_surf_idx"):
            raise RuntimeError(
                "marker_force requires the FEM marker simulator after TactileManager.setup()."
            )
        self._mf_surf_idx = torch.as_tensor(mm.marker_surf_idx, device=self.device, dtype=torch.long)
        self._mf_weight = torch.as_tensor(mm.marker_weight, device=self.device, dtype=torch.float32)
        self._mf_surf_global = torch.as_tensor(mm.vertices_on_surface, device=self.device, dtype=torch.long)
        self._mf_num_markers = int(self._mf_surf_idx.shape[0])

        surf_xy = mm.init_surface_vertices_camera[:, :2].to(self.device, dtype=torch.float32)
        marker_xy = (surf_xy[self._mf_surf_idx] * self._mf_weight[..., None]).sum(1)
        self._mf_nearest = torch.argmin(torch.cdist(surf_xy, marker_xy), dim=1)

    def get_marker_force(self, mode: str = "interp", in_sensor_frame: bool = True, reshape: bool = False):
        if not hasattr(self, "_mf_surf_idx"):
            self._precompute_marker_force_maps()

        force_w = self._get_contact_force()
        force_surf = force_w[self._mf_surf_global]
        if mode == "interp":
            marker_force = (force_surf[self._mf_surf_idx] * self._mf_weight[..., None]).sum(1)
        elif mode == "scatter":
            marker_force = torch.zeros((self._mf_num_markers, 3), dtype=force_surf.dtype, device=self.device)
            marker_force.index_add_(0, self._mf_nearest, force_surf)
        else:
            raise ValueError(f"Unknown marker force mode: {mode!r}")

        if in_sensor_frame:
            marker_force = marker_force @ self._world_to_sensor_rot().to(marker_force.dtype)

        if reshape:
            sx, sy = self.sensor.marker_motion_simulator.marker_motion_sim.marker_shape
            if marker_force.shape[0] == sx * sy:
                marker_force = marker_force.reshape(sy, sx, 3)
        return marker_force

    def get_marker_force_image(
        self,
        mode: str = "interp",
        style: str = "tacff",
        base: str = None,
        shear_scale: float = None,
        normal_scale: float = None,
    ):
        import cv2

        if not hasattr(self, "_mf_surf_idx"):
            self._precompute_marker_force_maps()
        mm = self.sensor.marker_motion_simulator.marker_motion_sim

        if base is None:
            base = "black" if style == "tacff" else "rgb"
        rgb = self.sensor.data.output["tactile_rgb"].squeeze(0).detach().cpu().numpy()
        H, W = rgb.shape[:2]
        if base == "black":
            img = np.zeros((H, W, 3), dtype=np.uint8)
        elif base == "white":
            img = np.full((H, W, 3), 255, dtype=np.uint8)
        elif base == "rgb":
            img = rgb.astype(np.uint8)
        elif base == "rgb_marker":
            img = self.sensor.data.output["marker_rgb"].squeeze(0).detach().cpu().numpy().astype(np.uint8)
        else:
            raise ValueError(f"Unknown marker_force image base: {base!r}")
        img = np.ascontiguousarray(img)

        force = self.get_marker_force(mode=mode, in_sensor_frame=True).detach().cpu().numpy()
        shear = force[:, :2]
        smag = np.linalg.norm(shear, axis=1)
        fz = force[:, 2]

        ref = getattr(mm, "reference_surface_vertices_camera", mm.init_surface_vertices_camera)
        ref_pts = (
            ref[self._mf_surf_idx].detach().cpu().numpy()
            * self._mf_weight.detach().cpu().numpy()[..., None]
        ).sum(1).astype(np.float32)
        uv = np.asarray(mm.gen_marker_uv(ref_pts), dtype=np.float32)

        s_n = np.abs(fz).max() if normal_scale is None else normal_scale
        s_n = s_n if s_n > 1e-9 else 1.0
        t = np.clip(np.abs(fz) / s_n, 0.0, 1.0)
        if style == "tacff":
            colors = np.stack([t * 255, (1.0 - t) * 255, np.zeros_like(t)], axis=1)
        else:
            cidx = (np.clip(fz / s_n, -1.0, 1.0) * 0.5 + 0.5) * 255.0
            colors = cv2.applyColorMap(cidx.reshape(-1, 1).astype(np.uint8), cv2.COLORMAP_JET)
            colors = colors.reshape(-1, 3)[:, ::-1]
        colors = colors.astype(np.uint8)

        if shear_scale is None:
            shear_scale = (20.0 / smag.max()) if smag.max() > 1e-9 else 0.0
        for i in range(uv.shape[0]):
            u, v = int(round(uv[i, 0])), int(round(uv[i, 1]))
            if not (0 <= u < W and 0 <= v < H):
                continue
            color = tuple(int(c) for c in colors[i])
            du = int(round(shear[i, 0] * shear_scale))
            dv = int(round(shear[i, 1] * shear_scale))
            cv2.circle(img, (u, v), 1 if style == "tacff" else 3, color, -1, cv2.LINE_AA)
            if smag[i] > 1e-9:
                cv2.arrowedLine(
                    img,
                    (u, v),
                    (u + du, v + dv),
                    color if style == "tacff" else (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                    0,
                    0.35 if style == "tacff" else 0.3,
                )
        return torch.as_tensor(img, dtype=torch.uint8, device=self.device)

    def _precompute_force_field_map(self, grid):
        from scipy.spatial import Delaunay

        if not hasattr(self, "_mf_surf_global"):
            self._precompute_marker_force_maps()
        mm = self.sensor.marker_motion_simulator.marker_motion_sim
        surf_xy = mm.init_surface_vertices_camera[:, :2].detach().cpu().numpy().astype(np.float64)
        W, H = int(grid[0]), int(grid[1])
        (xmin, ymin), (xmax, ymax) = surf_xy.min(0), surf_xy.max(0)
        gx, gy = np.meshgrid(np.linspace(xmin, xmax, W), np.linspace(ymin, ymax, H))
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)

        tri = Delaunay(surf_xy)
        simplex = tri.find_simplex(pts)
        valid = simplex >= 0
        safe_simplex = simplex.copy()
        safe_simplex[~valid] = 0
        transform = tri.transform[safe_simplex]
        bc = np.einsum("nij,nj->ni", transform[:, :2, :], pts - transform[:, 2, :])
        bary = np.concatenate([bc, 1.0 - bc.sum(1, keepdims=True)], axis=1)
        verts = tri.simplices[safe_simplex]
        bary[~valid] = 0.0
        verts[~valid] = 0

        self._ff_grid = (W, H)
        self._ff_verts = torch.as_tensor(verts, device=self.device, dtype=torch.long)
        self._ff_bary = torch.as_tensor(bary, device=self.device, dtype=torch.float32)
        self._ff_valid = torch.as_tensor(valid, device=self.device, dtype=torch.float32)[:, None]

    def get_force_field(self, grid=(64, 48), in_sensor_frame: bool = True):
        if getattr(self, "_ff_grid", None) != (int(grid[0]), int(grid[1])):
            self._precompute_force_field_map(grid)
        force_surf = self._get_contact_force()[self._mf_surf_global]
        field = (force_surf[self._ff_verts] * self._ff_bary[..., None]).sum(1) * self._ff_valid
        if in_sensor_frame:
            field = field @ self._world_to_sensor_rot().to(field.dtype)
        W, H = self._ff_grid
        return field.reshape(H, W, 3)

    def get_force_field_image(self, grid=(64, 48), upscale=8, arrow_every=6,
                              normal_scale=None, shear_scale=None):
        import cv2

        field = self.get_force_field(grid=grid, in_sensor_frame=True).detach().cpu().numpy()
        H, W = field.shape[:2]
        fz = field[..., 2]
        shear = field[..., :2]
        s_n = np.abs(fz).max() if normal_scale is None else normal_scale
        s_n = s_n if s_n > 1e-9 else 1.0
        t = np.clip(np.abs(fz) / s_n, 0.0, 1.0)
        img = np.zeros((H, W, 3), dtype=np.float32)
        img[..., 0] = t * t * 255.0
        img[..., 1] = (1.0 - t) * t * 255.0
        img = np.ascontiguousarray(np.clip(img, 0, 255).astype(np.uint8))
        img = cv2.resize(img, (W * upscale, H * upscale), interpolation=cv2.INTER_LINEAR)

        smag = np.linalg.norm(shear, axis=2)
        if shear_scale is None:
            max_shear = smag.max()
            shear_scale = (arrow_every * upscale * 0.9 / max_shear) if max_shear > 1e-9 else 0.0
        for r in range(0, H, arrow_every):
            for c in range(0, W, arrow_every):
                if smag[r, c] <= 1e-9:
                    continue
                u, v = int((c + 0.5) * upscale), int((r + 0.5) * upscale)
                du = int(shear[r, c, 0] * shear_scale)
                dv = int(shear[r, c, 1] * shear_scale)
                cv2.arrowedLine(img, (u, v), (u + du, v + dv), (255, 255, 255),
                                1, cv2.LINE_AA, 0, 0.3)
        return torch.as_tensor(img, dtype=torch.uint8, device=self.device)

    def dump_force_field_meta(self, path, grid=None):
        if grid is None:
            grid = self.force_field_grid
        if getattr(self, "_ff_grid", None) != (int(grid[0]), int(grid[1])):
            self._precompute_force_field_map(grid)
        mm = self.sensor.marker_motion_simulator.marker_motion_sim
        np.savez(
            str(path),
            ff_verts=self._ff_verts.detach().cpu().numpy(),
            ff_bary=self._ff_bary.detach().cpu().numpy(),
            ff_valid=self._ff_valid.detach().cpu().numpy(),
            surf_global=self._mf_surf_global.detach().cpu().numpy(),
            grid=np.array(self._ff_grid, dtype=np.int64),
            surf_ref_xy=mm.init_surface_vertices_camera[:, :2].detach().cpu().numpy(),
        )

    def _init_dense_particles(self, grid=(52, 40), seed=0):
        import cv2

        mm = self.sensor.marker_motion_simulator.marker_motion_sim
        surf = mm.init_surface_vertices_camera[:, :2].detach().cpu().numpy()
        (xmin, ymin), (xmax, ymax) = surf.min(0), surf.max(0)
        px, py = 0.03 * (xmax - xmin), 0.03 * (ymax - ymin)
        gx, gy = np.meshgrid(
            np.linspace(xmin + px, xmax - px, grid[0]),
            np.linspace(ymin + py, ymax - py, grid[1]),
        )
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        self._pp_idx, weights = mm._gen_marker_weight(pts)
        self._pp_w = np.asarray(weights, dtype=np.float32)
        self._pp_grid = tuple(grid)

        init3d = mm.init_surface_vertices_camera.detach().cpu().numpy()
        rest_pts = (init3d[self._pp_idx] * self._pp_w[..., None]).sum(1).astype(np.float32)
        self._pp_uv0 = np.asarray(mm.gen_marker_uv(rest_pts), dtype=np.float32)

        rgb = self.sensor.data.output["tactile_rgb"].squeeze(0).detach().cpu().numpy()
        H, W = rgb.shape[:2]
        rng = np.random.RandomState(seed)
        n = int(0.09 * H * W)
        xs = rng.uniform(0, W, n)
        ys = rng.uniform(0, H, n)
        if self._is_neote_sensor():
            # Neote reference footage has a warm translucent gel with
            # low-saturation red/orange and cyan-green pigment grains.
            warm = rng.rand(n) < 0.65
            hue = np.empty(n, dtype=np.float32)
            hue[warm] = rng.normal(8.0, 8.0, int(warm.sum()))
            hue[~warm] = rng.normal(74.0, 14.0, int((~warm).sum()))
            hue = np.mod(hue, 180).astype(np.uint8)
            sat = np.clip(rng.normal(105.0, 32.0, n), 45, 165).astype(np.uint8)
            val = np.clip(rng.normal(205.0, 28.0, n), 145, 245).astype(np.uint8)
        else:
            hue = rng.uniform(0, 180, n).astype(np.uint8)
            sat = np.clip(rng.normal(120.0, 35.0, n), 55, 190).astype(np.uint8)
            val = np.clip(rng.normal(205.0, 28.0, n), 145, 245).astype(np.uint8)
        cols = cv2.cvtColor(np.stack([hue, sat, val], 1)[None], cv2.COLOR_HSV2RGB)[0].astype(np.float32)

        coat = np.zeros((H, W, 3), dtype=np.float32)
        alpha = np.zeros((H, W), dtype=np.float32)
        for x, y, color, a in zip(xs, ys, cols, rng.uniform(0.35, 0.85, n)):
            cv2.circle(coat, (int(x), int(y)), 1, color.tolist(), -1, cv2.LINE_AA)
            cv2.circle(alpha, (int(x), int(y)), 1, float(a), -1, cv2.LINE_AA)
        self._pp_coat = cv2.GaussianBlur(coat, (0, 0), 0.65)
        self._pp_calpha = cv2.GaussianBlur(alpha, (0, 0), 0.65)

        grain = np.random.RandomState(seed + 1).randn(H, W).astype(np.float32)
        grain = cv2.GaussianBlur(grain, (0, 0), 0.7)
        self._pp_grain = grain / (grain.std() + 1e-6)
        self._pp_mesh = np.stack(np.meshgrid(np.arange(W), np.arange(H)), 0).astype(np.float32)

    def get_gel_particle_image(self, grid=(52, 40)):
        import cv2

        mm = self.sensor.marker_motion_simulator.marker_motion_sim
        if getattr(self, "_pp_grid", None) != tuple(grid):
            self._init_dense_particles(grid)

        curr = mm.get_surface_vertices_camera().detach().cpu().numpy()
        pts = (curr[self._pp_idx] * self._pp_w[..., None]).sum(1).astype(np.float32)
        mean_motion = np.mean(
            mm.get_vertices_camera()[mm.constrain_ids].detach().cpu().numpy() - mm.constrain_pts,
            axis=0,
        )
        pts[:, :2] -= mean_motion[:2]
        uv = np.asarray(mm.gen_marker_uv(pts), dtype=np.float32)
        disp = uv - self._pp_uv0

        rgb = self.sensor.data.output["tactile_rgb"].squeeze(0).detach().cpu().numpy().astype(np.float32)
        H, W = rgb.shape[:2]
        scale = 8
        dh, dw = max(H // scale, 1), max(W // scale, 1)
        accx = np.zeros((dh, dw), dtype=np.float32)
        accy = np.zeros((dh, dw), dtype=np.float32)
        cnt = np.zeros((dh, dw), dtype=np.float32)
        xi = np.clip((self._pp_uv0[:, 0] / scale).astype(int), 0, dw - 1)
        yi = np.clip((self._pp_uv0[:, 1] / scale).astype(int), 0, dh - 1)
        np.add.at(accx, (yi, xi), disp[:, 0])
        np.add.at(accy, (yi, xi), disp[:, 1])
        np.add.at(cnt, (yi, xi), 1.0)

        weight = cv2.GaussianBlur(cnt, (0, 0), 3.0) + 1e-3
        dx = cv2.resize(cv2.GaussianBlur(accx, (0, 0), 3.0) / weight, (W, H), interpolation=cv2.INTER_LINEAR)
        dy = cv2.resize(cv2.GaussianBlur(accy, (0, 0), 3.0) / weight, (W, H), interpolation=cv2.INTER_LINEAR)

        lum = cv2.GaussianBlur(rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32), (0, 0), 1.5)
        g0 = np.clip(158.0 + (lum - lum.mean()) * 0.65, 112, 205)
        gel = np.stack([g0 * 1.06, g0 * 0.99, g0 * 0.94], -1)
        gel += self._pp_grain[..., None] * np.array([4.0, 3.5, 3.0], dtype=np.float32)
        gel += cv2.GaussianBlur(self._pp_grain, (0, 0), 3.0)[..., None] * np.array([3.0, 5.0, 4.0], dtype=np.float32)

        mapx = (self._pp_mesh[0] - dx).astype(np.float32)
        mapy = (self._pp_mesh[1] - dy).astype(np.float32)
        speckle = cv2.remap(self._pp_coat, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        alpha = cv2.remap(self._pp_calpha, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        a = np.clip(alpha * 0.34, 0.0, 1.0)[..., None]
        out = gel * (1.0 - a) + speckle * a
        shade = np.clip(cv2.GaussianBlur(np.hypot(dx, dy), (0, 0), 8.0) / 6.0, 0.0, 1.0)[..., None]
        out = np.clip(out * (1.0 - 0.08 * shade), 0, 255).astype(np.uint8)

        dump_dir = os.environ.get("GELPART_DUMP")
        if dump_dir:
            os.makedirs(os.path.join(dump_dir, "gp"), exist_ok=True)
            os.makedirs(os.path.join(dump_dir, "rgb"), exist_ok=True)
            n = getattr(self, "_pp_dump_n", 0)
            cv2.imwrite(os.path.join(dump_dir, "gp", f"{self.name}_{n:04d}.png"),
                        cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(dump_dir, "rgb", f"{self.name}_{n:04d}.png"),
                        cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            self._pp_dump_n = n + 1
        return torch.as_tensor(out, dtype=torch.uint8, device=self.device)
    
    def _reset_idx(self):
        if not self._is_xsense_sensor():
            self.init_pose_mat = self.get_attach_pose().to_transformation_matrix()
            return

        reference_points = self._get_gelpad_points_for_current_attachment()
        if reference_points is not None:
            restored = self._write_gelpad_vertices(reference_points)
        else:
            restored = self._reset_gelpad_vertices_to_initial()
        self._refresh_attachment_reference()
        self.init_pose_mat = self.get_attach_pose().to_transformation_matrix()
        if restored:
            self._debug_log("reset gelpad vertices and attachment reference")

    def reset_reference(self):
        if not self._is_xsense_sensor():
            marker_simulator = getattr(self.sensor, "marker_motion_simulator", None)
            marker_motion_sim = getattr(marker_simulator, "marker_motion_sim", None)
            if marker_motion_sim is None:
                return False
            marker_motion_sim.init_vertices()
            return True

        reference_points = self._get_gelpad_points_for_current_attachment()
        if reference_points is not None:
            self._write_gelpad_vertices(reference_points)
        self._refresh_attachment_reference()
        marker_simulator = getattr(self.sensor, "marker_motion_simulator", None)
        marker_motion_sim = getattr(marker_simulator, "marker_motion_sim", None)
        if marker_motion_sim is None:
            return False
        marker_motion_sim.init_vertices()
        return True

    def reset_marker_reference(self):
        marker_simulator = getattr(self.sensor, "marker_motion_simulator", None)
        marker_motion_sim = getattr(marker_simulator, "marker_motion_sim", None)
        if marker_motion_sim is None:
            return False

        reference_updated = marker_motion_sim.init_vertices()
        if reference_updated is False:
            return False
        if hasattr(marker_simulator, 'reset_reference'):
            marker_simulator.reset_reference()
        marker_simulator.marker_data.zero_()
        output = self.sensor.data.output
        marker_motion = None
        if 'marker_motion' in output or 'marker_rgb' in output:
            marker_motion = marker_simulator.marker_motion_simulation()
        if 'marker_motion' in output:
            output['marker_motion'].zero_()
            output['marker_motion'][:] = marker_motion
        if (
            'marker_rgb' in output
            and 'tactile_rgb' in output
            and marker_motion is not None
        ):
            marker_uv = marker_motion[0, 1]
            marker_img = marker_simulator.draw_markers(marker_uv=marker_uv)
            tactile_rgb = output['tactile_rgb'].to(dtype=torch.float32) / 255.0
            tactile_rgb *= torch.dstack([marker_img / 255.0] * 3)
            output['marker_rgb'] = (tactile_rgb * 255.0).to(dtype=torch.uint8)
        return True
    
    def get_min_depth(self):
        return torch.min(self.sensor.data.output['height_map']).item()

class TactileManager:
    def __init__(self, cfg_list: list[TactileCfg], task:'BaseTask'):
        self.task = task
        self.scene = task.scene
        self.uipc_sim = task.uipc_sim
        self.robot = task._robot_manager.robot
        
        self.tactiles = {
            cfg.name: VisualTactileSensor(
                cfg.name, cfg, self.robot, self.scene, self.uipc_sim
            ) for cfg in cfg_list
        }
        grid = tuple(getattr(task.cfg, "force_field_grid", (64, 48)))
        for tact in self.tactiles.values():
            tact.force_field_grid = grid

    def update(self, dt, force_recompute=False):
        for tact in self.tactiles.values():
            tact.update(dt=dt, force_recompute=force_recompute)
 
    def set_debug_vis(self, debug_vis):
        if not debug_vis: return
        for tact in self.tactiles.values():
            tact.set_debug_vis()

    def get_observations(self, data_types: list[str] = None):
        obs = {}
        for name, tact in self.tactiles.items():
            obs[name] = tact.get_observations(data_types)
        return obs

    def dump_force_field_meta(self, save_dir, grid=None):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, tact in self.tactiles.items():
            tact.dump_force_field_meta(save_dir / f"ff_meta_{name}.npz", grid=grid)

    def get_min_depth(self):
        self.task._update_render()
        depth = []
        for tact in self.tactiles.values():
            depth.append(tact.get_min_depth())
        return torch.tensor(depth, dtype=torch.float32, device=self.task.device)

    def reset_reference(self):
        results = {}
        for name, tact in self.tactiles.items():
            results[name] = bool(tact.reset_reference())
        return results

    def reset_marker_reference(self):
        results = {}
        for name, tact in self.tactiles.items():
            results[name] = bool(tact.reset_marker_reference())
        return results

    def _reset_idx(self):
        for tact in self.tactiles.values():
            tact._reset_idx()

    def setup(self):
        for tact in self.tactiles.values():
            tact.setup()
