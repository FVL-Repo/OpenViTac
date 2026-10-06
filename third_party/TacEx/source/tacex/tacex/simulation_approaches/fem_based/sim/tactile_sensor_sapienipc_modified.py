###
# Implementation of the simulation approach from
# @ARTICLE{chen2024tactilesim2real,
#     author={Chen, Weihang and Xu, Jing and Xiang, Fanbo and Yuan, Xiaodi and Su, Hao and Chen, Rui},
#     journal={IEEE Transactions on Robotics},
#     title={General-Purpose Sim2Real Protocol for Learning Contact-Rich Manipulation With Marker-Based Visuotactile Sensors},
#     year={2024},
#     volume={40},
#     pages={1509-1526},
#     doi={10.1109/TRO.2024.3352969}
# }
# Original code can be found here
# https://github.com/chuanyune/ManiSkill-ViTac2025/blob/a3d7df54bca9a2e57f34b37be3a3df36dc218915/Track_1/envs/tactile_sensor_sapienipc.py
##

import cv2
import os
import time
import math
import numpy as np
import torch

import usdrt
import usdrt.UsdGeom
from sklearn.neighbors import NearestNeighbors

import isaaclab.utils.math as math_utils

from tacex_uipc.objects import UipcObject
from tacex_uipc.sim import UipcSim

from .utils.geometry import in_hull

try:
    from isaacsim.util.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()
except Exception:
    import warnings

    warnings.warn("_debug_draw failed to import", ImportWarning)
    draw = None

from .gelpad_info import *
class VisionTactileSensorUIPC:
    def __init__(
        self,
        uipc_gelpad: UipcObject,
        camera,
        sensor_type: str,
        tactile_img_width=320,
        tactile_img_height=240,
        marker_shape=(9, 7),
        marker_interval=(2.40625, 2.45833),
        marker_random_rotation_range=0,  # 标记点旋转范围（rad）
        marker_random_translation_range=(0, 0),  # 随机偏移范围（mm）
        marker_random_noise=0.0,
        sub_marker_num=0,
        marker_lose_tracking_probability=0.0,
        normalize: bool = False,
        num_markers: int = 128,
        camera_to_surface: float = 0.0283,
        real_size: tuple[float, float] = (0.0266, 0.0209),
        marker_radius: float = 12,
        marker_binding_center_x_m: float | None = None,
        **kwargs,
    ):
        """
        param: marker_interval_rang, in mm.
        param: marker_rotation_range: overall marker rotation, in radian.
        param: marker_translation_range: overall marker translation, in mm. first two elements: x-axis; last two elements: y-xis.
        param: marker_pos_shift_range: independent marker position shift, in mm, in x- and y-axis. caused by fabrication errors.
        param: marker_random_noise: std of Gaussian marker noise, in pixel. caused by CMOS noise and image processing.
        param: loss_tracking_probability: the probability of losing tracking, appled to each marker
        param: normalize: whether to normalize the output marker flow
        param: marker_flow_size: the size of the output marker flow
        param: camera_params: (fx, fy, cx, cy, distortion)
        """
        
        self.gelpad_obj = uipc_gelpad
        self.sensor_type = sensor_type
        self.uipc_sim: UipcSim = uipc_gelpad.uipc_sim
        self.scene = self.uipc_sim.scene

        self.camera = camera

        use_precomputed_indices = self.sensor_type in CONSTRAIN_PTS
        if not use_precomputed_indices:
            gelpad_info = get_gelpad_info(self.gelpad_obj.uipc_meshes[0])
            self.constrain_ids = gelpad_info['bottom']['ids']
            self.faces_on_surfaces = gelpad_info['surface']['faces']
        else:
            self.constrain_ids = CONSTRAIN_PTS[self.sensor_type]
            self.faces_on_surfaces = SURFACE_FACES[self.sensor_type]
        self._debug_xsense_camera_faces(self.faces_on_surfaces)
        self.vertices_on_surface = np.sort(np.unique(self.faces_on_surfaces.flatten()))
        self.init_surface_vertices = self.get_surface_vertices_world()

        self.marker_shape = marker_shape
        self.tactile_img_width = tactile_img_width
        self.tactile_img_height = tactile_img_height

        self.marker_interval = marker_interval
        self.marker_random_rotation_range = marker_random_rotation_range
        self.marker_random_translation_range = marker_random_translation_range
        self.marker_random_noise = marker_random_noise
        self.marker_lose_tracking_probability = marker_lose_tracking_probability
        self.normalize = normalize
        self.num_markers = num_markers

        self.sub_marker_num = sub_marker_num
        self.marker_radius = marker_radius
        self.marker_binding_center_x_m = marker_binding_center_x_m

        real_size = np.array(real_size)
        img_size = np.array([tactile_img_width, tactile_img_height])
        fx, fy = img_size * camera_to_surface / real_size
        cx, cy = img_size / 2
        self.camera_intrinsic = np.array(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.camera_distort_coeffs = np.array([0, 0, 0, 0, 0], dtype=np.float32)

        self.marker_grid = self._gen_marker_grid()
        self.init_vertices()
        # self.phong_shading_renderer = PhongShadingRenderer()
    
    def init_vertices(self):
        surface_vertices_camera = self.get_surface_vertices_camera().clone()
        vertices_camera = self.get_vertices_camera()
        constrain_pts = vertices_camera[self.constrain_ids].cpu().numpy()

        if str(self.sensor_type).startswith("xense"):
            valid, diagnostics = self._validate_xsense_reference_geometry(
                surface_vertices_camera,
                constrain_pts,
            )
            self.last_reference_diagnostics = diagnostics
            if not valid:
                reject_count = int(getattr(self, "_reference_reject_count", 0)) + 1
                self._reference_reject_count = reject_count
                if reject_count == 1 or reject_count % 20 == 0:
                    print(
                        "[xense-reference] rejected transient FEM reference",
                        diagnostics,
                    )
                if hasattr(self, "reference_surface_vertices_camera"):
                    return False
                raise RuntimeError(
                    "Initial XSense FEM reference is outside the physical camera range: "
                    f"{diagnostics}"
                )

        self.init_surface_vertices_camera = surface_vertices_camera
        self.reference_surface_vertices_camera = surface_vertices_camera.clone()
        if not str(self.sensor_type).startswith("xense") or not hasattr(self, "marker_surf_idx"):
            self.marker_surf_idx, self.marker_weight = self._gen_marker_weight(self.marker_grid)
        self.constrain_pts = constrain_pts
        previous_reject_count = int(getattr(self, "_reference_reject_count", 0))
        self._reference_reject_count = 0
        if str(self.sensor_type).startswith("xense"):
            self.last_reference_diagnostics = {
                **self.last_reference_diagnostics,
                "accepted": True,
                "settled_after_rejections": previous_reject_count,
            }
            if previous_reject_count:
                print(
                    "[xense-reference] accepted settled FEM reference",
                    self.last_reference_diagnostics,
                )
        return True

    def _validate_xsense_reference_geometry(self, surface_vertices, constrain_pts):
        surface = surface_vertices.detach().cpu().numpy().astype(np.float64)
        constrain = np.asarray(constrain_pts, dtype=np.float64)
        surface_finite = bool(np.isfinite(surface).all())
        constrain_finite = bool(np.isfinite(constrain).all())

        surface_depth_percentiles = (
            np.percentile(surface[:, 2], [1.0, 5.0, 50.0, 95.0, 99.0])
            if surface.size and surface_finite
            else np.full(5, np.nan, dtype=np.float64)
        )
        surface_depth = float(surface_depth_percentiles[2])
        constrain_depth = (
            float(np.median(constrain[:, 2])) if constrain.size else float("nan")
        )
        surface_lateral_max = (
            float(np.abs(surface[:, :2]).max()) if surface.size else float("inf")
        )
        constrain_lateral_max = (
            float(np.abs(constrain[:, :2]).max()) if constrain.size else float("inf")
        )

        surface_plane_p95 = float("inf")
        if surface.size and surface_finite:
            surface_centered = surface - np.median(surface, axis=0, keepdims=True)
            _, _, right_vectors = np.linalg.svd(surface_centered, full_matrices=False)
            surface_plane_distance = np.abs(surface_centered @ right_vectors[-1])
            surface_plane_p95 = float(np.percentile(surface_plane_distance, 95.0))

        marker_grid_affine_p95_px = float("nan")
        if hasattr(self, "marker_surf_idx"):
            marker_xyz = (
                surface_vertices.detach().cpu().numpy()[self.marker_surf_idx]
                * self.marker_weight[..., None]
            ).sum(axis=1)
            marker_uv = self.gen_marker_uv(marker_xyz)
            grid_columns, grid_rows = (int(value) for value in self.marker_shape)
            grid_y, grid_x = np.meshgrid(
                np.arange(grid_rows, dtype=np.float64),
                np.arange(grid_columns, dtype=np.float64),
                indexing="ij",
            )
            grid_design = np.stack(
                [grid_x.ravel(), grid_y.ravel(), np.ones(grid_rows * grid_columns)],
                axis=1,
            )
            if marker_uv.shape[0] == grid_design.shape[0]:
                affine = np.linalg.lstsq(grid_design, marker_uv, rcond=None)[0]
                affine_residual = np.linalg.norm(
                    marker_uv - grid_design @ affine,
                    axis=1,
                )
                marker_grid_affine_p95_px = float(
                    np.percentile(affine_residual, 95.0)
                )

        valid = (
            surface_finite
            and constrain_finite
            and 0.024 <= surface_depth_percentiles[0]
            and surface_depth_percentiles[4] <= 0.030
            and 0.015 <= constrain_depth <= 0.030
            and surface_lateral_max <= 0.050
            and constrain_lateral_max <= 0.050
            and surface_plane_p95 <= 0.0002
            and (
                not np.isfinite(marker_grid_affine_p95_px)
                or marker_grid_affine_p95_px <= 2.0
            )
        )
        diagnostics = {
            "accepted": bool(valid),
            "surface_depth_mm": surface_depth * 1000.0,
            "surface_depth_p01_mm": float(surface_depth_percentiles[0] * 1000.0),
            "surface_depth_p99_mm": float(surface_depth_percentiles[4] * 1000.0),
            "constrain_depth_mm": constrain_depth * 1000.0,
            "surface_lateral_max_mm": surface_lateral_max * 1000.0,
            "constrain_lateral_max_mm": constrain_lateral_max * 1000.0,
            "surface_plane_p95_mm": surface_plane_p95 * 1000.0,
            "marker_grid_affine_p95_px": marker_grid_affine_p95_px,
        }
        return bool(valid), diagnostics

    def get_vertices_world(self):
        v = self.gelpad_obj._data.nodal_pos_w
        return v

    def get_surface_vertices_world(self):
        all_v = self.gelpad_obj._data.nodal_pos_w
        surf_v = all_v[self.vertices_on_surface]
        return surf_v

    # todo find out what's wrong with this method -> frame coor. sys. seems to be wrong
    def transform_camera_to_world_frame(self, input_vertices):
        self.camera._update_poses(self.camera._ALL_INDICES)
        # math_utils.convert_camera_frame_orientation_convention
        cam_pos_w = self.camera._data.pos_w
        cam_quat_w = self.camera._data.quat_w_ros  # quat_w_opengl#quat_w_world
        v_cv = math_utils.transform_points(input_vertices, pos=cam_pos_w, quat=cam_quat_w)
        return v_cv

    def transform_world_to_camera_frame(self, input_vertices):
        self.camera._update_poses(self.camera._ALL_INDICES)
        # math_utils.convert_camera_frame_orientation_convention
        cam_pos_w = self.camera._data.pos_w
        cam_quat_w = self.camera._data.quat_w_ros
        cam_quat_w_inv = math_utils.quat_inv(cam_quat_w)

        rot_inv = math_utils.matrix_from_quat(cam_quat_w_inv)
        # convert to batched
        if rot_inv.dim() == 2:
            rot_inv = rot_inv[None]  # (3, 3) -> (1, 3, 3)

        t_target = input_vertices - cam_pos_w
        # convert to batched #todo fix it for multi env
        t_target = t_target[None, :, :]  # (N, 3) -> (N, 1, 3)

        v_cv = torch.matmul(rot_inv.to(torch.float64), t_target.transpose_(1, 2))
        v_cv = v_cv.transpose_(1, 2)
        # todo fix it for multi env
        v_cv = v_cv[0]
        return v_cv

    def get_init_surface_vertices_camera(self):
        return self.transform_world_to_camera_frame(self.get_surface_vertices_world()).clone()

    def transform_to_init_gelpad_frame(self, input_vertices):
        world_tf = self.gelpad_obj.init_world_transform
        pos_w, rot_mat = math_utils.unmake_pose(world_tf)
        quat_w = math_utils.quat_from_matrix(rot_mat).type(torch.float32)

        vertices_gelpad_frame = math_utils.transform_points(input_vertices, pos=pos_w, quat=quat_w)
        return vertices_gelpad_frame

    def get_surface_vertices_in_gelpad_frame(self):
        v = self.get_surface_vertices_world()
        v_cv = self.transform_to_init_gelpad_frame(v)
        return v_cv

    def get_vertices_camera(self):
        v = self.get_vertices_world()
        v_cv = self.transform_world_to_camera_frame(v)
        return v_cv

    def get_surface_vertices_camera(self):
        v = self.get_surface_vertices_world()
        v_cv = self.transform_world_to_camera_frame(v)
        return v_cv

    def camera_points_to_opencv(self, points):
        return np.asarray(points)

    def _debug_xsense_camera_faces(self, faces):
        if not str(self.sensor_type).startswith("xense") or os.environ.get("XENSE_MARKER_BINDING_DEBUG") != "1":
            return
        vertices = self.get_vertices_camera().cpu().numpy()
        face_vertices = vertices[faces]
        normals = np.cross(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(1e-12)
        centers = face_vertices.mean(axis=1)
        depth = centers[:, 2]
        diagnostics = {
            "prim_path": str(self.gelpad_obj.cfg.prim_path),
            "vertex_bounds": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
            "face_count": int(len(faces)),
            "depth_range": [float(depth.min()), float(depth.max())],
        }
        for name, mask in (
            ("negative_y", normals[:, 1] < -0.5),
            ("positive_y", normals[:, 1] > 0.5),
            ("aligned_y", np.abs(normals[:, 1]) > 0.5),
            ("negative_z", normals[:, 2] < -0.5),
            ("positive_z", normals[:, 2] > 0.5),
            ("aligned_z", np.abs(normals[:, 2]) > 0.5),
        ):
            group_vertices = face_vertices[mask].reshape(-1, 3)
            diagnostics[name] = {
                "faces": int(mask.sum()),
                "depth_range": [float(depth[mask].min()), float(depth[mask].max())] if mask.any() else None,
                "bounds": [group_vertices.min(axis=0).tolist(), group_vertices.max(axis=0).tolist()] if mask.any() else None,
            }
        print("[xense-camera-faces]", diagnostics)

    def set_reference_surface_vertices_camera(self):
        surface_vertices_camera = self.get_surface_vertices_camera().clone()
        constrain_pts = self.get_vertices_camera()[self.constrain_ids].cpu().numpy()
        if str(self.sensor_type).startswith("xense"):
            valid, diagnostics = self._validate_xsense_reference_geometry(
                surface_vertices_camera,
                constrain_pts,
            )
            self.last_reference_diagnostics = diagnostics
            if not valid:
                return False
        self.reference_surface_vertices_camera = surface_vertices_camera
        return True

    def _gen_marker_grid(self):
        '''生成标记格点'''
        def rand_between(a, b, size=None)->float:
            '''在 [a, b] 范围内生成一个随机数'''
            return (b - a) * np.random.random(size) + a

        # 间隔
        marker_interval_x, marker_interval_y = self.marker_interval
        # 在给定范围内随机旋转角度
        rot_range = self.marker_random_rotation_range
        marker_rotation_angle = rand_between(-rot_range, rot_range)
        # 在给定范围内随机偏移
        trans_range = self.marker_random_translation_range
        marker_translation_x = rand_between(-trans_range[0], trans_range[0])
        marker_translation_y = rand_between(-trans_range[1], trans_range[1])

        # 生成网格
        sx, sy = self.marker_shape
        marker_width = marker_interval_x * (sx - 1)
        marker_x_start = - marker_width / 2 + marker_translation_x
        marker_x = marker_x_start + np.arange(sx) * marker_interval_x

        marker_height = marker_interval_y * (sy - 1)
        marker_y_start = - marker_height / 2 + marker_translation_y
        marker_y = marker_y_start + np.arange(sy) * marker_interval_y

        marker_xy = np.array(np.meshgrid(marker_x, marker_y)).reshape((2, -1)).T

        rot_mat = np.array([
            [math.cos(marker_rotation_angle), -math.sin(marker_rotation_angle)],
            [math.sin(marker_rotation_angle), math.cos(marker_rotation_angle)],
        ])
        marker_rotated_xy = (marker_xy @ rot_mat.T) / 1000.0

        if self.sub_marker_num > 0:
            sub_marker_num = self.sub_marker_num
            sub_marker = np.zeros((marker_rotated_xy.shape[0]*sub_marker_num, 2))
            for i in range(marker_rotated_xy.shape[0]):
                x, y = marker_rotated_xy[i]
                theta = np.linspace(0, 2 * np.pi, sub_marker_num, endpoint=False)
                np.random.shuffle(theta)
                start = i*sub_marker_num
                sub_marker[start:start+sub_marker_num, 0] = x + self.marker_radius/2000 * np.cos(theta)
                sub_marker[start:start+sub_marker_num, 1] = y + self.marker_radius/2000 * np.sin(theta)

            marker_grid = np.concatenate([marker_rotated_xy, sub_marker], axis=0)
        else:
            marker_grid = marker_rotated_xy
        return marker_grid
    
    def _gen_marker_weight(self, marker_pts):
        if not str(self.sensor_type).startswith("xense"):
            return self._gen_native_marker_weight(marker_pts)

        surface_pts_camera = self.init_surface_vertices_camera.cpu().numpy()
        surface_pts = self.camera_points_to_opencv(surface_pts_camera)[:, :2]
        marker_pts = np.asarray(marker_pts).copy()
        if str(self.sensor_type).startswith("xense"):
            surface_center = 0.5 * (surface_pts.min(axis=0) + surface_pts.max(axis=0))
            if self.marker_binding_center_x_m is not None:
                surface_center[0] = float(self.marker_binding_center_x_m)
            marker_pts += surface_center
        # marker_on_surface = in_hull(marker_pts, surface_pts)
        # marker_pts = marker_pts[marker_on_surface]

        face_idx_to_surface_idx = np.zeros(np.max(self.faces_on_surfaces)+1, dtype=np.int32)
        face_idx_to_surface_idx[self.vertices_on_surface] = np.arange(surface_pts.shape[0])
        faces_v_on_surface = surface_pts[face_idx_to_surface_idx[self.faces_on_surfaces.flatten()]].reshape(-1, 3, 2)
        edge_1 = faces_v_on_surface[:, 1] - faces_v_on_surface[:, 0]
        edge_2 = faces_v_on_surface[:, 2] - faces_v_on_surface[:, 0]
        face_double_area = np.abs(edge_1[:, 0] * edge_2[:, 1] - edge_1[:, 1] * edge_2[:, 0])
        valid_face_mask = face_double_area > 1e-14
        if not np.any(valid_face_mask):
            raise ValueError("No non-degenerate gelpad surface faces found for marker initialization")
        surface_faces_for_markers = self.faces_on_surfaces[valid_face_mask]
        faces_v_on_surface = faces_v_on_surface[valid_face_mask]
        f_centers = np.mean(faces_v_on_surface, axis=1)

        n_neighbors = len(f_centers) if str(self.sensor_type).startswith("xense") else min(8, len(f_centers))
        if n_neighbors == 0:
            raise ValueError("No gelpad surface faces found for marker initialization")
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree").fit(f_centers)
        distances, face_idx = nbrs.kneighbors(marker_pts)

        marker_pts_surface_idx = []
        marker_pts_surface_weight = []
        valid_marker_idx = []
        inside_marker_count = 0

        # compute barycentric weight of each vertex
        for i in range(marker_pts.shape[0]):
            possible_face_ids = face_idx[i]
            p = marker_pts[i]
            fallback_idx = None
            fallback_weight = None
            fallback_distance = np.inf
            for possible_face_id in possible_face_ids.tolist():
                face_vertices_idx = face_idx_to_surface_idx[surface_faces_for_markers[possible_face_id]]
                vertices_of_face_i = surface_pts[face_vertices_idx]
                p0, p1, p2 = vertices_of_face_i
                A = np.stack([p1 - p0, p2 - p0], axis=1)
                det = np.linalg.det(A)
                if abs(det) <= 1e-14:
                    continue
                w12 = np.linalg.solve(A, p - p0)
                weight = np.array([1 - w12.sum(), w12[0], w12[1]])
                if w12[0] >= 0 and w12[1] >= 0 and w12[0] + w12[1] <= 1:
                    fallback_idx = face_vertices_idx
                    fallback_weight = weight
                    inside_marker_count += 1
                    break
                for edge_start, edge_end, edge_weights in (
                    (0, 1, (0, 1)),
                    (1, 2, (1, 2)),
                    (2, 0, (2, 0)),
                ):
                    edge = vertices_of_face_i[edge_end] - vertices_of_face_i[edge_start]
                    edge_length_sq = float(np.dot(edge, edge))
                    if edge_length_sq <= 1e-14:
                        continue
                    t = float(np.clip(np.dot(p - vertices_of_face_i[edge_start], edge) / edge_length_sq, 0.0, 1.0))
                    closest = vertices_of_face_i[edge_start] + t * edge
                    distance = float(np.dot(p - closest, p - closest))
                    if distance < fallback_distance:
                        fallback_distance = distance
                        fallback_idx = face_vertices_idx
                        fallback_weight = np.zeros(3, dtype=np.float64)
                        fallback_weight[edge_weights[0]] = 1.0 - t
                        fallback_weight[edge_weights[1]] = t
            if fallback_idx is None:
                nearest_vertex = np.argmin(np.linalg.norm(surface_pts - p, axis=1))
                fallback_idx = np.array([nearest_vertex, nearest_vertex, nearest_vertex], dtype=np.int32)
                fallback_weight = np.array([1.0, 0.0, 0.0])
            marker_pts_surface_idx.append(fallback_idx)
            marker_pts_surface_weight.append(fallback_weight)
            valid_marker_idx.append(i)

        valid_marker_idx = np.array(valid_marker_idx).astype(np.int32)
        marker_pts = marker_pts[valid_marker_idx]
        marker_pts_surface_idx = np.stack(marker_pts_surface_idx)
        marker_pts_surface_weight = np.stack(marker_pts_surface_weight)
        reconstructed_marker_pts = (
            surface_pts[marker_pts_surface_idx] * marker_pts_surface_weight[..., None]
        ).sum(1)[:, :2]
        marker_fit_err = np.abs(reconstructed_marker_pts - marker_pts).max()
        if self.sensor_type == "xensews" and not np.allclose(reconstructed_marker_pts, marker_pts):
            print(f"[xense-debug] marker_surface_fallback max_err={marker_fit_err}")
        rounded_weights = np.round(marker_pts_surface_weight, decimals=6)
        binding_keys = np.concatenate([marker_pts_surface_idx, rounded_weights], axis=1)
        unique_binding_count = int(np.unique(binding_keys, axis=0).shape[0])
        if str(self.sensor_type).startswith("xense") and os.environ.get("XENSE_MARKER_BINDING_DEBUG") == "1":
            print(
                "[xense-marker-binding]",
                {
                    "prim_path": str(self.gelpad_obj.cfg.prim_path),
                    "marker_bounds": [marker_pts.min(axis=0).tolist(), marker_pts.max(axis=0).tolist()],
                    "surface_bounds": [surface_pts.min(axis=0).tolist(), surface_pts.max(axis=0).tolist()],
                    "surface_camera_bounds": [surface_pts_camera.min(axis=0).tolist(), surface_pts_camera.max(axis=0).tolist()],
                    "surface_vertices": int(surface_pts.shape[0]),
                    "surface_faces": int(faces_v_on_surface.shape[0]),
                    "inside_markers": int(inside_marker_count),
                    "fallback_markers": int(marker_pts.shape[0] - inside_marker_count),
                    "unique_bindings": unique_binding_count,
                    "unique_reconstructed": int(np.unique(np.round(reconstructed_marker_pts, decimals=6), axis=0).shape[0]),
                    "weight_min": float(marker_pts_surface_weight.min()),
                    "weight_max": float(marker_pts_surface_weight.max()),
                    "fit_error": float(marker_fit_err),
                },
            )
        if str(self.sensor_type).startswith("xense"):
            if inside_marker_count != marker_pts.shape[0]:
                raise RuntimeError(
                    "Every XSense marker must lie inside a FEM surface triangle: "
                    f"inside={inside_marker_count}, expected={marker_pts.shape[0]}"
                )
            if unique_binding_count != marker_pts.shape[0]:
                raise RuntimeError(
                    "Every XSense marker must have a unique FEM binding: "
                    f"unique={unique_binding_count}, expected={marker_pts.shape[0]}"
                )

        return marker_pts_surface_idx, marker_pts_surface_weight

    def _gen_native_marker_weight(self, marker_pts):
        surface_pts = self.init_surface_vertices_camera.cpu().numpy()[:, :2]

        face_idx_to_surface_idx = np.zeros(
            np.max(self.faces_on_surfaces) + 1, dtype=np.int32
        )
        face_idx_to_surface_idx[self.vertices_on_surface] = np.arange(
            surface_pts.shape[0]
        )
        faces_v_on_surface = surface_pts[
            face_idx_to_surface_idx[self.faces_on_surfaces.flatten()]
        ].reshape(-1, 3, 2)
        face_centers = np.mean(faces_v_on_surface, axis=1)

        neighbors = NearestNeighbors(n_neighbors=4, algorithm="ball_tree").fit(
            face_centers
        )
        _, face_idx = neighbors.kneighbors(marker_pts)

        marker_pts_surface_idx = []
        marker_pts_surface_weight = []
        valid_marker_idx = []

        for marker_idx in range(marker_pts.shape[0]):
            possible_face_ids = face_idx[marker_idx]
            marker_point = marker_pts[marker_idx]
            for possible_face_id in possible_face_ids.tolist():
                face_vertices_idx = face_idx_to_surface_idx[
                    self.faces_on_surfaces[possible_face_id]
                ]
                vertices_of_face = surface_pts[face_vertices_idx]
                point_0, point_1, point_2 = vertices_of_face
                matrix = np.stack([point_1 - point_0, point_2 - point_0], axis=1)
                weight_12 = np.linalg.inv(matrix) @ (marker_point - point_0)
                weight = np.array(
                    [1 - weight_12.sum(), weight_12[0], weight_12[1]]
                )
                is_inside = (
                    weight_12[0] >= 0
                    and weight_12[1] >= 0
                    and weight_12[0] + weight_12[1] <= 1
                )
                if possible_face_id == possible_face_ids[0]:
                    marker_pts_surface_idx.append(face_vertices_idx)
                    marker_pts_surface_weight.append(weight)
                    valid_marker_idx.append(marker_idx)
                    if is_inside:
                        break
                elif is_inside:
                    marker_pts_surface_idx[-1] = face_vertices_idx
                    marker_pts_surface_weight[-1] = weight
                    valid_marker_idx[-1] = marker_idx
                    break

        valid_marker_idx = np.asarray(valid_marker_idx, dtype=np.int32)
        valid_marker_pts = marker_pts[valid_marker_idx]
        marker_pts_surface_idx = np.stack(marker_pts_surface_idx)
        marker_pts_surface_weight = np.stack(marker_pts_surface_weight)
        reconstructed = (
            surface_pts[marker_pts_surface_idx]
            * marker_pts_surface_weight[..., None]
        ).sum(1)[:, :2]
        assert np.allclose(reconstructed, valid_marker_pts), (
            "max err: "
            f"{np.abs(reconstructed - valid_marker_pts).max()}"
        )
        return marker_pts_surface_idx, marker_pts_surface_weight

    def gen_marker_uv(self, marker_pts):
        marker_pts = self.camera_points_to_opencv(marker_pts)
        marker_uv = cv2.projectPoints(
            marker_pts, 
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            self.camera_intrinsic,
            self.camera_distort_coeffs
        )[0].squeeze(1)

        return marker_uv

    def _register_marker_points_to_constrained_frame(self, marker_pts):
        reference_constrain = np.asarray(self.constrain_pts, dtype=np.float64)
        current_constrain = self.get_vertices_camera()[self.constrain_ids].cpu().numpy().astype(np.float64)
        if reference_constrain.shape != current_constrain.shape or reference_constrain.shape[0] < 3:
            return marker_pts

        reference_center = reference_constrain.mean(axis=0, keepdims=True)
        current_center = current_constrain.mean(axis=0, keepdims=True)
        covariance = (current_constrain - current_center).T @ (reference_constrain - reference_center)
        left, _, right = np.linalg.svd(covariance)
        rotation = left @ right
        if np.linalg.det(rotation) < 0.0:
            left[:, -1] *= -1.0
            rotation = left @ right
        return (marker_pts - current_center) @ rotation + reference_center

    def gen_marker_flow(self):
        is_xsense = str(self.sensor_type).startswith("xense")
        init_marker_pts = (
            self.reference_surface_vertices_camera[self.marker_surf_idx].cpu().numpy()
            * self.marker_weight[..., None]
        ).sum(1)
        curr_marker_pts = (
            self.get_surface_vertices_camera()[self.marker_surf_idx].cpu().numpy()
            * self.marker_weight[..., None]
        ).sum(1)

        if is_xsense:
            curr_marker_pts = self._register_marker_points_to_constrained_frame(curr_marker_pts)
        else:
            mean_motion = np.mean(
                self.get_vertices_camera()[self.constrain_ids].cpu().numpy() - self.constrain_pts,
                axis=0,
            )
            curr_marker_pts[:, :2] -= mean_motion[:2]

        init_marker_uv = self.gen_marker_uv(init_marker_pts)
        curr_marker_uv = self.gen_marker_uv(curr_marker_pts)
        marker_flow = np.stack([init_marker_uv, curr_marker_uv], axis=0)

        if not is_xsense:
            marker_mask = np.logical_and.reduce([
                curr_marker_uv[:, 0] > 0,
                curr_marker_uv[:, 0] < self.tactile_img_width,
                curr_marker_uv[:, 1] > 0,
                curr_marker_uv[:, 1] < self.tactile_img_height,
            ])
            marker_flow = marker_flow[:, marker_mask]

        # post processing
        if not is_xsense and self.marker_lose_tracking_probability > 0:
            no_lose_tracking_mask = np.random.rand(marker_flow.shape[1]) > self.marker_lose_tracking_probability
            marker_flow = marker_flow[:, no_lose_tracking_mask, :]
        if not is_xsense:
            noise = np.random.randn(*marker_flow.shape) * self.marker_random_noise
            marker_flow += noise

        original_point_num = marker_flow.shape[1]

        if is_xsense:
            if original_point_num != self.num_markers:
                raise RuntimeError(
                    "XSense must preserve every FEM marker binding: "
                    f"got {original_point_num}, expected {self.num_markers}"
                )
            ret = marker_flow.copy()
        elif original_point_num >= self.num_markers:
            chosen = np.random.choice(original_point_num, self.num_markers, replace=False)
            ret = marker_flow[:, chosen, ...]
        else:
            ret = np.zeros((marker_flow.shape[0], self.num_markers, marker_flow.shape[-1]))
            if original_point_num > 0:
                ret[:, :original_point_num, :] = marker_flow.copy()
                ret[:, original_point_num:, :] = ret[:, original_point_num - 1 : original_point_num, :]

        if self.normalize:
            ret /= self.tactile_img_width / 2
            ret -= 1.0

        ret = torch.tensor(ret, device="cuda:0")
        self.curr_marker_uv = ret[1]
        return ret

    def get_marker_img(self):
        curr_marker_uv = self.curr_marker_uv
        curr_marker_img = self.draw_markers(curr_marker_uv)
        # cv2.imwrite("curr_marker_img.png", curr_marker_img)
        return curr_marker_img

    # def gen_rgb_image(self):
    #     # generate RGB image from depth
    #     depth = self._gen_depth()
    #     rgb = self.phong_shading_renderer.generate(depth)
    #     rgb = rgb.astype(np.float64)

    #     # generate markers
    #     marker_grid = self._gen_marker_grid()
    #     marker_pts_surface_idx, marker_pts_surface_weight = self._gen_marker_weight(
    #         marker_grid
    #     )
    #     curr_marker_pts = (
    #         self.get_surface_vertices_in_camera_frame()[marker_pts_surface_idx]
    #         * marker_pts_surface_weight[..., None]
    #     ).sum(1)
    #     curr_marker_uv = self.gen_marker_uv(curr_marker_pts)

    #     curr_marker = self.draw_marker(curr_marker_uv)
    #     rgb = rgb.astype(np.float64)
    #     rgb *= np.dstack([curr_marker.astype(np.float64) / 255] * 3)
    #     rgb = rgb.astype(np.uint8)
    #     return rgb

    # def _gen_depth(self):
    #     # hide the gel to get the depth of the object in contact
    #     self.render_component.disable()
    #     self.cam_entity.set_pose(cv2ex2pose(self.get_camera_pose()))
    #     self.scene.update_render()
    #     ipc_update_render_all(self.scene)
    #     self.cam.take_picture()
    #     position = self.cam.get_picture("Position")  # [H, W, 4]
    #     depth = -position[..., 2]  # float in meter
    #     fem_smooth_sigma = 2
    #     depth = gaussian_filter(depth, fem_smooth_sigma)
    #     self.render_component.enable()

    #     return depth
