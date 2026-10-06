# Copyright (c) 2022-2025, The TacEx Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log
import omni.physics.tensors.impl.api as physx
import omni.usd
import usdrt
import usdrt.UsdGeom
from isaacsim.core.prims import XFormPrim
from pxr import UsdGeom

try:
    from isaacsim.util.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()
except Exception:
    import warnings

    warnings.warn("_debug_draw failed to import", ImportWarning)
    draw = None

import numpy as np

import warp as wp
from uipc import builtin, view
from uipc.constitution import (
    AffineBodyConstitution,
    DiscreteShellBending,
    ElasticModuli,
    HookeanSpring,
    KirchhoffRodBending,
    NeoHookeanShell,
    StableNeoHookean,
)
from uipc.geometry import extract_surface, flip_inward_triangles, label_surface, label_triangle_orient, linemesh, tetmesh, trimesh
from uipc.unit import MPa

import isaaclab.utils.string as string_utils
from isaaclab.assets import AssetBase, AssetBaseCfg
from isaaclab.utils import configclass

wp.init()


from tacex_uipc.utils import MeshGenerator, TetMeshCfg

from .uipc_object_deformable_data import UipcObjectDeformableData
from .uipc_object_rigid_data import UipcObjectRigidData

if TYPE_CHECKING:
    from tacex_uipc.sim import UipcIsaacAttachmentsCfg, UipcSim


@configclass
class UipcObjectCfg(AssetBaseCfg):
    mesh_cfg: TetMeshCfg = None
    # contact_model:

    mass_density: float = 1e3

    @configclass
    class AffineBodyConstitutionCfg:
        # class_type = AffineBodyConstitution # doesn't work, cause no builtin signature found for AffineBodyConstitution class
        m_kappa: float = 100.0
        """Stiffness (hardness) of the object
        in [MPa]

        E.g. 100.0 MPa = hard-rubber-like material
        """

        kinematic: bool = False
        """Makes the DoF of the ABD body fixed.

        """

    @configclass
    class StableNeoHookeanCfg:
        # class_type = StableNeoHookean
        youngs_modulus: float = 0.01
        """
        in [MPa]
        """

        poisson_rate: float = 0.49
        """ Poission Rate

        Has to be < 0.5.
        """

    @configclass
    class NeoHookeanShellCfg:
        # class_type = NeoHookeanShell
        youngs_modulus: float = 0.01
        """
        in [MPa]
        """

        poisson_rate: float = 0.499
        """ Poission Rate

        Has to be < 0.5.
        """

        thickness: float = 0.001
        """
        Cloth thickness in [m].
        """

        enable_bending: bool = True
        """Apply DiscreteShellBending in addition to the membrane shell constitution."""

        bending_stiffness: float = 10.0
        """Bending stiffness passed to DiscreteShellBending."""

        render_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Visual-only offset applied when writing simulated cloth vertices to Isaac/Fabric."""

    @configclass
    class HookeanSpringCfg:
        kappa: float = 4.0e4
        """Axial spring stiffness for 1D thread/cable elements."""

        thickness: float = 0.006
        """Collision thickness/radius for codimensional line contact in [m]."""

        enable_bending: bool = True
        """Apply KirchhoffRodBending in addition to the stretch spring constitution."""

        bending_stiffness: float = 1.0e5
        """Bending stiffness passed to KirchhoffRodBending."""

        render_radius: float = 0.004
        """Radius of the Isaac render tube used to visualize the 1D line mesh."""

        render_sides: int = 8
        """Number of sides used by the render tube cross-section."""

        render_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Visual-only offset applied when writing simulated thread vertices to Isaac/Fabric."""

    constitution_cfg: AffineBodyConstitutionCfg | StableNeoHookeanCfg | NeoHookeanShellCfg | HookeanSpringCfg = None

    attachment_cfg: UipcIsaacAttachmentsCfg = None


class UipcObject(AssetBase):
    """A rigid object asset class.

    Rigid objects are assets comprising of rigid bodies. They can be used to represent dynamic objects
    such as boxes, spheres, etc. A rigid body is described by its pose, velocity and mass distribution.

    For an asset to be considered a rigid object, the root prim of the asset must have the `USD RigidBodyAPI`_
    applied to it. This API is used to define the simulation properties of the rigid body. On playing the
    simulation, the physics engine will automatically register the rigid body and create a corresponding
    rigid body handle. This handle can be accessed using the :attr:`root_physx_view` attribute.

    .. note::

        For users familiar with Isaac Sim, the PhysX view class API is not the exactly same as Isaac Sim view
        class API. Similar to Isaac Lab, Isaac Sim wraps around the PhysX view API. However, as of now (2023.1 release),
        we see a large difference in initializing the view classes in Isaac Sim. This is because the view classes
        in Isaac Sim perform additional USD-related operations which are slow and also not required.

    .. _`USD RigidBodyAPI`: https://openusd.org/dev/api/class_usd_physics_rigid_body_a_p_i.html
    """

    cfg: UipcObjectCfg
    """Configuration instance for the rigid object."""

    def __init__(self, cfg: UipcObjectCfg, uipc_sim: UipcSim):
        """Initialize the uipc object.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)
        self._uipc_sim: UipcSim = uipc_sim

        prim_paths_expr = self.cfg.prim_path  # + "/mesh"
        omni.log.info(f"Initializing uipc objects {prim_paths_expr}...")
        self._prim_view = XFormPrim(prim_paths_expr=prim_paths_expr, name=f"{prim_paths_expr}", usd=False)
        self._prim_view.initialize()

        self.stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())

        self.uipc_scene_objects = []
        self.geo_slot_list = []
        self._is_line_mesh = isinstance(self.cfg.constitution_cfg, UipcObjectCfg.HookeanSpringCfg)
        self._line_render_fabric_prim = None

        def find_mesh(prim):
            if prim.GetTypeName() == "Mesh":
                return prim
            for child in prim.GetChildren():
                mesh_prim = find_mesh(child)
                if mesh_prim is not None:
                    return mesh_prim
            return None

        self.uipc_meshes = []
        # setup tet meshes for uipc
        for (
            prim
        ) in (
            self._prim_view.prims
        ):  # todo dont loop over all prims of the view -> just take one base prim. Rather loop over the prim children?
            # need to access the mesh data of the usd prim
            prim_children = [find_mesh(prim)]
            usd_mesh = UsdGeom.Mesh(prim_children[0])
            usd_mesh_path = str(usd_mesh.GetPath())
            omni.log.info("usd_mesh_path ", usd_mesh_path)

            if isinstance(self.cfg.constitution_cfg, UipcObjectCfg.NeoHookeanShellCfg):
                mesh, tet_surf_points_world, tet_surf_tri = self._create_cloth_mesh_from_usd(usd_mesh)
                replace_color = False
            elif self._is_line_mesh:
                mesh, tet_surf_points_world, tet_surf_tri = self._create_line_mesh_from_usd(usd_mesh)
                replace_color = False
            else:
                mesh, tet_surf_points_world, tet_surf_tri, replace_color = self._create_tet_mesh_from_usd(usd_mesh)

            self.uipc_meshes.append(mesh)

            # Set Vertex and Triangle data into USD mesh for rendering, skip
            MeshGenerator.update_usd_mesh(
                prim=usd_mesh, surf_points=tet_surf_points_world, triangles=tet_surf_tri,
                replace_color=replace_color
            )

            # enable contact for uipc meshes etc.
            # mesh = self.uipc_meshes[0] #todo code properly cloned envs (i.e. for instanced objects?)
            self._create_constitutions(mesh)

            # setup mesh updates via Fabric
            fabric_prim = self.stage.GetPrimAtPath(usdrt.Sdf.Path(usd_mesh_path))
            if not fabric_prim:
                omni.log.warning(f"Prim at path {usd_mesh_path} is not in Fabric")
            if not fabric_prim.HasAttribute("points"):
                omni.log.warning(f"Prim at path {usd_mesh_path} does not have points attribute")

            # Tell OmniHydra to render points from Fabric
            if not fabric_prim.HasAttribute("Deformable"):
                fabric_prim.CreateAttribute("Deformable", usdrt.Sdf.ValueTypeNames.PrimTypeTag, True)

            # extract world transform
            rtxformable = usdrt.Rt.Xformable(fabric_prim)
            rtxformable.CreateFabricHierarchyWorldMatrixAttr()
            # set world matrix to identity matrix -> uipc already gives us vertices in world frame
            rtxformable.GetFabricHierarchyWorldMatrixAttr().Set(usdrt.Gf.Matrix4d())

            # update fabric mesh with world coor. points
            render_offset = np.array(getattr(self.cfg.constitution_cfg, "render_offset", (0.0, 0.0, 0.0)))
            fabric_mesh_points_attr = fabric_prim.GetAttribute("points")
            fabric_mesh_points_attr.Set(usdrt.Vt.Vec3fArray(tet_surf_points_world + render_offset))

            self.fabric_prim = fabric_prim

            if self._is_line_mesh:
                self._line_render_fabric_prim = fabric_prim
                self._uipc_sim._line_renderers.append(self)
            else:
                # add fabric meshes to uipc sim class for updating the render meshes
                self._uipc_sim._fabric_meshes.append(fabric_prim)
                self._uipc_sim._fabric_mesh_offsets.append(render_offset)

                # save surface offsets for finding corresponding surface points of the meshes for rendering
                num_surf_points = tet_surf_points_world.shape[0]  # np.unique(tet_surf_indices)
                self._surf_vertex_offset_start = self._uipc_sim._surf_vertex_offsets[-1]
                self._surf_vertex_offset_end = self._surf_vertex_offset_start + num_surf_points
                self._uipc_sim._surf_vertex_offsets.append(self._surf_vertex_offset_end)

            # required for writing vertex positions to sim
            num_vertex_points = mesh.positions().view().shape[0]
            self._vertex_count = num_vertex_points

            # update local vertex offset of the subsystem
            if self._system_name not in self._uipc_sim._system_vertex_offsets:
                self._uipc_sim._system_vertex_offsets[self._system_name] = [0]
            self._uipc_sim._system_vertex_offsets[self._system_name].append(
                self._uipc_sim._system_vertex_offsets[self._system_name][-1] + self._vertex_count
            )
            self.local_system_id = len(self._uipc_sim._system_vertex_offsets[self._system_name]) - 1

            # will be updated once _uipc_sim.setup_sim() is called
            self.global_system_id = 0

            self._data = None

    """
    Properties
    """

    @property
    def data(self) -> UipcObjectDeformableData | UipcObjectRigidData:
        return self._data

    @property
    def num_instances(self) -> int:
        return self._prim_view.count

    @property
    def num_bodies(self) -> int:
        """Number of bodies in the asset.

        This is always 1 since each object is a single rigid body.
        """
        return 1

    @property
    def body_names(self) -> list[str]:
        """Ordered names of bodies in the rigid object."""
        prim_paths = self.root_physx_view.prim_paths[: self.num_bodies]
        return [path.split("/")[-1] for path in prim_paths]

    @property
    def uipc_sim(self) -> physx.RigidBodyView:
        """uipc simulation instance of this uipc object."""
        return self._uipc_sim

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        # TODO implement this
        pass
        # # resolve all indices
        # if env_ids is None:
        #     env_ids = slice(None)

    def write_data_to_sim(self):
        pass

    def update(self, dt: float):
        self._data.update(dt)

    """
    Operations - Finders.
    """

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Find bodies in the rigid body based on the name keys.

        Please check the :meth:`isaaclab.utils.string_utils.resolve_matching_names` function for more
        information on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the body indices and names.
        """
        return string_utils.resolve_matching_names(name_keys, self.body_names, preserve_order)

    """
    Operations - Write to simulation.
    """

    def write_vertex_positions_to_sim(self, vertex_positions: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).

        Args:
            root_pose: Root poses in simulation frame. Shape is (len(env_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # resolve all indices
        # physx_env_ids = env_ids
        # if env_ids is None:
        #     env_ids = slice(None)
        #     physx_env_ids = self._ALL_INDICES

        # # note: we need to do this here since tensors are not set into simulation until step.
        # # set into internal buffers
        # self._data.root_state_w[env_ids, :7] = root_pose.clone()
        # # convert root quaternion from wxyz to xyzw
        # root_poses_xyzw = self._data.root_state_w[:, :7].clone()
        # root_poses_xyzw[:, 3:] = math_utils.convert_quat(root_poses_xyzw[:, 3:], to="xyzw")
        # # set into simulation
        # self.root_physx_view.set_transforms(root_poses_xyzw, indices=physx_env_ids)
        # omni.log.info("")
        # omni.log.info(f"Write vertex pos for {self.cfg.prim_path} with obj id [{self.obj_id}]")

        # omni.log.info(f"num geo_slots: {len(self.geo_slot_list)}")
        # omni.log.info(f"global sys id: {self.global_system_id}")
        geo_slot = self.geo_slot_list[0]
        geo = geo_slot.geometry()
        gvo = geo.meta().find(builtin.global_vertex_offset)
        # omni.log.info(f"global Vertex Offset: {gvo.view()}")
        global_vertex_offset = int(gvo.view()[0])
        local_vertex_offset = self._uipc_sim._system_vertex_offsets[self._system_name][self.local_system_id - 1]
        # omni.log.info(f"system: {self._system_name}")
        # omni.log.info(f"local sys id: {self.local_system_id}")
        # omni.log.info(f"local vertex offset: {local_vertex_offset}")
        # omni.log.info(f"vertex count: {self._vertex_count}")
        # omni.log.info("")
        if self._system_name == "uipc::backend::cuda::AffineBodyDynamics":
            self.uipc_sim.world.write_vertex_pos_to_sim(
                vertex_positions.cpu().numpy(),
                global_vertex_offset,
                self.local_system_id - 1,
                self._vertex_count,
                self._system_name,
            )
        else:
            self.uipc_sim.world.write_vertex_pos_to_sim(
                vertex_positions.cpu().numpy(),
                global_vertex_offset,
                local_vertex_offset,
                self._vertex_count,
                self._system_name,
            )

    @property
    def vertex_positions(self) -> np.ndarray:
        """Current UIPC vertex positions for volume, shell, or line meshes."""
        if len(self.geo_slot_list) == 0:
            return np.zeros((0, 3), dtype=np.float64)
        geo_slot = self.geo_slot_list[0]
        return np.asarray(geo_slot.geometry().positions().view()).reshape(-1, 3)

    """
    Internal helper.
    """

    @staticmethod
    def _make_tube_mesh(centerline: np.ndarray, radius: float, sides: int):
        centerline = np.asarray(centerline, dtype=np.float64).reshape(-1, 3)
        sides = max(3, int(sides))
        if centerline.shape[0] == 0:
            return centerline, []

        tangents = np.zeros_like(centerline)
        if centerline.shape[0] == 1:
            tangents[:] = np.array([1.0, 0.0, 0.0])
        else:
            tangents[0] = centerline[1] - centerline[0]
            tangents[-1] = centerline[-1] - centerline[-2]
            if centerline.shape[0] > 2:
                tangents[1:-1] = centerline[2:] - centerline[:-2]

        points = []
        last_normal = None
        for tangent in tangents:
            norm = np.linalg.norm(tangent)
            if norm < 1e-9:
                tangent = np.array([1.0, 0.0, 0.0])
            else:
                tangent = tangent / norm

            if last_normal is None:
                ref = np.array([0.0, 0.0, 1.0])
                if abs(float(np.dot(ref, tangent))) > 0.95:
                    ref = np.array([0.0, 1.0, 0.0])
                normal = ref - np.dot(ref, tangent) * tangent
            else:
                normal = last_normal - np.dot(last_normal, tangent) * tangent
                if np.linalg.norm(normal) < 1e-9:
                    normal = np.cross(tangent, np.array([0.0, 0.0, 1.0]))
            normal_norm = np.linalg.norm(normal)
            if normal_norm < 1e-9:
                normal = np.array([0.0, 1.0, 0.0])
            else:
                normal = normal / normal_norm
            binormal = np.cross(tangent, normal)
            binormal /= max(np.linalg.norm(binormal), 1e-9)
            last_normal = normal

            for side in range(sides):
                angle = 2.0 * np.pi * side / sides
                points.append(radius * (np.cos(angle) * normal + np.sin(angle) * binormal))

        tube_points = np.repeat(centerline, sides, axis=0) + np.asarray(points)
        triangles: list[int] = []
        for i in range(centerline.shape[0] - 1):
            ring_a = i * sides
            ring_b = (i + 1) * sides
            for side in range(sides):
                a0 = ring_a + side
                a1 = ring_a + ((side + 1) % sides)
                b0 = ring_b + side
                b1 = ring_b + ((side + 1) % sides)
                triangles.extend([a0, b0, a1, a1, b0, b1])

        return tube_points, triangles

    def _create_tet_mesh_from_usd(self, usd_mesh: UsdGeom.Mesh):
        # Load precomputed mesh data from USD prim.
        mesh_prim = usd_mesh.GetPrim()
        tet_points = np.array(mesh_prim.GetAttribute("tet_points").Get())
        tet_indices = mesh_prim.GetAttribute("tet_indices").Get()
        surf_points = np.array(mesh_prim.GetAttribute("tet_surf_points").Get())
        tet_surf_indices = mesh_prim.GetAttribute("tet_surf_indices").Get()

        replace_color = False
        if tet_indices is None:
            mesh_gen = MeshGenerator(
                config=TetMeshCfg(
                    stop_quality=8,
                    max_its=100,
                    edge_length_r=1 / 5,
                    epsilon_r=0.001,
                )
            )
            tet_points, tet_indices, surf_points, tet_surf_indices = mesh_gen.generate_tet_mesh_for_prim(usd_mesh)
            replace_color = True

        # transform local tet points to world coor
        tf_world = omni.usd.get_world_transform_matrix(usd_mesh)

        tet_points_world = np.array(tf_world).T @ np.vstack((tet_points.T, np.ones(tet_points.shape[0])))
        tet_points_world = tet_points_world[:-1].T

        self.init_world_transform = torch.tensor(np.array(tf_world).T.copy(), device=self.uipc_sim.cfg.device)

        # uipc wants 2D array
        tet_indices = np.array(tet_indices).reshape(-1, 4)
        tet_surf_indices = np.array(tet_surf_indices).reshape(-1, 3)

        # create uipc mesh
        mesh = tetmesh(tet_points_world.copy(), tet_indices.copy())
        # enable the contact by labeling the surface
        label_surface(mesh)
        label_triangle_orient(mesh)
        # flip the triangles inward for better rendering
        mesh = flip_inward_triangles(mesh)  # todo idk if this makes a difference for us

        # libuipc uses different indexing for the surface topology
        surf = extract_surface(mesh)
        surf_points_world = surf.positions().view().reshape(-1, 3)
        surf_tri = surf.triangles().topo().view().reshape(-1).tolist()

        return mesh, surf_points_world, surf_tri, replace_color

    def _create_line_mesh_from_usd(self, usd_mesh: UsdGeom.Mesh):
        mesh_prim = usd_mesh.GetPrim()
        thread_points_attr = mesh_prim.GetAttribute("thread_points")
        if thread_points_attr.IsValid() and thread_points_attr.Get() is not None:
            local_points = np.array(thread_points_attr.Get(), dtype=np.float64)
        else:
            local_points = np.array(usd_mesh.GetPointsAttr().Get(), dtype=np.float64)

        if local_points.ndim != 2 or local_points.shape[1] != 3 or local_points.shape[0] < 2:
            raise ValueError(f"Thread mesh {usd_mesh.GetPath()} needs at least two 3D points.")

        thread_edges_attr = mesh_prim.GetAttribute("thread_edges")
        if thread_edges_attr.IsValid() and thread_edges_attr.Get() is not None:
            edges = np.array(thread_edges_attr.Get(), dtype=np.int32).reshape(-1, 2)
        else:
            edges = np.column_stack(
                [
                    np.arange(local_points.shape[0] - 1, dtype=np.int32),
                    np.arange(1, local_points.shape[0], dtype=np.int32),
                ]
            )

        tf_world = omni.usd.get_world_transform_matrix(usd_mesh)
        points_world = np.array(tf_world).T @ np.vstack((local_points.T, np.ones(local_points.shape[0])))
        points_world = points_world[:-1].T

        self.init_world_transform = torch.tensor(np.array(tf_world).T.copy(), device=self.uipc_sim.cfg.device)
        self._line_edges = edges.copy()

        mesh = linemesh(points_world.copy(), edges.copy())
        label_surface(mesh)

        tube_points, tube_tri = self._make_tube_mesh(
            points_world,
            radius=self.cfg.constitution_cfg.render_radius,
            sides=self.cfg.constitution_cfg.render_sides,
        )
        return mesh, tube_points, tube_tri

    def update_line_render_mesh(self):
        if self._line_render_fabric_prim is None or len(self.geo_slot_list) == 0:
            return
        points = self.vertex_positions
        render_offset = np.array(getattr(self.cfg.constitution_cfg, "render_offset", (0.0, 0.0, 0.0)))
        tube_points, _ = self._make_tube_mesh(
            points + render_offset,
            radius=self.cfg.constitution_cfg.render_radius,
            sides=self.cfg.constitution_cfg.render_sides,
        )
        fabric_mesh_points_attr = self._line_render_fabric_prim.GetAttribute("points")
        fabric_mesh_points_attr.Set(usdrt.Vt.Vec3fArray(tube_points))

    def _create_cloth_mesh_from_usd(self, usd_mesh: UsdGeom.Mesh):
        local_points = np.array(usd_mesh.GetPointsAttr().Get(), dtype=np.float64)
        face_vertex_counts = np.array(usd_mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        face_vertex_indices = np.array(usd_mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)

        triangles = []
        index_offset = 0
        for face_count in face_vertex_counts:
            face = face_vertex_indices[index_offset : index_offset + face_count]
            index_offset += face_count
            if face_count == 3:
                triangles.append(face)
            elif face_count == 4:
                triangles.append(face[[0, 1, 2]])
                triangles.append(face[[0, 2, 3]])
            else:
                raise ValueError(
                    f"Cloth mesh {usd_mesh.GetPath()} only supports triangle or quad faces, got {face_count}."
                )

        tf_world = omni.usd.get_world_transform_matrix(usd_mesh)
        points_world = np.array(tf_world).T @ np.vstack((local_points.T, np.ones(local_points.shape[0])))
        points_world = points_world[:-1].T

        self.init_world_transform = torch.tensor(np.array(tf_world).T.copy(), device=self.uipc_sim.cfg.device)

        triangles = np.array(triangles, dtype=np.int32).reshape(-1, 3)
        mesh = trimesh(points_world.copy(), triangles.copy())
        label_surface(mesh)

        surf_points_world = mesh.positions().view().reshape(-1, 3)
        surf_tri = mesh.triangles().topo().view().reshape(-1).tolist()

        return mesh, surf_points_world, surf_tri

    def _initialize_impl(self):
        # create objects in the uipc scene for the meshes
        mesh = self.uipc_meshes[0]

        obj = self._uipc_sim.scene.objects().create(self.cfg.prim_path)
        self.uipc_scene_objects.append(obj)

        obj_geo_slot, _ = obj.geometries().create(mesh)
        self.obj_id = obj_geo_slot.id()
        omni.log.info(f"obj id of {self.cfg.prim_path}: {self.obj_id} ")
        self.geo_slot_list.append(obj_geo_slot)

        # save initial world vertex positions
        geom = self._uipc_sim.scene.geometries()
        geo_slot, geo_slot_rest = geom.find(self.obj_id)
        self.init_vertex_pos = torch.tensor(
            geo_slot.geometry().positions().view().copy().reshape(-1, 3), device=self.device
        )

        # log information the uipc body
        omni.log.info(f"UIPC body initialized at: {self.cfg.prim_path}.")
        omni.log.info(f"Number of instances: {self.num_instances}")

        # create buffers

        # container for data access
        if type(self.constitution) is StableNeoHookean:
            self._data = UipcObjectDeformableData(self._uipc_sim, self, self.device)
        elif type(self.constitution) is AffineBodyConstitution:
            self._data = UipcObjectRigidData(self._uipc_sim, self, self.device)
        elif type(self.constitution) is NeoHookeanShell:
            self._data = UipcObjectDeformableData(self._uipc_sim, self, self.device)
        elif type(self.constitution) is HookeanSpring:
            self._data = UipcObjectDeformableData(self._uipc_sim, self, self.device)

        self._create_buffers()
        # process configuration
        self._process_cfg()
        # update the uipc_object data
        self.update(0.0)

        # add this object to the list of all uipc objects in the world
        self._uipc_sim.uipc_objects.append(self)

    def _create_buffers(self):
        """Create buffers for storing data."""
        # constants
        self._ALL_INDICES = torch.arange(self.num_instances, dtype=torch.long, device=self.device)

        # self._data._nodal_pos_w = torch.zeros(self.num_instances, self._vertex_count)

    #     # set information about rigid body into data
    #     self._data.body_names = self.body_names
    #     self._data.default_mass = self.root_physx_view.get_masses().clone()
    #     self._data.default_inertia = self.root_physx_view.get_inertias().clone()

    def _process_cfg(self):
        """Post processing of configuration parameters."""
        # default state
        # -- root state
        # note: we cast to tuple to avoid torch/numpy type mismatch.
        default_root_state = (
            tuple(self.cfg.init_state.pos)
            + tuple(self.cfg.init_state.rot)
            # + tuple(self.cfg.init_state.lin_vel)
            # + tuple(self.cfg.init_state.ang_vel)
        )
        default_root_state = torch.tensor(default_root_state, dtype=torch.float, device=self.device)
        # self._data.default_root_state = default_root_state.repeat(self.num_instances, 1)

    def _create_constitutions(self, mesh):
        # create constitutions
        constitution_types = {
            UipcObjectCfg.AffineBodyConstitutionCfg: AffineBodyConstitution,
            UipcObjectCfg.StableNeoHookeanCfg: StableNeoHookean,
            UipcObjectCfg.NeoHookeanShellCfg: NeoHookeanShell,
            UipcObjectCfg.HookeanSpringCfg: HookeanSpring,
        }
        self.constitution = constitution_types[type(self.cfg.constitution_cfg)]()

        if type(self.constitution) is StableNeoHookean:
            youngs = self.cfg.constitution_cfg.youngs_modulus
            poisson = self.cfg.constitution_cfg.poisson_rate
            moduli = ElasticModuli.youngs_poisson(youngs * MPa, poisson)
            # apply the constitution and contact model to the base mesh
            self.constitution.apply_to(mesh, moduli, mass_density=self.cfg.mass_density)
            # needed for writing vertex position to sim
            self._system_name = "uipc::backend::cuda::FiniteElementMethod"
        elif type(self.constitution) is AffineBodyConstitution:
            stiffness = self.cfg.constitution_cfg.m_kappa
            self.constitution.apply_to(mesh, stiffness * MPa, mass_density=self.cfg.mass_density)
            self._system_name = "uipc::backend::cuda::AffineBodyDynamics"

            # make ABD body kinematic
            if self.cfg.constitution_cfg.kinematic:
                is_fixed_attr = mesh.instances().find(builtin.is_fixed)
                view(is_fixed_attr)[0] = 1
        elif type(self.constitution) is NeoHookeanShell:
            youngs = self.cfg.constitution_cfg.youngs_modulus
            poisson = self.cfg.constitution_cfg.poisson_rate
            moduli = ElasticModuli.youngs_poisson(youngs * MPa, poisson)
            self.constitution.apply_to(
                mesh,
                moduli=moduli,
                mass_density=self.cfg.mass_density,
                thickness=self.cfg.constitution_cfg.thickness,
            )
            if self.cfg.constitution_cfg.enable_bending:
                self.bending_constitution = DiscreteShellBending()
                self.bending_constitution.apply_to(mesh, E=self.cfg.constitution_cfg.bending_stiffness)
            self._system_name = "uipc::backend::cuda::FiniteElementMethod"
        elif type(self.constitution) is HookeanSpring:
            self.constitution.apply_to(
                mesh,
                self.cfg.constitution_cfg.kappa,
                mass_density=self.cfg.mass_density,
                thickness=self.cfg.constitution_cfg.thickness,
            )
            if self.cfg.constitution_cfg.enable_bending:
                self.bending_constitution = KirchhoffRodBending()
                self.bending_constitution.apply_to(mesh, E=self.cfg.constitution_cfg.bending_stiffness)
            self._system_name = "uipc::backend::cuda::FiniteElementMethod"

        # apply the default contact model to the base mesh
        default_element = self._uipc_sim.scene.contact_tabular().default_element()
        default_element.apply_to(mesh)

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._physics_sim_view = None
        self._root_physx_view = None
