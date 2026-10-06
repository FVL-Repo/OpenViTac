"""Reach and hold two physical strap tensions using tactile feedback."""
from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from uipc import view
from ._base_task import *
from ._force_task_utils import (
    ElasticResultant,
    ForceTaskScene,
    actor_contact_resultant,
    image_features,
    read_rgb,
    sample_physics,
    task_parameters,
    tetra_faces,
    write_tet_asset,
)


# Force task probe

def strap_ladder(task, pose):
    levels=list(task.params.get('probe_force_levels_N',[14.,18.,22.,18.]))
    if not levels or any(not 2.<=float(v)<=28. for v in levels):
        raise ValueError("Force diagnostic levels must be 2--28 N")
    task.active=True
    origin=pose.p.copy();position=origin.copy()
    task.metadata['diagnostic_protocol']={'type':'force ladder','levels_N':levels,
        'hold_s':1.,'control':'privileged physical force; images recorded independently'}
    task.failure=None
    completed=[]
    for level_index, level in enumerate(levels):
        stable=0
        task.probe_level_index=level_index
        task.probe_target_N=float(level)
        task.phase=f'diagnostic {level:g} N'
        for tick in range(1200):
            if tick%4==0:
                row=task._record()
                error=float(level)-row['tension_N']
                slip=np.linalg.norm(np.asarray(row['tab_in_hand_m'])-task.initial_tab_in_hand)
                if row['tension_N']>48 or row['maximum_stretch']>1.6:
                    task.failure='probe_overload';break
                if slip>.012:
                    task.failure='probe_grasp_lost';break
                if position[2]-origin[2]>.055:
                    task.failure='probe_travel_limit';break
                stable=stable+4 if abs(error)<.25 else 0
                if stable>=120:
                    completed.append({'level_N':level,'step':task.step_count,'force_N':row['tension_N']})
                    break
                velocity=float(np.clip(error/1400.,-.002,.002))
                if abs(error)<.08:velocity=0.
            position[2]+=velocity*task.cfg.sim.dt
            position[2]=min(position[2],task._robot_manager.get_ee_pose().p[2]+.001)
            task._servo(position,pose.q)
            if not task.plan_success:break
        if stable<120 or task.failure or not task.plan_success:
            task.failure=task.failure or 'probe_level_timeout';break
    task.active=False
    task._record()
    task.metadata['diagnostic_completed']=len(completed)==len(levels)
    task.metadata['diagnostic_levels']=completed
    task.metadata['diagnostic_only']=True
    task.metadata['strap_failure']=task.failure
    task.metadata['strap_final']=task.measured


# Strap lifecycle

@dataclass(frozen=True)
class StrapAcceptanceLimits:
    targets_N: tuple = (12.0, 18.0)
    tolerance_N: float = 0.5
    hold_seconds: float = 3.0
    timeout_seconds: float = 60.0
    physics_dt: float = 1.0 / 120.0

class StrapLifecycle:
    def __init__(self, start_step, limits=None):
        self.limits = limits or StrapAcceptanceLimits()
        self.start_step = int(start_step)
        self.last_step = self.start_step
        self.stage_index = 0
        self.stable_since = None
        self.completed_steps = []
        self.failure = ''
        self.success = False

    def advance(self, step, tension_N, *, grasped, damage=''):
        step = int(step)
        if step < self.last_step:
            raise ValueError('Physical time moved backwards')
        if step == self.last_step:
            return
        consecutive = step == self.last_step + 1
        self.last_step = step
        if self.failure:
            return
        if damage:
            self.failure = str(damage)
            self.success = False
            return
        if not math.isfinite(float(tension_N)):
            self.failure = 'invalid_tension'
            return
        if self.success:
            return
        if not consecutive:
            self.stable_since = None
        target = self.limits.targets_N[self.stage_index]
        if grasped and abs(float(tension_N) - target) <= self.limits.tolerance_N:
            if self.stable_since is None:
                self.stable_since = step
            if (step - self.stable_since) * self.limits.physics_dt >= self.limits.hold_seconds - 1e-12:
                self.completed_steps.append(step)
                self.stage_index += 1
                self.stable_since = None
                self.success = self.stage_index == len(self.limits.targets_N)
        else:
            self.stable_since = None
        if not self.success and (step-self.start_step)*self.limits.physics_dt >= self.limits.timeout_seconds-1e-12:
            self.failure = 'timeout'

    def snapshot(self):
        return {
            'contract': 'strap_two_stage_v1',
            'target_sequence_N': list(self.limits.targets_N),
            'tolerance_N': self.limits.tolerance_N,
            'hold_seconds_per_stage': self.limits.hold_seconds,
            'timeout_seconds': self.limits.timeout_seconds,
            'physics_dt_s': self.limits.physics_dt,
            'stage_index': self.stage_index,
            'completed_steps': self.completed_steps.copy(),
            'current_continuous_hold_seconds': (
                0.0 if self.stable_since is None else
                (self.last_step-self.stable_since)*self.limits.physics_dt),
            'policy_elapsed_seconds': (self.last_step-self.start_step)*self.limits.physics_dt,
            'success': self.success,
            'failure': self.failure,
        }


# Tension strap

class StrapDamageStop(RuntimeError):
    pass


# Tension strap fracture

def capture_state(simulation):
    return [(np.asarray(b.geo_slot_list[0].geometry().positions().view()).copy(),
             np.asarray(b.geo_slot_list[0].geometry().transforms().view()).copy())
            for b in simulation.uipc_objects]

def _install_eroded_topology(geometry, template, keep, rest_volumes):
    from uipc import view
    # Copy every per-element field before compaction (including constitutive
    # fields); replace topology and surface labels from a fresh closure.
    fields = {}
    for name in geometry.tetrahedra().to_json():
        slot = geometry.tetrahedra().find(name)
        fields[name] = np.asarray(slot.view()).copy()[keep]
    geometry.tetrahedra().resize(int(keep.sum()))
    for name, values in fields.items():
        view(geometry.tetrahedra().find(name))[:] = values
    for method in ['edges', 'triangles']:
        dst, src = getattr(geometry, method)(), getattr(template, method)()
        dst.resize(src.size())
        view(dst.topo())[:] = np.asarray(src.topo().view())
    for method in ['vertices', 'edges', 'triangles', 'tetrahedra']:
        dst, src = getattr(geometry, method)(), getattr(template, method)()
        for name in ['is_surf', 'parent_id', 'orient']:
            a, b = dst.find(name), src.find(name)
            if b is not None:
                if a is None:
                    a = dst.create(name, 0)
                view(a)[:] = np.asarray(b.view())
    view(geometry.vertices().find('volume'))[:] = rest_volumes

def connected_component(tets, start, vertex_count):
    neighbours = [set() for _ in range(vertex_count)]
    for tet in tets:
        for v in tet:
            neighbours[v].update(tet)
    reached, pending = {int(start)}, [int(start)]
    while pending:
        new = neighbours[pending.pop()] - reached
        reached.update(new)
        pending.extend(new)
    mask = np.zeros(vertex_count, dtype=bool)
    mask[list(reached)] = True
    return mask

def _notch_cut_lips(rest, tets, initial_keep):
    """Chip the cut lips while retaining a closed, two-component FEM solid."""
    from ._force_task_utils import tetra_faces
    keep=initial_keep.copy()
    centers=rest[tets].mean(axis=1)
    candidates=[]
    for x,z in [(-.025,.060),(0.,.070),(.025,.065)]:
        for sign,lo,hi in [(-1,z-.005,z),(1,z+.005,z+.010)]:
            selected=((np.abs(centers[:,0]-x)<.01001)
                      &(centers[:,2]>lo+1e-7)&(centers[:,2]<hi-1e-7)
                      &((centers[:,0]-x)*(-sign)>.003)&keep)
            ids=np.flatnonzero(selected).tolist()
            ids.sort(key=lambda i:(-(centers[i,2] if sign<0 else -centers[i,2]),
                                   -abs(centers[i,0]-x)))
            candidates.extend(ids)
    for index in candidates:
        trial=keep.copy();trial[index]=False
        if len(np.unique(tets[trial]))!=len(rest):
            continue
        faces=tetra_faces(tets[trial])
        edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
        _,counts=np.unique(edges,axis=0,return_counts=True)
        if not np.all(counts==2):
            continue
        lower=connected_component(tets[trial],np.flatnonzero(rest[:,2]<=.01401)[0],len(rest))
        upper=connected_component(tets[trial],np.flatnonzero(rest[:,2]>=.154)[0],len(rest))
        if np.any(lower&upper) or not np.all(lower|upper):
            continue
        keep=trial
    if keep.sum()==initial_keep.sum():
        raise ValueError("Unable to form valid notched cut faces")
    return keep

def rupture_elements(rest, tets, profile="flat"):
    """Remove a 5 mm cell row in each band, keeping all existing vertices."""
    rest, tets = np.asarray(rest), np.asarray(tets)
    centers = rest[tets].mean(axis=1)
    remove = np.zeros(len(tets), dtype=bool)
    for x, z in [(-.025, .060), (0., .070), (.025, .065)]:
        remove |= ((np.abs(centers[:, 0]-x) < .01001)
                   & (centers[:, 2] > z+1e-7)
                   & (centers[:, 2] < z+.005-1e-7))
    keep = ~remove
    if not remove.any() or len(np.unique(tets[keep])) != len(rest):
        raise ValueError("Rupture requires the canonical three-band volume mesh")
    if profile == "notched":
        keep = _notch_cut_lips(rest, tets, keep)
    elif profile != "flat":
        raise ValueError("Unknown strap rupture profile")
    return keep

def rupture(task):
    """Sever all three load paths once and rebuild UIPC from the current state."""
    from uipc import builtin, view
    from uipc.core import Engine, World
    from uipc.geometry import (tetmesh, label_surface, label_triangle_orient,
                               flip_inward_triangles, extract_surface)
    from uipc.constitution import StableNeoHookean, ElasticModuli
    from pxr import UsdGeom, Vt
    import omni.usd
    import usdrt
    from ._force_task_utils import ElasticResultant

    if getattr(task, 'ruptured', False):
        return task.metadata['strap_rupture']
    sim, body = task.uipc_sim, task.strap
    current = capture_state(sim)
    previous = getattr(task, '_rupture_previous_state', current)
    keep = rupture_elements(task.mesh_points, task.tets, task.params.get("rupture_profile","flat"))
    new_tets = task.tets[keep]
    template = tetmesh(task.rest.copy(), new_tets.copy())
    label_surface(template)
    label_triangle_orient(template)
    template = flip_inward_triangles(template)
    StableNeoHookean().apply_to(template, ElasticModuli.youngs_poisson(1e6, .4), 1050.)
    volumes = np.asarray(template.vertices().find('volume').view()).copy()
    removed_mass = float(task.resultant.volume[~keep].sum()*1050.)
    for slot in sim.scene.geometries().find(body.geo_slot_list[0].id()):
        _install_eroded_topology(slot.geometry(), template, keep, volumes)
    # Keep per-body motion when the backend is rebuilt. No imposed recoil kick.
    temporary = []
    for i, obj in enumerate(sim.uipc_objects):
        geo = obj.geo_slot_list[0].geometry()
        affine = geo.meta().find(builtin.backend_abd_body_offset) is not None
        collection = geo.instances() if affine else geo.vertices()
        motion = 1 if affine else 0
        velocity = (current[i][motion]-previous[i][motion])/task.cfg.sim.dt
        slot = collection.find('velocity')
        original = np.asarray(slot.view()).copy() if slot is not None else None
        if slot is None:
            slot = collection.create('velocity', np.zeros((4,4)) if affine else np.zeros((3,1)))
        view(slot)[:] = velocity
        temporary.append((collection, slot, original))
    engine = Engine(sim.cfg.device, sim.cfg.workspace)
    world = World(engine)
    world.init(sim.scene)
    if not world.is_valid():
        raise RuntimeError("strap rupture rejected: rebuilt collision world is invalid")
    world.retrieve()
    sim.world, sim.engine = world, engine
    sim._contact_grad_cache = None
    for collection, slot, original in temporary:
        if original is None:
            collection.destroy('velocity')
        else:
            view(slot)[:] = original
    offsets = [0]
    preservation = []
    for i, obj in enumerate(sim.uipc_objects):
        geo = obj.geo_slot_list[0].geometry()
        offsets.append(int(geo.meta().find(builtin.global_vertex_offset).view()[0]))
        obj.global_system_id = len(offsets)-1
        obj._data.update(task.cfg.sim.dt)
        preservation.append(float(np.max(np.abs(np.asarray(geo.positions().view())-current[i][0]))))
    sim._system_vertex_offsets['uipc::backend::cuda::GlobalVertexManager'] = offsets

    # New cut faces must enter BOTH contact geometry and the rendered surface.
    stage = omni.usd.get_context().get_stage()
    surface_offsets = [0]
    for obj in sim.uipc_objects:
        if getattr(obj, '_is_line_mesh', False):
            continue
        surface = extract_surface(obj.geo_slot_list[0].geometry())
        count = len(surface.positions().view())
        obj._surf_vertex_offset_start = surface_offsets[-1]
        obj._surf_vertex_offset_end = surface_offsets[-1]+count
        surface_offsets.append(surface_offsets[-1]+count)
        if obj is body:
            xyz = np.asarray(surface.positions().view()).reshape(-1,3)
            faces = np.asarray(surface.triangles().topo().view()).reshape(-1,3)
            mesh = UsdGeom.Mesh(stage.GetPrimAtPath(str(obj.fabric_prim.GetPath())))
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(xyz.astype(np.float32)))
            mesh.GetFaceVertexCountsAttr().Set([3]*len(faces))
            mesh.GetFaceVertexIndicesAttr().Set(faces.ravel().tolist())
            mesh.GetNormalsAttr().Set([])
            obj.fabric_prim.GetAttribute('points').Set(usdrt.Vt.Vec3fArray(xyz))
            # Fabric also stores topology once the mesh is dynamic.
            for name, values in [('faceVertexCounts', [3]*len(faces)),
                                 ('faceVertexIndices', faces.ravel().tolist())]:
                attr = obj.fabric_prim.GetAttribute(name)
                if attr:
                    attr.Set(usdrt.Vt.IntArray(values))
            rest_surface = extract_surface(template)
            task.presentation.color_strap(mesh,
                np.asarray(rest_surface.positions().view()).reshape(-1,3))
            # Keep the actual broken FEM surface and its changing normals in
            # USD, as with the verified vase pad renderer. The old Fabric mesh
            # otherwise retains stale/empty normals after a topology change.
            from scipy.spatial import cKDTree
            from pxr import UsdShade
            distance, indices = cKDTree(body.vertex_positions).query(xyz)
            if distance.max()>1e-6 or len(np.unique(indices))!=len(indices):
                raise RuntimeError("broken strap render surface must map uniquely to FEM vertices")
            UsdGeom.Imageable(mesh.GetPrim()).MakeInvisible()
            root=task.scene.env_prim_paths[0]+'/force_task_presentation'
            live=UsdGeom.Mesh.Define(stage,root+'/broken_strap')
            live.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(xyz.astype(np.float32)))
            live.CreateFaceVertexCountsAttr([3]*len(faces))
            live.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
            live.CreateSubdivisionSchemeAttr('none')
            live.CreateDoubleSidedAttr(True)
            live.SetNormalsInterpolation('vertex')
            material=UsdShade.Material(stage.GetPrimAtPath(root+'/Looks/Elastomer'))
            UsdShade.MaterialBindingAPI.Apply(live.GetPrim()).Bind(material)
            # Explicit per-face material bindings survive runtime topology
            # replacement in RTX; a dynamic displayColor reader previously
            # rendered most of the new surface black after the handoff.
            local_rest=task.mesh_points[indices]
            centers=local_rest[faces].mean(axis=1)
            band=(centers[:,2]>.01601)&(centers[:,2]<.13199)
            which=np.full(len(faces),3,dtype=int)
            band_centers=np.asarray(task.params.get('band_centers_m',[-.025,0.,.025]))
            which[band]=np.abs(centers[band,0,None]-band_centers[None,:]).argmin(axis=1)
            palette=np.asarray([[50,126,207],[253,230,64],[253,77,54],[55,58,62]],float)/255.
            palette=np.where(palette<=.04045,palette/12.92,((palette+.055)/1.055)**2.4)
            from ._force_task_utils import _material
            binding=UsdShade.MaterialBindingAPI.Apply(live.GetPrim())
            color_counts=[]
            for color_index,color in enumerate(palette):
                face_ids=np.flatnonzero(which==color_index).astype(np.int32)
                color_counts.append(int(len(face_ids)))
                if not len(face_ids):
                    raise RuntimeError('Broken strap lost a required color region')
                mat=_material(stage,root+f'/Looks/BrokenStrap{color_index}',tuple(color),.78)
                subset=binding.CreateMaterialBindSubset(f'color_{color_index}',Vt.IntArray.FromNumpy(face_ids))
                UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(mat)
            task.metadata['strap_broken_surface_colors']={
                'method':'constant-color material subsets by rest-mesh face location',
                'faces_per_color_blue_yellow_red_tab':color_counts,
                'rest_local_bounds_m':[local_rest.min(0).tolist(),local_rest.max(0).tolist()]}
            np.savez_compressed(task.work/'broken_surface_mapping.npz',
                live_points=xyz.copy(),faces=faces.copy(),volume_indices=indices.copy(),
                local_rest=local_rest.copy(),face_color_indices=which.copy())
            task.broken_strap_surface=(live,indices.copy(),faces.copy())
    sim._surf_vertex_offsets = surface_offsets
    mu, lame = task.resultant.mu[keep], task.resultant.lame[keep]
    task.tets = new_tets
    task.resultant = ElasticResultant(task.rest, new_tets, mu, lame)
    lower = connected_component(new_tets, np.flatnonzero(task.anchor_mask)[0], len(task.rest))
    if np.any(lower & task.tab_mask):
        raise RuntimeError("Rupture did not disconnect the grip from the anchor")
    task.ruptured = True
    task.phase = 'rupture'
    event = dict(step=task.step_count, trigger_tension_N=task.measured['tension_N'],
                 trigger_maximum_stretch=task.measured['maximum_stretch'],
                 threshold_N=float(task.params.get('overload_N',48.)),
                 model='FEM cell erosion with passive elastic recoil',
                 rupture_profile=task.params.get('rupture_profile','flat'),
                 removed_tets=int((~keep).sum()), remaining_tets=len(new_tets),
                 removed_mass_kg=removed_mass, anchor_grip_connected=False,
                 max_position_change_on_rebuild_m=max(preservation),
                 imposed_recoil_velocity_m_s=0.)
    task.metadata['strap_rupture'] = event
    sim.update_render_meshes()
    return event


# Tension strap geometry

def material_fields(rest, tets, band_modulus_mpa):
    z = np.asarray(rest)[tets, 2].mean(axis=1)
    young = np.where((z >= .025) & (z <= .130), band_modulus_mpa, 15.) * 1e6
    poisson = .40
    mu = young / (2 * (1 + poisson))
    lame = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    return mu, lame

def strap_mesh():
    zs = np.unique(np.r_[0., .008, .014, np.linspace(.020, .130, 23),
                         .138, .146, .154, .162, .170, .178, .186, .194])
    nx, ny = 5, 5
    vertices = []
    for z in zs:
        if z <= .014:
            width, depth = .048, .036
        elif z < .025:
            u = (z - .014) / .011
            width, depth = .048 + u * (.030 - .048), .036 + u * (.004 - .036)
        elif z <= .130:
            width, depth = .030, .004
        elif z < .154:
            u = (z - .130) / .024
            width, depth = .030 + u * (.040 - .030), .004 + u * (.040 - .004)
        else:
            width, depth = .040, .040
        for x in np.linspace(-width / 2, width / 2, nx):
            for y in np.linspace(-depth / 2, depth / 2, ny):
                vertices.append([x, y, z])
    pts = np.asarray(vertices)
    tet = []
    def idx(k, x, y): return k * nx * ny + x * ny + y
    for k in range(len(zs) - 1):
        for x in range(nx - 1):
            for y in range(ny - 1):
                a, b, c, d = [idx(k, x + dx, y + dy) for dx, dy in [(0,0),(1,0),(0,1),(1,1)]]
                e, f, g, h = [idx(k+1, x + dx, y + dy) for dx, dy in [(0,0),(1,0),(0,1),(1,1)]]
                tet.extend([(a,b,d,h),(a,d,c,h),(a,c,g,h),(a,g,e,h),(a,e,f,h),(a,f,b,h)])
    tet = np.asarray(tet, dtype=np.int32)
    matrices = np.stack([pts[tet[:, i]] - pts[tet[:, 0]] for i in (1,2,3)], axis=-1)
    negative = np.linalg.det(matrices) < 0
    tet[negative, 1], tet[negative, 2] = tet[negative, 2].copy(), tet[negative, 1].copy()
    return pts, tet, tetra_faces(tet)


# Tension strap

class Task(BaseTask):
    def __init__(self, cfg, **kwargs):
        self.params = task_parameters(cfg)
        self.params.setdefault('public_tactile_grasp',True)
        self.params.setdefault('controller_deadband_N',.02)
        if self.params.get('mesh_path'):
            mesh_path=Path(self.params['mesh_path'])
            if not mesh_path.is_absolute():
                mesh_path=Path(__file__).resolve().parents[1]/mesh_path
            self.params['mesh_path']=str(mesh_path.resolve())
        for key,value in sample_physics('tension_strap',self.params.get('physics_seed',0)).items():
            self.params.setdefault(key,value)
        self.ruptured = False
        self._monitor_strap=False;self._expert_active=False;self._grip_released=False
        self._accepted_result=None
        self._submission_requested=False;self._submission_anchor=None;self._submission_hold_steps=0
        self.active = False
        self.trace = []
        self.measured = {}
        self.phase = 'setup'
        self.reference = None
        self.held_steps = 0
        self.failure = None
        self.modulus = float(self.params.get('modulus_mpa', .25))
        self.target = 12.0
        self.tolerance = 0.5
        self.target_sequence = (12.0, 18.0)
        self._stage_scorer = None
        self._measure_step = -1
        self.controller = self.params.get('controller', 'diagnostic_force')
        self.work = Path(self.params['workspace'])
        self.work.mkdir(parents=True, exist_ok=True)
        eye=np.array([.75,.23,.19]); aim=np.array([.55,0.,.11])
        z=(eye-aim)/np.linalg.norm(eye-aim)
        x=np.cross([0.,0.,1.],z); x/=np.linalg.norm(x)
        y=np.cross(z,x)
        q=t3d.quaternions.mat2quat(np.stack([x,y,z],axis=1))
        cfg.cameras[0]=CameraCfg(name='head',prim_path='/World/envs/env_.*/Camera',
            offset=CameraCfg.OffsetCfg(pos=tuple(eye),rot=tuple(q),convention='opengl'),
            data_types=['rgb','depth'],
            spawn=sim_utils.PinholeCameraCfg(focal_length=1.6,focus_distance=1.,
                horizontal_aperture=2.4,clipping_range=(.02,100.)),
            width=480,height=270,update_period=1/120)
        cfg.uipc_sim.contact.eps_velocity=.001
        if 'newton_velocity_tol_m_s' in self.params:
            cfg.uipc_sim.newton.velocity_tol=float(self.params['newton_velocity_tol_m_s'])
        if 'newton_max_iter' in self.params:
            cfg.uipc_sim.newton.max_iter=int(self.params['newton_max_iter'])
        if 'contact_d_hat_m' in self.params:
            cfg.uipc_sim.contact.d_hat=float(self.params['contact_d_hat_m'])
        from ._force_task_utils import configure_final_task
        configure_final_task(cfg, self.params, max_policy_seconds=60)
        super().__init__(cfg, **kwargs)
        self.video_handler.fps = 120 / cfg.video_frequency
        self.video_handler.encoder_threads = 2

    def _setup_scene(self):
        from ._force_task_utils import configure_gel
        configure_gel(self.cfg.robot.tactiles,self.params)
        super()._setup_scene()
        self.presentation=ForceTaskScene(self,'strap')
        contacts=self.uipc_sim.scene.contact_tabular()
        tab=contacts.create('elastic_strap_grip')
        contacts.insert(tab,contacts.default_element(),friction_rate=2.5,
                        resistance=self.cfg.uipc_sim.contact.default_contact_resistance*1e9)
        if self.params.get('strap_self_contact', True) is False:
            contacts.insert(tab,tab,friction_rate=2.5,
                            resistance=self.cfg.uipc_sim.contact.default_contact_resistance*1e9,
                            enable=False)
        for mesh in self.strap.uipc_meshes:
            tab.apply_to(mesh)

    def create_actors(self):
        if self.params.get('mesh_path'):
            if self.controller == 'marker_rgb':
                import hashlib, json
                model=json.loads(Path(self.params['calibration']).read_text())
                expected={
                    'geometry_sha256':hashlib.sha256(Path(self.params['mesh_path']).read_bytes()).hexdigest(),
                    'grasp_depth_mm':float(self.params.get('grasp_depth_mm',27.5)),
                    'strap_self_contact':bool(self.params.get('strap_self_contact',True)),
                    'contact_d_hat_m':float(self.params.get('contact_d_hat_m',.001)),
                    'newton_velocity_tol_m_s':float(self.params.get('newton_velocity_tol_m_s',.05)),
                }
                expected['gel_modulus_mpa']=float(self.params.get('gel_modulus_mpa',.1))
                contract={'gel_modulus_mpa':.1,'newton_velocity_tol_m_s':.05,**model.get('physical_contract',{})}
                if any(contract.get(k)!=v for k,v in expected.items()):
                    raise ValueError('Marker calibration does not match the strap geometry/contact contract')
            mesh_data = np.load(self.params['mesh_path'], allow_pickle=False)
            points = np.asarray(mesh_data['points'], dtype=np.float64)
            tets = np.asarray(mesh_data['tets'], dtype=np.int32)
            faces = tetra_faces(tets)
        else:
            points, tets, faces = strap_mesh()
        self.mesh_points, self.tets = points, tets
        self.base = np.array([.55, 0., .005])
        path = write_tet_asset(self.work / 'elastic_strap.usda', points, tets, faces)
        self.strap = self._actor_manager.add_from_usd_file(
            name='elastic_strap', asset_path=path,
            pose=Pose(self.base, [1.,0.,0.,0.]),
            constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(
                youngs_modulus=self.modulus, poisson_rate=.4),
            density=1050., show_physics_mesh=True, keep_constrained=True)
        # Material fields are installed BEFORE the UIPC world initializes.
        mu, lame = material_fields(points, tets, self.modulus)
        geo = self.strap.uipc_meshes[0]
        view(geo.tetrahedra().find('mu'))[:] = mu
        view(geo.tetrahedra().find('lambda'))[:] = lame
        self.resultant = ElasticResultant(points+self.base, tets, mu, lame)
        self.anchor_mask = points[:,2] <= .01401
        self.upper_mask = points[:,2] >= .080
        self.tab_mask = points[:,2] >= .154
        self.rest = points + self.base

    def reset(self,*args,**kwargs):
        self._stage_scorer=None
        self._accepted_result=None
        self._monitor_strap=False
        result=super().reset(*args,**kwargs)
        self._grip_released=False;self._accepted_result=None
        self._submission_requested=False;self._submission_anchor=None;self._submission_hold_steps=0
        self._stage_scorer=StrapLifecycle(self.policy_start_step)
        self._measure_step=-1
        self._monitor_strap=True
        self.metadata['strap_acceptance']=self._stage_scorer.snapshot()
        return result

    def _step(self,is_save=True):
        active=getattr(self,'_monitor_strap',False)
        if active and not self.ruptured and self.failure is None:
            self._rupture_previous_state=capture_state(self.uipc_sim)
        previous_step=self.step_count
        super()._step(is_save=is_save)
        if not active or self.step_count==previous_step:
            return
        row=self._measure()
        if not self._grip_released and self.step_count % 2 == 0:
            depth=self._tactile_manager.get_min_depth().detach().cpu().numpy().reshape(-1)
            # Release the initialization fixture once both physical pads
            # have acquired the tab. A 0.1 mm acquisition margin accommodates
            # discretized joint commands; it does not relax load success limits.
            release_depth=float(self.params.get('grasp_depth_mm',27.6))+.1
            if np.all(depth<=release_depth):
                self.strap.set_vertex_targets(self.rest,self.anchor_mask)
                self._actor_manager.update(dt=0.)
                self._grip_released=True
                self.initial_tab_in_hand=np.asarray(row['tab_in_hand_m'])
                self._pull_start_z=float(self._robot_manager.get_ee_pose().p[2])
                self.metadata['bilateral_grip_release_step']=self.step_count
                self.metadata['bilateral_grip_release_depth_mm']=depth.tolist()
        if self._grip_released and not self.failure and not self.ruptured:
            if row['tension_N']>float(self.params.get('overload_N',24.)) or row['maximum_stretch']>1.6:
                self.failure='overload_or_strain'
                if self.params.get('rupture_enabled',True) and self.params.get('geometry')=='bundle':
                    rupture(self)
                    self.failure='strap_ruptured'
            elif np.linalg.norm(np.asarray(row['tab_in_hand_m'])-self.initial_tab_in_hand)>.012:
                self.failure='grasp_lost'
            elif float(self._robot_manager.get_ee_pose().p[2])-self._pull_start_z>.055:
                self.failure='travel_limit'
        self._stage_scorer.advance(
            self.step_count, row['tension_N'], grasped=self._grip_released,
            damage=self.failure or '')
        if self._stage_scorer.failure and self.failure is None:
            self.failure=self._stage_scorer.failure
        self._submission_requested=self._stage_scorer.success
        self.metadata['strap_acceptance']=self._stage_scorer.snapshot()
        self.held_steps=int(round(self.metadata['strap_acceptance']['current_continuous_hold_seconds']/self.cfg.sim.dt))
        if self.failure and self._expert_active:
            raise StrapDamageStop(self.failure)

    def check_early_stop(self):
        return self.failure is not None

    def take_action(self,*args,**kwargs):
        if self._accepted_result is not None:
            return bool(self._accepted_result),bool(self._accepted_result)
        executed,success=super().take_action(*args,**kwargs)
        if success or self.failure:
            self._finish_strap_episode()
            return bool(executed and not self.failure),bool(self._accepted_result)
        return executed,success

    def _finish_strap_episode(self):
        if self._accepted_result is not None:return
        self._accepted_result=bool(self.check_success())
        if not self._accepted_result and self.failure is None:
            self.failure='incomplete_two_stage_sequence'
        self._monitor_strap=False;self.active=False
        self._set_phase(self.PHASE_TERMINAL,terminal_reason='success' if self._accepted_result else (self.failure or 'not_accepted'))
        manager=self._robot_manager
        joints=manager.robot.data.joint_pos[:,manager._arm_ids][0]
        manager.set_arm(joints,torch.zeros_like(joints),force=True)
        manager.set_gripper(float(manager.get_gripper_qpos()),force=True)
        self._record()
        self.metadata['strap_physical_verdict_step']=int(self.step_count)
        self.metadata['strap_final']=dict(self.measured)
        self.metadata['strap_final']['physical_acceptance']=self._stage_scorer.snapshot()
        self.metadata['strap_acceptance']=self._stage_scorer.snapshot()
        self.metadata['strap_failure']=self.failure
        self.metadata['strap_hold_steps']=self.held_steps
        self.metadata['strap_accepted']=self._accepted_result
        self.metadata['strap_trace_path']=str(self.work/'trace.jsonl')
        self.metadata['diagnostic_only']=self.controller!='marker_rgb' or bool(self.params.get('overload_demo')) or bool(self.params.get('strap_review_scenario'))
        self.metadata['success_scoring']='independent physical 12 N then 18 N, each within 0.5 N continuously 3 s; total 60 s'
        from ._force_task_utils import record_terminal_observation
        record_terminal_observation(self,'success' if self._accepted_result else (self.failure or 'incomplete'))
        if self.ruptured and self.plan_success:
            self.phase='rupture'
            for tick in range(180):
                self._step(is_save=True)
                if tick%2==1:self._record()
            self.metadata['strap_failure_presentation']=dict(
                final_step=int(self.step_count),final_tension_N=float(self.measured['tension_N']),
                phase='terminal',physical_verdict_frozen=True)


    def _reset_actors(self):
        if int(self.cfg.seed)!=int(self.params.get('physics_seed',0)):
            raise ValueError('Create one force-task process per seed so physical rest geometry/materials are sampled before UIPC initialization')
        self.active = False
        self.trace = []
        self.reference = None
        self.failure = None
        self.held_steps = 0
        self.target = 12.0
        self._measure_step = -1
        self.phase = 'setup'
        self.policy_hold = 0
        if self.ruptured:
            raise RuntimeError('Create a new task process after strap rupture')
        self.strap.set_vertex_targets(self.rest)
        self.metadata['strap_parameters'] = dict(self.params)
        self.metadata['control_semantics'] = self.controller
        self.metadata['backend_energy_volume_scale']=6.0
        self.metadata['material_randomization'] = 'same mesh; hidden band modulus per process'
        self.metadata['force_units'] = 'SI gradient of installed StableNeoHookean elastic energy'
        self.metadata['success_tension_bands_N'] = [[11.5,12.5],[17.5,18.5]]

    def _release_reset_constraints(self):
        pass

    def build_instruction(self):
        return 'Pull the elastic strap to12 newtons for3 seconds, then18 newtons for3 seconds; remain within0.5 newtons of each target.'

    def pre_move(self):
        self.move(self.atom.open_gripper(.85), tag='open_for_strap', delay=False)

    def _measure(self):
        if self._measure_step == self.step_count:
            return self.measured
        points = self.strap.vertex_positions.copy()
        m = self.resultant.measure(points, self.upper_mask)
        if self.params.get('mesh_path') and m['maximum_stretch'] > 1.6:
            np.savez_compressed(self.work/'excess_strain.npz',
                                points=points, rest=self.rest, tets=self.tets)
        if self.params.get('record_mesh') and (len(self.trace) % 12 == 0 or not self.active):
            folder=self.work/'mesh_states'
            folder.mkdir(exist_ok=True)
            np.savez_compressed(folder/f'{self.step_count:05d}.npz',points=points)
        tension = max(0., -float(m['elastic_force_N'][2]))
        tab = points[self.tab_mask].mean(axis=0)
        center = self._robot_manager.get_gripper_center_pose().p
        m.update(ruptured=self.ruptured, tension_N=tension, tab_z_m=float(tab[2]),
                 extension_m=float(tab[2]-self.rest[self.tab_mask,2].mean()),
                 tab_in_hand_m=(tab-center).tolist())
        m['elastic_force_N'] = m['elastic_force_N'].tolist()
        m['elastic_torque_Nm'] = m['elastic_torque_Nm'].tolist()
        # Raw UIPC gradients are kept separately until the dt^2 conversion is validated.
        for name,sensor in self._tactile_manager.tactiles.items():
            forces = sensor._get_contact_force().detach().cpu().numpy().reshape(-1,3)
            m[name+'_gradient_sum'] = forces.sum(axis=0).tolist()
        m['strap_contact_resultant']=actor_contact_resultant(self,self.strap,self.upper_mask,tab)
        m['upper_mass_kg']=float((self.resultant.volume[:,None]*1050/4*self.upper_mask[self.tets]).sum())
        self.measured = m
        self._measure_step = self.step_count
        return m

    def _record(self):
        current = read_rgb(self) if self.reference is not None else None
        m = self._measure()
        row = dict(step=self.step_count, phase=self.phase, **m)
        row['controller_target_N']=float(self.target)
        if self._stage_scorer is not None:
            row['physical_acceptance']=self._stage_scorer.snapshot()
        if hasattr(self,'probe_target_N'):
            row.update(probe_target_N=self.probe_target_N,probe_level_index=self.probe_level_index)
        if self.reference is not None:
            try:
                extractor=(getattr(self,'calibration',None) or {}).get('feature_extractor',
                    self.params.get('image_feature_extractor','flow'))
                if extractor in ('tracked_flow','tracked_flow_rgb_grid'):
                    from ._force_task_utils import MarkerFlowTracker
                    if getattr(self,'_tracker_reference',None) is not self.reference:
                        self._image_tracker=MarkerFlowTracker(self.reference)
                        self._tracker_reference=self.reference
                    if extractor=='tracked_flow_rgb_grid':
                        from ._force_task_utils import tracked_flow_rgb_features
                        features,details=tracked_flow_rgb_features(self.reference,current,self._image_tracker)
                    else:
                        features,details=self._image_tracker.update(current)
                else:
                    features,details=image_features(self.reference,current)
                row['image_feature_extractor']=extractor
                row['image_features'] = features.tolist()
                row['image_tracking'] = details
            except ValueError as e:
                row['image_error'] = str(e)
            if self.step_count % int(self.params.get('record_images_every',1000000000)) == 0:
                import cv2
                for name,im in current.items():
                    cv2.imwrite(str(self.work / f'{self.step_count:05d}_{name}.png'), cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
        if self.controller=='marker_rgb' and getattr(self,'calibration',None) and 'image_features' in row:
            from ._force_task_utils import predict_calibrated
            row['controller_tension_N']=predict_calibrated(self.calibration,row['image_features'])
        self.trace.append(row)
        with (self.work/'trace.jsonl').open('a') as f:
            import json
            f.write(json.dumps(row)+'\n')
        return row

    def _servo(self, target, q):
        manager = self._robot_manager
        pos, quat = manager.get_ee_pose_tensor()
        joints = manager.robot.data.joint_pos[:,manager._arm_ids]
        command = torch.tensor(np.r_[target,q], dtype=torch.float32,device=self.device).reshape(1,7)
        manager._ik_controller.set_command(command)
        goal = manager._ik_controller.compute(pos,quat,manager.jacobian_b[:,:,manager._arm_ids],joints)
        delta = goal-joints
        if not bool(torch.all(torch.isfinite(goal))) or float(torch.abs(delta).max()) > .08:
            raise RuntimeError('strap IK discontinuity')
        limits = manager.robot.data.soft_joint_pos_limits[:,manager._arm_ids]
        goal = torch.clamp(goal,limits[...,0],limits[...,1])
        from ._force_task_utils import execute_joint_target
        return execute_joint_target(self, arm=goal[0], ticks=2)[0]

    def _play_once(self):
        self._expert_active=True
        try:
            self._perform_strap_episode()
        except StrapDamageStop:
            pass
        finally:
            self._expert_active=False
        if self.controller!='diagnostic_ladder':
            self._finish_strap_episode()

    def _close_strap_public(self):
        """Use the same two-tick qpos path recorded for public replay."""
        from ._force_task_utils import next_grip_qpos
        from ._force_task_utils import execute_joint_target
        target_depth=float(self.params.get('grasp_depth_mm',27.6))
        record=dict(depth_target_mm=target_depth,action_repeat=2,
                    minimum_step_m=5e-6,maximum_step_m=1e-4,
                    inputs='bilateral raw tactile depth and current jaw position',
                    samples=[],stop_reason='budget')
        self.metadata['public_grasp_control']=record
        # Preparatory legacy moves may end between saved60Hz timestamps.
        if self.step_count%2:
            self.delay(1)
        for _ in range(400):
            depths=self._tactile_manager.get_min_depth().detach().cpu().numpy().reshape(-1)
            current=float(self._robot_manager.get_gripper_qpos())
            try:
                target=next_grip_qpos(depths,current,target_depth_mm=target_depth)
            except ValueError as error:
                record['stop_reason']='invalid_tactile'
                record['error']=str(error)
                self.failure='invalid_grasp_tactile'
                return False
            record['samples'].append(dict(step=int(self.step_count),
                depths_mm=depths.tolist(),qpos_m=current,target_qpos_m=target))
            if target is None:
                record['stop_reason']='bilateral_depth_target'
                return True
            if target>=current:
                record['stop_reason']='travel_limit'
                self.failure='gripper_travel_limit'
                return False
            executed,accepted=execute_joint_target(self,gripper=target,ticks=2)
            if not executed or accepted or self.failure:
                record['stop_reason']=self.failure or 'execution_stopped'
                return False
        self.failure='gripper_close_budget_exceeded'
        return False

    def _perform_strap_episode(self):
        center = self.base + [0.,0.,.174]
        q = self._robot_manager.get_gripper_center_pose().q
        for dz,tag in [(.05,'above_strap'),(0.,'approach_strap')]:
            target = Pose(center+[0.,0.,dz],q)
            self.move(self.atom.move_to_pose(self._robot_manager.gripper_center_to_ee(target)),
                      tag=tag, time_dilation_factor=.5, delay=False)
            if not self.plan_success:
                return
        if self.params.get('public_tactile_grasp',True):
            if not self._close_strap_public():
                return
        else:
            self.move(self.atom.close_gripper(0.,
                          depth_threshold=self.params.get('grasp_depth_mm', 'auto')),
                      tag='grasp_strap_tab', delay=False)
        if not self.plan_success:
            return
        # The expert uses the same observed bilateral-acquisition event as
        # arbitrary policy actions; it cannot bypass the initialization fixture.
        for _ in range(16):
            if self._grip_released:
                break
            self._step(is_save=True)
        if not self._grip_released:
            self.failure='bilateral_grip_not_acquired'
            return
        self.delay(30)
        # Use the same60Hz image timestamps as saved policy observations.
        # Off-grid renders would expose a different gel image to the expert.
        remainder = (-self.step_count) % 2
        if remainder:
            self.delay(remainder)
        self.phase = 'pull'
        self.reference = read_rgb(self)
        import cv2
        for name,im in self.reference.items():
            cv2.imwrite(str(self.work / f'reference_{name}.png'),cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
        initial = self._record()
        self.initial_tab_in_hand = np.array(initial['tab_in_hand_m'])
        target_pose = self._robot_manager.get_ee_pose()
        target_position = target_pose.p.copy()
        initial_z = float(target_position[2])
        calibration = None
        if self.controller == 'marker_rgb':
            import json
            calibration=json.loads(Path(self.params['calibration']).read_text())
        self.calibration=calibration
        if self.controller=='diagnostic_ladder':
            strap_ladder(self,target_pose)
            return
        self.active = True
        filtered=None
        control_dt=2*self.cfg.sim.dt
        goal_index=0
        image_hold_seconds=0.0
        controller_deadband=float(self.params.get('controller_deadband_N',.02))
        if not np.isfinite(controller_deadband) or not 0. <= controller_deadband <= .10:
            raise ValueError('controller_deadband_N must be finite and between 0 and 0.10 N')
        self.metadata['controller_deadband_N']=controller_deadband
        scenario=self.params.get('strap_review_scenario','')
        goals=(18.0,) if scenario=='only_second' else self.target_sequence
        self.target=float(goals[0])
        self.metadata['controller_goal_transitions']=[]
        while (not self.check_success() and self.failure is None
               and self.plan_success and self._accepted_result is None):
            row=self._record()
            if self.controller=='marker_rgb':
                sensed=row.get('controller_tension_N')
                if sensed is None:
                    self.failure='marker_tracking_lost'
                    break
            else:
                sensed=row['tension_N']
            if scenario=='idle_timeout':
                velocity=0.0
                self.phase='idle_without_target_progress'
            elif self.params.get('overload_demo',False) or scenario=='overload':
                velocity=.002
                self.phase='overload demonstration'
            elif self.controller=='position_reference':
                displacement=float(self.params['reference_displacement_m'])
                if not .002<=displacement<=.055:
                    raise ValueError('Fixed displacement reference must be2--55mm')
                error=displacement-(float(self._robot_manager.get_ee_pose().p[2])-initial_z)
                velocity=float(np.clip(2.*error,-.002,.002))
                if abs(error)<.00001:velocity=0.0
                self.phase='fixed displacement'
            else:
                alpha=1.-np.exp(-control_dt/.15)
                filtered=float(sensed) if filtered is None else filtered+alpha*(float(sensed)-filtered)
                error=self.target-filtered
                velocity=float(np.clip(error/float(self.params.get('force_gain_Ns_per_m',2000.)),-.002,.002))
                if abs(error)<controller_deadband:velocity=0.0
                image_hold_seconds=image_hold_seconds+control_dt if abs(error)<.20 else 0.0
                self.phase=('hold' if abs(error)<.20 else ('pull' if velocity>0 else 'relax'))+f'_{self.target:g}N'
                # Switch from image-estimated hold only; never read the scorer's stage.
                dwell=1.0 if scenario=='short_hold' else 3.30
                if image_hold_seconds>=dwell and scenario=='only_second':
                    self.metadata['review_controller_end']='submitted after holding18N without completing12N'
                    break
                if image_hold_seconds>=dwell and scenario=='only_first':
                    self.metadata['review_controller_end']='submitted after the first image-estimated hold only'
                    break
                if image_hold_seconds>=dwell and scenario=='short_hold' and goal_index==len(goals)-1:
                    self.metadata['review_controller_end']='submitted after deliberately short holds'
                    break
                if image_hold_seconds>=dwell and scenario=='slip':
                    from ._force_task_utils import execute_joint_target
                    self.metadata['review_slip_opening_start_step']=self.step_count
                    self.phase='opening_for_slip'
                    gripper=float(self._robot_manager.get_gripper_qpos())
                    for _ in range(160):
                        gripper=min(.039,gripper+.000025)
                        executed,accepted=execute_joint_target(self,gripper=gripper,ticks=2)
                        if not executed or accepted or self.failure:break
                    self.metadata['review_controller_end']='released grip through public joint commands under tension'
                    break
                if image_hold_seconds>=dwell and goal_index+1<len(goals) and scenario!='only_first':
                    self.metadata['controller_goal_transitions'].append({
                        'step':int(self.step_count),'completed_estimated_target_N':self.target,
                        'image_hold_seconds':image_hold_seconds})
                    goal_index+=1
                    self.target=float(goals[goal_index])
                    image_hold_seconds=0.0
            target_position[2]+=velocity*control_dt
            target_position[2]=min(target_position[2],self._robot_manager.get_ee_pose().p[2]+.001)
            if not self._servo(target_position,target_pose.q):
                break
        self.active=False
        if self.failure is None and not self.check_success():
            self.failure='incomplete_two_stage_sequence'
        self._record()
        self.metadata['strap_final']=dict(self.measured)
        self.metadata['strap_final']['physical_acceptance']=self._stage_scorer.snapshot()
        self.metadata['strap_acceptance']=self._stage_scorer.snapshot()
        self.metadata['strap_failure']=self.failure
        self.metadata['strap_trace_path']=str(self.work/'trace.jsonl')
        self.metadata['diagnostic_only']=self.controller!='marker_rgb' or bool(scenario)

    def check_success(self):
        if self._accepted_result is not None:return self._accepted_result
        if self.controller=='diagnostic_ladder':return False
        return bool(self._stage_scorer is not None and self._stage_scorer.success
                    and self.failure is None and self.plan_success)

    def _update_render(self):
        surface=getattr(self,'broken_strap_surface',None)
        if surface is not None:
            from pxr import Vt
            mesh,indices,faces=surface
            points=self.strap.vertex_positions[indices].copy()
            tri=points[faces]
            face_normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
            normals=np.zeros_like(points)
            for corner in range(3):np.add.at(normals,faces[:,corner],face_normals)
            normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-15)
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
            mesh.CreateNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
            mesh.CreateExtentAttr().Set(Vt.Vec3fArray.FromNumpy(
                np.array([points.min(0),points.max(0)],dtype=np.float32)))
        super()._update_render()

    def get_frame_shot(self, obs):
        return BaseTask.get_frame_shot(self, obs)

@configclass
class TaskCfg(BaseTaskCfg):
    step_lim = 4000
    max_save_frames = 2000
    video_size = (1120, 320)

