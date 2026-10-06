"""Wipe a randomized porcelain vase with versioned cleaning and pressure acceptance."""
from __future__ import annotations

import json
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
from ._base_task import *
from ._force_task_utils import (
    _cylinder,
    _material,
    _rounded_box,
    configure_gel,
    flow_rgb_features,
    read_rgb,
    task_parameters,
)


# Vase soil visual

def shrink_soil_triangles(triangles,remaining_coverage):
    """Reduce visible patch area continuously while retaining its surface plane."""
    triangles=np.asarray(triangles,dtype=float)
    coverage=np.asarray(remaining_coverage,dtype=float)
    if triangles.ndim!=3 or triangles.shape[1:]!=(3,3) or coverage.shape!=(len(triangles),):
        raise ValueError("Soil triangles and remaining coverage must have matching shapes")
    if not np.isfinite(triangles).all() or not np.isfinite(coverage).all():
        raise ValueError("Soil geometry and remaining coverage must be finite")
    center=triangles.mean(axis=1,keepdims=True)
    return center+(triangles-center)*np.sqrt(np.clip(coverage,0,1))[:,None,None]

def update_soil_visual(task):
    from pxr import Vt
    if not hasattr(task,"_soil_visual_density"):return
    # Interpolate removal on the shared original vertices, avoiding visible
    # triangular deletion while preserving the physically accumulated dose.
    numerator=np.zeros(len(task.soil.points));denominator=np.zeros_like(numerator)
    w=task.soil.initial*task.soil.area
    fraction=np.divide(task.soil.remaining,task.soil.initial,out=np.zeros_like(w),where=task.soil.initial>0)
    for corner in range(3):
        np.add.at(numerator,task.soil.faces[:,corner],w*fraction)
        np.add.at(denominator,task.soil.faces[:,corner],w)
    vertex=np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>0)
    cleaned_vertex=vertex[task.soil.faces[task.soil_faces]]
    remaining=np.einsum("vc,fc->fv",task._soil_visual_bary,cleaned_vertex)
    opacity=task._soil_visual_density*np.clip(remaining,0,1)
    task.soil_opacity.Set(Vt.FloatArray.FromNumpy(opacity.astype(np.float32).ravel()))
    if hasattr(task,"_soil_grain_parts"):
        coverage=opacity.ravel()[task._soil_visual_faces].mean(axis=1)
        for part,ids,original in task._soil_grain_parts:
            points=shrink_soil_triangles(original,coverage[ids])
            part.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.reshape(-1,3).astype(np.float32)))
    if hasattr(task,"_soil_alpha_meshes"):
        face_opacity=opacity.ravel()[task._soil_visual_faces].mean(axis=1)
        buckets=np.clip(np.rint(face_opacity*32),0,32).astype(np.int32)
        previous=task._soil_previous_buckets
        if previous is None or not np.array_equal(previous,buckets):
            for bucket,part in enumerate(task._soil_alpha_meshes,1):
                chosen=task._soil_visual_faces[buckets==bucket]
                part.GetFaceVertexCountsAttr().Set([3]*len(chosen))
                part.GetFaceVertexIndicesAttr().Set(chosen.ravel().tolist())
            task._soil_previous_buckets=buckets.copy()


# Vase wiping geometry

def stain_mask(points,seed,center_angle=np.pi,radius_scale=1.):
    """Seeded irregular blobs; no visual encoding of foam stiffness."""
    rng=np.random.default_rng(seed);xyz=np.asarray(points)
    angle=np.arctan2(xyz[:,1],xyz[:,0])
    relative=np.arctan2(np.sin(angle-center_angle),np.cos(angle-center_angle))
    arc=relative*np.linalg.norm(xyz[:,:2],axis=1)
    coverage=np.zeros(len(xyz));blobs=[]
    for center_z in [.047,.064,.083]:
        cx=rng.uniform(-.003,.003);cz=center_z+rng.uniform(-.002,.002)
        rx=rng.uniform(.0045,.0075)*radius_scale;rz=rng.uniform(.006,.009)*radius_scale;a=rng.uniform(-.5,.5)
        dx=arc-cx;dz=xyz[:,2]-cz
        u=(np.cos(a)*dx+np.sin(a)*dz)/rx;v=(-np.sin(a)*dx+np.cos(a)*dz)/rz
        phi=np.arctan2(v,u);rho=np.hypot(u,v);phases=rng.uniform(-np.pi,np.pi,3)
        edge=1+.17*np.sin(3*phi+phases[0])+.10*np.sin(5*phi+phases[1])+.06*np.sin(7*phi+phases[2])
        coverage=np.maximum(coverage,np.clip((edge-rho)/.16,0,1))
        blobs.append(dict(center_arc_m=cx,center_z_m=cz,radii_m=[rx,rz],angle_rad=a,edge_phases=phases.tolist()))
    return coverage,blobs


# Vase soil visual

def build_soil_visual(task,stage,root):
    from pxr import Gf,Sdf,UsdGeom,UsdShade,Vt
    # Subdivide only the visual overlay. Physics and area scoring retain the
    # original surface, and each visual vertex maps back to that surface.
    selected=np.flatnonzero(task.soil.initial>0)
    n=4
    bary=[]
    index={}
    for a in range(n+1):
        for b in range(n+1-a):
            index[a,b]=len(bary);bary.append([1-(a+b)/n,a/n,b/n])
    small=[]
    for a in range(n):
        for b in range(n-a):
            small.append([index[a,b],index[a+1,b],index[a,b+1]])
            if a+b<n-1:small.append([index[a+1,b],index[a+1,b+1],index[a,b+1]])
    bary=np.asarray(bary,float);small=np.asarray(small,int)
    tri=task.soil.points[task.soil.faces[selected]]
    local=np.einsum("vc,fcd->fvd",bary,tri)
    normals=task.soil.normals[selected]
    world=(local+normals[:,None,:]*.00008)@task.vase_rot.T+task.base
    faces=(small[None]+np.arange(len(selected))[:,None,None]*len(bary)).reshape(-1,3)
    opacity,_=stain_mask(local.reshape(-1,3),task.physics_seed,task.soil.center_angle,task.soil.radius_scale)
    xyz=local.reshape(-1,3)
    # Spatially continuous density and color; no marks encode stiffness or force.
    noise=.5+.24*np.sin(xyz[:,2]*1510+xyz[:,0]*880)+.18*np.sin(xyz[:,1]*2100-xyz[:,2]*710)
    density=np.clip(.55+.30*noise,.42,.88)
    color=np.column_stack((.17+.065*noise,.095+.05*noise,.043+.030*noise))
    mesh=UsdGeom.Mesh.Define(stage,root+"/soil")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(world.reshape(-1,3).astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr([3]*len(faces));mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
    mesh.CreateSubdivisionSchemeAttr("none");mesh.CreateDoubleSidedAttr(True)
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.repeat(normals@task.vase_rot.T,len(bary),axis=0).astype(np.float32)))
    mesh.SetNormalsInterpolation("vertex")
    flattened=world.reshape(-1,3).astype(np.float32)
    mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.array([flattened.min(0),flattened.max(0)],np.float32)))
    pv=UsdGeom.PrimvarsAPI(mesh)
    pv.CreatePrimvar("soil_color",Sdf.ValueTypeNames.Color3fArray,"vertex").Set(Vt.Vec3fArray.FromNumpy(color.astype(np.float32)))
    opacity_pv=pv.CreatePrimvar("soil_opacity",Sdf.ValueTypeNames.FloatArray,"vertex")
    mat=UsdShade.Material.Define(stage,root+"/Looks/Soil")
    shader=UsdShade.Shader.Define(stage,str(mat.GetPath())+"/Shader");shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness",Sdf.ValueTypeNames.Float).Set(.94)
    shader.CreateInput("metallic",Sdf.ValueTypeNames.Float).Set(0.)
    for name,kind,input_name,type_name in [
        ("soil_color","float3","diffuseColor",Sdf.ValueTypeNames.Color3f),
        ("soil_opacity","float","opacity",Sdf.ValueTypeNames.Float)]:
        reader=UsdShade.Shader.Define(stage,str(mat.GetPath())+"/"+name)
        reader.CreateIdAttr("UsdPrimvarReader_"+kind)
        reader.CreateInput("varname",Sdf.ValueTypeNames.String).Set(name)
        reader.CreateInput("fallback",type_name).Set(Gf.Vec3f(.22,.12,.06) if kind=="float3" else 1.)
        reader.CreateOutput("result",type_name)
        shader.CreateInput(input_name,type_name).ConnectToSource(reader.ConnectableAPI(),"result")
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),"surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    task.soil_mesh=mesh;task.soil_opacity=opacity_pv
    task.soil_faces=selected;task._soil_visual_bary=bary
    task._soil_visual_density=(opacity*density).reshape(len(selected),len(bary))
    if task.params.get("soil_material_mode","opaque_grains")=="constant_bands":
        # Renderer-compatible alternative: constant ordinary PreviewSurface
        # materials in32 opacity bands. Physical dose and visual geometry stay
        # identical; no custom primvar reader is required by this route.
        from ._force_task_utils import _material
        UsdGeom.Imageable(mesh.GetPrim()).MakeInvisible()
        task._soil_alpha_meshes=[]
        task._soil_visual_faces=faces
        for bucket in range(1,33):
            part=UsdGeom.Mesh.Define(stage,root+f"/soil_alpha_{bucket:02d}")
            part.CreatePointsAttr(mesh.GetPointsAttr().Get())
            part.CreateExtentAttr(mesh.GetExtentAttr().Get())
            part.CreateSubdivisionSchemeAttr("none");part.CreateDoubleSidedAttr(True)
            part.CreateNormalsAttr(mesh.GetNormalsAttr().Get());part.SetNormalsInterpolation("vertex")
            part.CreateFaceVertexCountsAttr([]);part.CreateFaceVertexIndicesAttr([])
            material=_material(stage,root+f"/Looks/SoilAlpha{bucket:02d}",(.20,.11,.045),.94)
            surface=UsdShade.Shader(stage.GetPrimAtPath(str(material.GetPath())+"/Shader"))
            surface.CreateInput("opacity",Sdf.ValueTypeNames.Float).Set(bucket/32.)
            UsdShade.MaterialBindingAPI.Apply(part.GetPrim()).Bind(material)
            task._soil_alpha_meshes.append(part)
        task._soil_previous_buckets=None
    if task.params.get("soil_material_mode","opaque_grains")=="opaque_grains":
        # Opaque micro-patches avoid renderer-dependent transparent cutout.
        # Patch area, not an alpha shader, follows the same remaining soil dose.
        from ._force_task_utils import _material
        UsdGeom.Imageable(mesh.GetPrim()).MakeInvisible()
        task._soil_grain_parts=[]
        triangles=flattened[faces]
        shade=np.clip((noise[faces].mean(1)*8).astype(int),0,7)
        task._soil_visual_faces=faces
        for bucket in range(8):
            ids=np.flatnonzero(shade==bucket)
            if not len(ids):continue
            part=UsdGeom.Mesh.Define(stage,root+f"/soil_grains_{bucket}")
            original=triangles[ids].copy()
            part.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(original.reshape(-1,3)))
            part.CreateExtentAttr(mesh.GetExtentAttr().Get())
            part.CreateFaceVertexCountsAttr([3]*len(ids))
            part.CreateFaceVertexIndicesAttr(list(range(3*len(ids))))
            part.CreateSubdivisionSchemeAttr("none");part.CreateDoubleSidedAttr(True)
            flat_normals=np.cross(original[:,1]-original[:,0],original[:,2]-original[:,0])
            flat_normals/=np.maximum(np.linalg.norm(flat_normals,axis=1,keepdims=True),1e-15)
            part.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.repeat(flat_normals,3,axis=0).astype(np.float32)))
            part.SetNormalsInterpolation("vertex")
            variation=(bucket+.5)/8
            color=(.17+.065*variation,.095+.05*variation,.043+.03*variation)
            material=_material(stage,root+f"/Looks/SoilGrain{bucket}",color,.94)
            UsdShade.MaterialBindingAPI.Apply(part.GetPrim()).Bind(material)
            task._soil_grain_parts.append((part,ids,original))
    update_soil_visual(task)
    task._soil_shader_audit=dict(
        material_mode=task.params.get("soil_material_mode","opaque_grains"),
        primvar_reader_varname_type="string",nonzero_density_vertices=int(np.count_nonzero(task._soil_visual_density)),
        maximum_initial_opacity=float(task._soil_visual_density.max()),
        overlay_world_bbox_m=[flattened.min(0).tolist(),flattened.max(0).tolist()],
        source_surface_offset_m=.00008)


# Vase surface damage

def show_scuff(task,center_world,tangent_world):
    """Expose matte white substrate only on the actual load-bearing patch."""
    from pxr import UsdGeom,UsdShade,Vt
    import omni.usd
    from ._force_task_utils import _material
    force=task._vase_forces()
    load=np.maximum(0.,-np.sum(force*task.vertex_normals,axis=1))
    faces=task.vase_data['faces']
    face_load=load[faces].sum(axis=1)
    radial=task.soil.centers.copy();radial[:,2]=0.
    outward=np.sum(task.soil.normals*radial,axis=1)>0
    threshold=max(1e-8,float(face_load.max())*.002)
    selected=np.flatnonzero((face_load>threshold)&outward)
    if not len(selected):
        raise RuntimeError('Cannot draw glaze loss without an actual contact patch')
    tri=task.vase_data['points'][faces[selected]]
    local=tri+task.soil.normals[selected,None,:]*.00016
    points=local.reshape(-1,3)@task.vase_rot.T+task.base
    stage=omni.usd.get_context().get_stage()
    root=task.scene.env_prim_paths[0]+'/vase_wiping_presentation'
    mesh=UsdGeom.Mesh.Define(stage,root+'/glaze_scuff')
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr([3]*len(selected))
    mesh.CreateFaceVertexIndicesAttr(np.arange(len(points)).tolist())
    mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateDoubleSidedAttr(True)
    material=_material(stage,root+'/Looks/ExposedCeramic',(1.,1.,1.),1.)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    task.glaze_scuff_mesh=mesh
    return dict(contact_center_m=np.asarray(center_world).tolist(),
                direction_world=np.asarray(tangent_world).tolist(),
                projected_on_actual_surface=True,exposed_faces=selected.tolist(),
                exposed_area_m2=float(task.soil.area[selected].sum()),
                appearance='opaque pure white matte ceramic; local glaze and pattern removed',
                irreversible=True,cleaning_credit=False)


# Vase wiping geometry

class SoilState:
    """A declared pressure-times-sliding-distance cleaning approximation."""
    def __init__(self,points,faces,seed,center_angle=np.pi,area_scale=1.10):
        self.points=np.asarray(points);self.faces=np.asarray(faces)
        tri=self.points[self.faces];self.centers=tri.mean(axis=1)
        cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        self.area=np.linalg.norm(cross,axis=1)*.5
        self.normals=cross/np.maximum(2*self.area[:,None],1e-16)
        radial=self.centers.copy();radial[:,2]=0
        outside=np.sum(self.normals*radial,axis=1)>0
        if not np.isfinite(area_scale) or not 1.<=area_scale<=1.25:
            raise ValueError("Soil area scale must be in [1,1.25]")
        self.seed=seed;self.center_angle=center_angle
        baseline,_=stain_mask(self.centers,seed,center_angle)
        self.baseline_area_m2=float(np.sum(baseline*outside*self.area))
        target=self.baseline_area_m2*area_scale
        low,high=1.,1.35
        for _ in range(36):
            mid=(low+high)/2
            candidate,_=stain_mask(self.centers,seed,center_angle,radius_scale=mid)
            if np.sum(candidate*outside*self.area)<target:low=mid
            else:high=mid
        self.radius_scale=(low+high)/2 if area_scale>1. else 1.
        initial,self.blobs=stain_mask(self.centers,seed,center_angle,radius_scale=self.radius_scale)
        self.initial=initial*outside;self.remaining=self.initial.copy()
        self.initial_area_m2=float(np.sum(self.initial*self.area))
        self.actual_area_scale=self.initial_area_m2/self.baseline_area_m2
        self.weight=self.initial*self.area
        assert self.weight.sum()>0,"stain region misses outer surface"
        self.dose=np.zeros(len(faces));self.vertex_area=np.zeros(len(points))
        for i in range(3):np.add.at(self.vertex_area,self.faces[:,i],self.area/3)

    def advance(self,vertex_force,sliding_m):
        force=np.asarray(vertex_force)
        pressure=np.maximum(0,-np.sum(force[self.faces].mean(axis=1)*self.normals,axis=1))
        pressure/=np.maximum(self.vertex_area[self.faces].mean(axis=1),1e-10)
        delta=np.maximum(0,pressure/10000)*max(0,float(sliding_m))/.006
        self.dose+=np.where(pressure>=4000,delta,0)
        self.remaining=self.initial*np.clip(1-self.dose,0,1)
        return self.fraction

    @property
    def fraction(self):
        return float(1-np.sum(self.remaining*self.area)/self.weight.sum())

def sample_vase_pose(seed, parameters):
    """Sample fixture-supported placement with a separate, repeatable RNG stream."""
    nominal = np.array([.60, .09, .080], dtype=float)
    bounds = np.asarray(parameters.get("vase_position_half_range_m", [0., 0., 0.]), dtype=float)
    if bounds.shape != (3,) or not np.all(np.isfinite(bounds)) or np.any(bounds < 0):
        raise ValueError("vase_position_half_range_m must contain three finite nonnegative values")
    seed = int(seed)
    if seed < 0:
        raise ValueError("Vase placement requires a nonnegative physical seed")
    override = parameters.get("vase_position_offset_m")
    if override is None:
        offset = np.random.default_rng(seed + 86473).uniform(-bounds, bounds)
        source = "independent_uniform_xyz"
    else:
        offset = np.asarray(override, dtype=float)
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            raise ValueError("vase_position_offset_m must contain three finite values")
        if np.any(np.abs(offset) > bounds + 1e-12):
            raise ValueError("Explicit vase offset exceeds configured placement bounds")
        source = "explicit_boundary_check"
    base = nominal + offset
    return base, dict(version="vase_placement_v1", source=source,
                      seed=seed, rng_seed=seed + 86473,
                      nominal_base_m=nominal.tolist(), half_range_m=bounds.tolist(),
                      offset_m=offset.tolist(), sampled_base_m=base.tolist(),
                      orientation="unchanged horizontal vase",
                      fixture="base plate translated in XY; support heights follow vase Z")


# Wipe porcelain vase

ASSET=OBJECTS_ROOT/'task_assets/vase_wiping'

class Task(BaseTask):
    def __init__(self,cfg,**kwargs):
        self.params=task_parameters(cfg);self.work=Path(self.params['workspace'])
        self.work.mkdir(parents=True,exist_ok=True)
        self.physics_seed=int(self.params.get('physics_seed',0))
        self.controller=self.params.get('controller','diagnostic_force')
        self.target=float(self.params.get('target_force_N',8.))
        self.required_clean_fraction=float(self.params.get('required_clean_fraction',.925))
        if self.required_clean_fraction not in (.925,.95):
            raise ValueError('Unsupported vase cleaning acceptance; use the approved92.5% or legacy95% profile')
        self.modulus=float(self.params.get('foam_modulus_mpa',np.random.default_rng(self.physics_seed+178).uniform(.12,.30)))
        self.vase_data=dict(np.load(ASSET/'vase.npz'))
        self.foam_data=dict(np.load(ASSET/'foam.npz'))
        self.base,self.placement=sample_vase_pose(self.physics_seed,self.params)
        self.foam_base=np.array([.455,-.09,.006])
        # The existing fixture stays on the table; its posts reach the new height.
        support_radii=np.interp([.014,.138],self.vase_data['profile_z'],self.vase_data['profile_r'])
        if (self.base[2]-float(np.max(self.vase_data['profile_r']))<=.010
                or np.any(self.base[2]-support_radii-.006+.002-.010<=0)):
            raise ValueError('Sampled vase height intersects the table-mounted fixture')
        self.vase_rot=t3d.axangles.axangle2mat([1,0,0],np.pi/2)
        self.vase_q=t3d.quaternions.mat2quat(self.vase_rot)
        self.phase='setup';self.reference=None;self.trace=[];self.failure=None;self.lane_angle=0.
        self.max_force=0.;self.last_foam_center=None;self.cleaned=0.;self.rgb_model=None
        self._monitor_wiping=False;self._sponge_free=False;self._recorded_step=None
        self.last_contact_foam_points=None;self._last_scored_step=None
        self.glaze_damage_work_J=0.;self._accepted_result=None
        self.wiping_distance_m=0.;self.good_wiping_distance_m=0.;self.pressure_band_fraction=0.
        self.soil=SoilState(self.vase_data['points'],self.vase_data['faces'],self.physics_seed,center_angle=np.pi/2)
        self.vertex_normals=np.zeros_like(self.vase_data['points'])
        for corner in range(3):
            np.add.at(self.vertex_normals,self.soil.faces[:,corner],self.soil.normals*self.soil.area[:,None])
        self.vertex_normals/=np.maximum(np.linalg.norm(self.vertex_normals,axis=1,keepdims=True),1e-15)
        self.vertex_normals=self.vertex_normals@self.vase_rot.T
        eye=np.array([.94,-.31,.45]);aim=np.array([.545,-.005,.078])
        z=(eye-aim)/np.linalg.norm(eye-aim);x=np.cross([0.,0.,1.],z);x/=np.linalg.norm(x)
        y=np.cross(z,x);q=t3d.quaternions.mat2quat(np.stack([x,y,z],axis=1))
        cfg.cameras[0]=CameraCfg(name='head',prim_path='/World/envs/env_.*/Camera',
            offset=CameraCfg.OffsetCfg(pos=tuple(eye),rot=tuple(q),convention='opengl'),
            data_types=['rgb','depth'],spawn=sim_utils.PinholeCameraCfg(focal_length=2.0,
            focus_distance=1.,horizontal_aperture=2.4,clipping_range=(.02,100.)),
            width=480,height=270,update_period=1/120)
        cfg.uipc_sim.contact.eps_velocity=.001
        cfg.uipc_sim.newton.velocity_tol=.001
        if 'newton_max_iter' in self.params:cfg.uipc_sim.newton.max_iter=int(self.params['newton_max_iter'])
        from ._force_task_utils import configure_final_task
        configure_final_task(cfg, self.params, max_policy_seconds=160)
        if self.params.get('enable_fractional_soil_opacity',True):
            settings=dict(cfg.sim.render.carb_settings or {})
            settings['/rtx/raytracing/fractionalCutoutOpacity']=True
            cfg.sim.render.carb_settings=settings
        super().__init__(cfg,**kwargs)
        self.video_handler.fps=120/cfg.video_frequency;self.video_handler.encoder_threads=2

    def _setup_scene(self):
        configure_gel(self.cfg.robot.tactiles,dict(gel_modulus_mpa=float(self.params.get('gel_modulus_mpa',.10))))
        super()._setup_scene()
        contacts=self.uipc_sim.scene.contact_tabular()
        foam=contacts.create('vase_wiping_foam');vase=contacts.create('vase_wiping_porcelain')
        resistance=self.cfg.uipc_sim.contact.default_contact_resistance*1e9
        contacts.insert(foam,contacts.default_element(),friction_rate=2.0,resistance=resistance)
        contacts.insert(foam,vase,friction_rate=.45,resistance=resistance)
        for actor,element in [(self.sponge,foam),(self.vase,vase)]:
            for mesh in actor.uipc_meshes:element.apply_to(mesh)
        self._build_presentation()

    def create_actors(self):
        self.vase=self._actor_manager.add_from_usd_file(name='porcelain_vase',
            asset_path=ASSET/'vase.usda',pose=Pose(self.base,self.vase_q),
            constitution_cfg=UipcObjectCfg.AffineBodyConstitutionCfg(kinematic=True),
            density=2200.,show_physics_mesh=True,keep_constrained=True)
        self.sponge=self._actor_manager.add_from_usd_file(name='wiping_sponge',
            asset_path=ASSET/'foam.usda',pose=Pose(self.foam_base,[1,0,0,0]),
            constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(youngs_modulus=self.modulus,poisson_rate=.30),
            density=250.,show_physics_mesh=True,keep_constrained=True)
        if self.params.get('foam_core_modulus_mpa') is not None:
            from uipc import view
            core=float(self.params['foam_core_modulus_mpa'])
            skin=float(self.params.get('foam_soft_skin_m',.008))
            if core<=0 or not 0.<skin<.042:raise ValueError('invalid layered wiping pad')
            z=self.foam_data['points'][self.foam_data['tets'],2].mean(axis=1)
            young=np.where(z<skin,self.modulus,core)*1e6
            geo=self.sponge.uipc_meshes[0]
            view(geo.tetrahedra().find('mu')).reshape(-1)[:]=young/(2.*1.3)
            view(geo.tetrahedra().find('lambda')).reshape(-1)[:]=young*.30/(1.3*.40)

    def _build_presentation(self):
        import omni.usd
        stage=omni.usd.get_context().get_stage()
        for env in self.scene.env_prim_paths:
            root=env+'/vase_wiping_presentation';UsdGeom.Xform.Define(stage,root)
            dark=_material(stage,root+'/Looks/Dark',(.035,.045,.052),.74,.15)
            alloy=_material(stage,root+'/Looks/Alloy',(.42,.45,.46),.4,.65)
            liner=_material(stage,root+'/Looks/Liner',(.045,.048,.044),.91)

            _rounded_box(stage,root+'/base',(self.base[0],self.base[1]-.085,.005),(.130,.205,.010),.013,dark)
            def saddle(path,axial,radius,thick,width,material):
                angles=np.linspace(np.pi+np.pi/9,2*np.pi-np.pi/9,49)
                points=np.array([[rr*np.cos(a),rr*np.sin(a),z] for z in [axial-width/2,axial+width/2]
                    for rr in [radius,radius+thick] for a in angles])
                n=len(angles);faces=[]
                for j in range(n-1):
                    for offset in [0,n]:
                        a=j+offset;b=a+1
                        faces.extend([(a,b,b+2*n),(a,b+2*n,a+2*n)])
                    faces.extend([(j,j+n,j+n+1),(j,j+n+1,j+1),
                        (j+2*n,j+2*n+1,j+3*n+1),(j+2*n,j+3*n+1,j+3*n)])
                for j in [0,n-1]:faces.extend([(j,j+2*n,j+3*n),(j,j+3*n,j+n)])
                points=points@self.vase_rot.T+self.base
                mesh=UsdGeom.Mesh.Define(stage,path);mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
                mesh.CreateFaceVertexCountsAttr([3]*len(faces));mesh.CreateFaceVertexIndicesAttr(np.asarray(faces).ravel().tolist())
                mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateDoubleSidedAttr(True)
                import trimesh
                mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(trimesh.Trimesh(points,faces,process=False).vertex_normals.astype(np.float32)))
                mesh.SetNormalsInterpolation('vertex');UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
            for n,axial in enumerate([.014,.138]):
                radius=float(np.interp(axial,self.vase_data['profile_z'],self.vase_data['profile_r']))
                bottom=self.base[2]-radius-.006
                _rounded_box(stage,root+f'/support_{n}',(self.base[0],self.base[1]-axial,(.010+bottom+.002)/2),
                    (.042,.024,bottom+.002-.010),.005,alloy)
                saddle(root+f'/cradle_{n}',axial,radius+.0005,.006,.014,alloy)
                saddle(root+f'/liner_{n}',axial,radius+.0001,.001,.015,liner)
            for n,(dx,dy) in enumerate([(-.048,-.086),(.048,-.086),(-.048,.086),(.048,.086)]):
                _cylinder(stage,root+f'/bolt_{n}',(self.base[0]+dx,self.base[1]-.085+dy,.0105),.003,.001,alloy)
            _rounded_box(stage,root+'/sponge_rest',tuple(self.foam_base-[0,0,.003]),(.070,.048,.006),.009,dark)
            vase_root=stage.GetPrimAtPath(env+'/porcelain_vase')
            for prim in Usd.PrimRange(vase_root):
                if not prim.IsA(UsdGeom.Mesh):continue
                mesh=UsdGeom.Mesh(prim)
                local=(np.asarray(mesh.GetPointsAttr().Get())-self.base)@self.vase_rot
                faces=np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
                uv=np.column_stack([(np.arctan2(local[:,1],local[:,0])/(2*np.pi))%1,local[:,2]/.17])[faces].copy()
                seam=np.ptp(uv[:,:,0],axis=1)>.5
                uv[seam,:,0]=np.where(uv[seam,:,0]<.5,uv[seam,:,0]+1,uv[seam,:,0])
                UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,
                    UsdGeom.Tokens.faceVarying).Set(Vt.Vec2fArray.FromNumpy(uv.reshape(-1,2).astype(np.float32)))
                import trimesh
                normals=trimesh.Trimesh(local,faces,process=False).vertex_normals@self.vase_rot.T
                mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
                mesh.SetNormalsInterpolation('vertex')
            sponge_root=stage.GetPrimAtPath(env+'/wiping_sponge')
            for prim in Usd.PrimRange(sponge_root):
                if not prim.IsA(UsdGeom.Mesh):continue
                source_mesh=UsdGeom.Mesh(prim)
                points=np.asarray(source_mesh.GetPointsAttr().Get())
                faces=np.asarray(source_mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
                # Publish the actual FEM surface through USD. Late material/subset
                # edits can leave Hydra displaying stale Fabric geometry.
                from scipy.spatial import cKDTree
                rest_world=np.asarray(self.sponge.uipc_meshes[0].positions().view()).reshape(-1,3)
                distance,indices=cKDTree(rest_world).query(points)
                if distance.max()>1e-6 or len(np.unique(indices))!=len(indices):
                    raise RuntimeError('Foam render surface does not map uniquely to FEM vertices')
                UsdGeom.Imageable(prim).MakeInvisible()
                mesh=UsdGeom.Mesh.Define(stage,root+'/wiping_pad')
                prim=mesh.GetPrim()
                mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
                mesh.CreateFaceVertexCountsAttr([3]*len(faces))
                mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
                mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateDoubleSidedAttr(True)
                self.sponge_render_mesh=mesh
                self.sponge_render_faces=faces.copy()
                self.sponge_surface_indices=indices.copy()
                # UVs follow the same deforming mesh; no rigid shell hides foam compression.
                local=points-self.foam_base
                centers=local[faces].mean(axis=1)
                tri=local[faces]
                cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);dominant=np.abs(cross).argmax(axis=1)
                uv=np.empty((len(faces),3,2))
                for axis,plane in [(0,[1,2]),(1,[0,2]),(2,[0,1])]:
                    selected=dominant==axis
                    uv[selected]=(tri[selected][:,:,plane]+np.array([.026,.014,0])[plane])/.022
                import trimesh
                mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(trimesh.Trimesh(local,faces,process=False).vertex_normals.astype(np.float32)))
                mesh.SetNormalsInterpolation('vertex')
                UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,
                    UsdGeom.Tokens.faceVarying).Set(Vt.Vec2fArray.FromNumpy(uv.reshape(-1,2).astype(np.float32)))
                for label,filename,color in [('Foam','foam.jpg',(.18,.50,.75)),('Scour','scouring.jpg',(.05,.21,.36))]:
                    mat=_material(stage,root+'/Looks/'+label,color,.82)
                    shader=UsdShade.Shader(stage.GetPrimAtPath(root+'/Looks/'+label+'/Shader'))
                    tex=UsdShade.Shader.Define(stage,root+'/Looks/'+label+'/Texture');tex.CreateIdAttr('UsdUVTexture')
                    tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(str(ASSET/filename))
                    tex.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB')
                    tex.CreateInput('wrapS',Sdf.ValueTypeNames.Token).Set('repeat');tex.CreateInput('wrapT',Sdf.ValueTypeNames.Token).Set('repeat')
                    if label=='Scour':tex.CreateInput('scale',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(.42,.60,.75,1.))
                    reader=UsdShade.Shader.Define(stage,root+'/Looks/'+label+'/UV');reader.CreateIdAttr('UsdPrimvarReader_float2')
                    reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st');reader.CreateOutput('result',Sdf.ValueTypeNames.Float2)
                    tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result')
                    tex.CreateOutput('rgb',Sdf.ValueTypeNames.Float3)
                    shader.GetInput('diffuseColor').ConnectToSource(tex.ConnectableAPI(),'rgb')
                    if label=='Foam':UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)
                    else:
                        subset=UsdGeom.Subset.Define(stage,str(prim.GetPath())+'/scouring_face')
                        subset.CreateElementTypeAttr('face');subset.CreateFamilyNameAttr('materialBind')
                        subset.CreateIndicesAttr(np.flatnonzero(centers[:,2]<.0001).tolist())
                        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(mat)
            build_soil_visual(self,stage,root)

    def _update_render(self):
        if hasattr(self,'sponge_render_mesh'):
            points=self.sponge.vertex_positions[self.sponge_surface_indices].copy()
            faces=self.sponge_render_faces
            tri=points[faces]
            face_normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
            normals=np.zeros_like(points)
            for corner in range(3):np.add.at(normals,faces[:,corner],face_normals)
            normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-15)
            mesh=self.sponge_render_mesh
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
            mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
            mesh.CreateExtentAttr().Set(Vt.Vec3fArray.FromNumpy(
                np.array([points.min(axis=0),points.max(axis=0)],dtype=np.float32)))
        super()._update_render()

    def _update_soil_visual(self):
        update_soil_visual(self)

    def reset(self,*args,**kwargs):
        self._monitor_wiping=False
        result=super().reset(*args,**kwargs)
        self._monitor_wiping=True
        return result

    def _reset_actors(self):
        if int(self.cfg.seed)!=self.physics_seed:
            raise ValueError('Create one vase process per physical seed')
        self.soil.remaining=self.soil.initial.copy()
        self.soil.dose.fill(0.)
        self.phase='setup';self.failure=None;self.trace=[];self.reference=None;self.rgb_model=None
        self.max_force=0.;self.last_foam_center=None;self.cleaned=0.;self.lane_angle=0.
        self._sponge_free=False;self._recorded_step=None;self.last_contact_foam_points=None;self._last_scored_step=None
        self.glaze_damage_work_J=0.;self._accepted_result=None
        if hasattr(self,'glaze_scuff_mesh'):
            stage=self.glaze_scuff_mesh.GetPrim().GetStage()
            stage.RemovePrim(self.glaze_scuff_mesh.GetPath())
            del self.glaze_scuff_mesh
        self.wiping_distance_m=0.;self.good_wiping_distance_m=0.;self.pressure_band_fraction=0.
        self._update_soil_visual()
        self.vase.set_pose(Pose(self.base,self.vase_q));self.sponge.set_pose(Pose(self.foam_base,[1,0,0,0]))
        self.metadata['vase_wiping_parameters']=dict(self.params,foam_modulus_mpa=self.modulus,gel_modulus_mpa=float(self.params.get('gel_modulus_mpa',.10)))
        self.metadata['vase_placement']=dict(self.placement)
        self.metadata['vase_placement']['fixed_during_episode']=True
        self.metadata['stain_blobs']=self.soil.blobs
        self.metadata['stain_area']=dict(baseline_m2=self.soil.baseline_area_m2,
            initial_m2=self.soil.initial_area_m2,ratio=self.soil.actual_area_scale,
            radius_scale=self.soil.radius_scale,definition='same-seed area-weighted initial soil coverage')
        self.metadata['soil_shader_audit']=getattr(self,'_soil_shader_audit',{})
        self.metadata['soil_render']='thin spatially varied layer; continuous physical-dose fading on the unchanged vase surface'
        self.metadata['diagnostic_only']=self.controller!='marker_rgb'
        self.metadata['cleaning_model']='Compressive contact pressure times force-weighted tangential motion of the contacting FEM soft skin; no handle-compression or waypoint erasure'
        self.metadata['render_contact_geometry']='Same source-textured vase; USD foam surface maps one-to-one to live FEM vertices with updated normals and extent'
        self.metadata['required_clean_fraction']=self.required_clean_fraction
        self.metadata['vase_acceptance_version']='clean925_v1' if self.required_clean_fraction==.925 else 'clean95_v1'
        self.metadata['stain_surface']='top of horizontal vase; local azimuth pi/2, world +Z'
        self.metadata['wiping_face']='pad bottom; physical downward pressure'

    def _release_reset_constraints(self):pass

    def build_instruction(self):
        return "Wipe the stains from the fixed porcelain vase while regulating contact pressure."

    def pre_move(self):
        self.move(self.atom.open_gripper(.9),tag='open_for_sponge',delay=False)

    def _vase_forces(self):
        from uipc import builtin
        idx,grad=self.uipc_sim.get_contact_gradient()
        idx=np.asarray(idx).reshape(-1);grad=np.asarray(grad).reshape(-1,3)
        start=int(self.vase.geo_slot_list[0].geometry().meta().find(builtin.global_vertex_offset).view()[0])
        forces=np.zeros_like(self.vase_data['points'])
        valid=(idx>=start)&(idx<start+len(forces))
        np.add.at(forces,(idx[valid]-start).astype(int),-grad[valid]/self.cfg.sim.dt**2)
        return forces

    def _contact_sliding(self,normal,normal_force):
        # Track the contacting soft-skin material itself. Motion of the handle
        # during normal compression must not count as a wiping stroke.
        points=self.sponge.vertex_positions.copy()
        previous=self.last_contact_foam_points
        self.last_contact_foam_points=points
        if previous is None or normal_force<=1e-8:return 0.
        from uipc import builtin
        idx,grad=self.uipc_sim.get_contact_gradient()
        idx=np.asarray(idx).reshape(-1);grad=np.asarray(grad).reshape(-1,3)
        start=int(self.sponge.geo_slot_list[0].geometry().meta().find(builtin.global_vertex_offset).view()[0])
        selected=(idx>=start)&(idx<start+len(points))
        force=np.zeros_like(points)
        np.add.at(force,(idx[selected]-start).astype(int),-grad[selected]/self.cfg.sim.dt**2)
        skin=self.foam_data['points'][:,2]<float(self.params.get('foam_soft_skin_m',.008))
        # A curved contact patch has different normals at each material point.
        # Project on local surface tangents so pure local indentation is not
        # mistaken for wiping because of a single averaged patch normal.
        if not hasattr(self,'_vase_normal_tree'):
            from scipy.spatial import cKDTree
            from ._force_task_utils import world_contact_vertices
            self._vase_normal_tree=cKDTree(world_contact_vertices(self.vase))
        _,nearest=self._vase_normal_tree.query(points)
        local_normals=self.vertex_normals[nearest]
        weights=np.maximum(0.,np.sum(force*local_normals,axis=1))*skin
        if weights.sum()<1e-8:return 0.
        delta=points-previous
        tangent=delta-np.sum(delta*local_normals,axis=1)[:,None]*local_normals
        self._contact_tangent_motion=(tangent*weights[:,None]).sum(axis=0)/weights.sum()
        return float((np.linalg.norm(tangent,axis=1)*weights).sum()/weights.sum())

    def _record(self):
        if self._recorded_step==self.step_count:return self.measured
        if self.last_render!=self.step_count:self._update_render()
        force=self._vase_forces()
        center=self.sponge.vertex_positions.mean(axis=0)
        fn=float(np.maximum(0,-np.sum(force*self.vertex_normals,axis=1)).sum())
        contact_load=np.maximum(0,-np.sum(force*self.vertex_normals,axis=1))
        normal=(self.vertex_normals*contact_load[:,None]).sum(axis=0)
        normal/=max(np.linalg.norm(normal),1e-12)
        # Physical dose must not depend on extra expert/logger reads. The
        # automatic monitor samples every four physics ticks for all policies.
        # Reference/model cache invalidation must not score the same tick twice.
        score_step=self.step_count%4==0 and self._last_scored_step!=self.step_count
        slip=0.
        if score_step:
            self._last_scored_step=self.step_count
            slip=self._contact_sliding(normal,fn)
            self.last_foam_center=center.copy()
            self.cleaned=self.soil.advance(force@self.vase_rot,slip)
            if fn>.5 and slip>1e-6:
                self.wiping_distance_m+=slip
                low,high=self.params.get('pressure_band_N',[7.2,8.8])
                if low<=fn<=high:self.good_wiping_distance_m+=slip
            self.pressure_band_fraction=self.good_wiping_distance_m/max(self.wiping_distance_m,1e-12)
            self._update_soil_visual()
        self.max_force=max(self.max_force,fn)
        load=np.maximum(0,-np.sum(force*self.vertex_normals,axis=1))
        from ._force_task_utils import world_contact_vertices
        vase_world=world_contact_vertices(self.vase)
        contact_center=(vase_world*load[:,None]).sum(axis=0)/max(fn,1e-12)
        if score_step and self.params.get('glaze_damage_enabled',True) and self.failure is None:
            limit=float(self.params.get('abrasion_force_N',10.5))
            self.glaze_damage_work_J+=max(0.,fn-limit)*slip*.45
            if self.glaze_damage_work_J>=float(self.params.get('glaze_damage_work_J',.00015)):
                self.failure='glaze_abraded'
                visual=show_scuff(self,contact_center,getattr(self,'_contact_tangent_motion',np.zeros(3)))
                self.metadata['glaze_damage']=dict(step=self.step_count,normal_force_N=fn,
                    excess_contact_work_J=self.glaze_damage_work_J,
                    force_threshold_N=limit,work_threshold_J=float(self.params.get('glaze_damage_work_J',.00015)),
                    model='cumulative excess normal load times actual contact sliding and friction; declared coating abrasion approximation',
                    **visual)
        row=dict(step=self.step_count,phase=self.phase,normal_force_N=fn,
            contact_center_m=contact_center.tolist(),
            ee_position_m=self._robot_manager.get_ee_pose().p.tolist(),
            cleaned_fraction=self.soil.fraction,sponge_center_m=center.tolist(),sliding_m=slip,
            vase_force_resultant_N=force.sum(axis=0).tolist(),
            vertex_force_peak_N=float(np.linalg.norm(force,axis=1).max()),
            vase_displacement_m=float(np.linalg.norm(vase_world-(self.vase_data['points']@self.vase_rot.T+self.base),axis=1).max()))
        if self.reference is not None:
            current=read_rgb(self)
            try:
                features,details=flow_rgb_features(self.reference,current)
                row['image_features']=features.tolist();row['image_tracking']=details
                if self.rgb_model is not None:
                    row['rgb_force_N']=float(features@np.asarray(self.rgb_model['weights'])+self.rgb_model['bias'])
            except ValueError as exc:row['image_error']=str(exc)
        self.trace.append(row)
        with (self.work/'trace.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        self.measured=row
        self._recorded_step=self.step_count
        return row

    def _step(self,is_save=True):
        previous_step=self.step_count
        super()._step(is_save=is_save)
        if self.step_count==previous_step or not getattr(self,'_monitor_wiping',False):return
        if not self._sponge_free:
            depth=self._tactile_manager.get_min_depth().detach().cpu().numpy().reshape(-1)
            if np.all(depth<=float(self.params.get('grasp_depth_mm',27.2))+.1):
                self.sponge.remove_animate(force=True)
                self._actor_manager.update(dt=0.)
                self._sponge_free=True
                self.metadata['bilateral_grip_release_step']=self.step_count
                self.metadata['bilateral_grip_release_depth_mm']=depth.tolist()
        if self.step_count%4==0:
            row=self._record()
            load=row['normal_force_N']
        else:
            force=self._vase_forces()
            load=float(np.maximum(0,-np.sum(force*self.vertex_normals,axis=1)).sum())
        self.max_force=max(self.max_force,load)
        if self.failure is None and load>float(self.params.get('overload_force_N',12.5)):
            self.failure='contact_overload'
            force=self._vase_forces()
            weights=np.maximum(0.,-np.sum(force*self.vertex_normals,axis=1))
            from ._force_task_utils import world_contact_vertices
            contact_center=(world_contact_vertices(self.vase)*weights[:,None]).sum(axis=0)/max(weights.sum(),1e-12)
            visual=show_scuff(self,contact_center,getattr(self,'_contact_tangent_motion',np.zeros(3)))
            self.metadata['glaze_damage']=dict(step=self.step_count,normal_force_N=load,
                trigger='instantaneous_contact_overload',
                force_threshold_N=float(self.params.get('overload_force_N',12.5)),
                excess_contact_work_J=self.glaze_damage_work_J,**visual)

        if (self.failure is None and self.phase_id==self.PHASE_POLICY
                and (self.step_count-self.policy_start_step)*self.cfg.sim.dt>=self.cfg.final_policy_timeout_seconds
                and not self.check_success()):
            self.failure='timeout'

    def check_early_stop(self):
        return self.failure is not None

    def _servo(self,position,q):
        m=self._robot_manager
        # Rendering runs every4ticks; refresh robot kinematics every control tick.
        m.robot.update(self.cfg.sim.dt)
        pos,quat=m.get_ee_pose_tensor();joints=m.robot.data.joint_pos[:,m._arm_ids]
        cmd=torch.tensor(np.r_[position,q],device=self.device,dtype=torch.float32).reshape(1,7)
        m._ik_controller.set_command(cmd)
        goal=m._ik_controller.compute(pos,quat,m.jacobian_b[:,:,m._arm_ids],joints);delta=goal-joints
        if not bool(torch.all(torch.isfinite(goal))) or float(delta.abs().max())>.08:
            self.failure='ik_discontinuity';return False
        limits=m.robot.data.soft_joint_pos_limits[:,m._arm_ids];goal=torch.clamp(goal,limits[...,0],limits[...,1])
        from ._force_task_utils import execute_joint_target
        executed,accepted=execute_joint_target(self,arm=goal[0],ticks=2)
        if accepted or not executed:return False
        # GT is used only as an independent safety/scoring monitor here.
        force=self._vase_forces()
        true_load=float(np.maximum(0,-np.sum(force*self.vertex_normals,axis=1)).sum())
        self.max_force=max(self.max_force,true_load)
        if true_load>float(self.params.get('overload_force_N',12.5)):
            self.failure='contact_overload';return False
        return self.plan_success and self.failure is None

    def _move_center(self,center,tag):
        q=self._robot_manager.get_gripper_center_pose().q
        return self.move(self.atom.move_to_pose(self._robot_manager.gripper_center_to_ee(Pose(center,q))),
            tag=tag,time_dilation_factor=.5,delay=False)

    def _surface_target(self,height,indent):
        # The pad bottom follows the actual upper wall; its +Z axis is outward.
        slope=np.gradient(self.vase_data['profile_r'],self.vase_data['profile_z'])
        dr=float(np.interp(height,self.vase_data['profile_z'],slope))
        radius=float(np.interp(height,self.vase_data['profile_z'],self.vase_data['profile_r']))
        angle=self.lane_angle
        rot=t3d.axangles.axangle2mat([0,1,0],-angle)@t3d.axangles.axangle2mat([1,0,0],-np.arctan(dr))
        point=self.base+self.vase_rot@np.array([-radius*np.sin(angle),radius*np.cos(angle),height])
        center=point+(.021-indent)*rot[:,2]
        position=center-rot@self.grip_offset
        quat=t3d.quaternions.mat2quat(rot@t3d.quaternions.quat2mat(self.contact_q))
        return position,quat

    def _play_once(self):
        if self.params.get('vase_review_scenario') in ('visual_only','visual_material_audit'):
            import carb
            settings=carb.settings.get_settings()
            keys=['/rtx/raytracing/fractionalCutoutOpacity','/rtx/translucency/enabled',
                '/rtx/reflections/enabled','/rtx/debug/onlyOpaqueRayFlags',
                '/rtx/renderMode','/rtx/material/translucencyAsOpacity']
            self.metadata['actual_render_settings']={key:settings.get(key) for key in keys}
            self.metadata['diagnostic_only']=True
            self.metadata['diagnostic_protocol']='unattempted scene appearance; not a physical failure example'
            self.delay(60,is_save=True)
            if self.params.get('vase_review_scenario')=='visual_material_audit':
                from ._force_task_utils import _material
                stage=self.soil_mesh.GetPrim().GetStage()
                path=self.scene.env_prim_paths[0]+'/vase_wiping_presentation/Looks/DiagnosticOpaqueSoil'
                mat=_material(stage,path,(.20,.09,.025),.94)
                shader=UsdShade.Shader(stage.GetPrimAtPath(path+'/Shader'))
                shader.CreateInput('opacity',Sdf.ValueTypeNames.Float).Set(1.)
                UsdShade.MaterialBindingAPI.Apply(self.soil_mesh.GetPrim()).Bind(mat)
                UsdGeom.Imageable(self.soil_mesh.GetPrim()).MakeVisible()
                self.metadata['visual_material_stages']=[dict(step=self.step_count,mode='constant_opaque')]
                self.delay(60,is_save=True)
                shader.GetInput('opacity').Set(.7)
                self.metadata['visual_material_stages'].append(dict(step=self.step_count,mode='constant_alpha07'))
                self.delay(60,is_save=True)
                shader.GetInput('opacity').Set(1.)
                points=np.asarray(self.soil_mesh.GetPointsAttr().Get())
                normals=np.asarray(self.soil_mesh.GetNormalsAttr().Get())
                self.soil_mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy((points+.001*normals).astype(np.float32)))
                self.metadata['visual_material_stages'].append(dict(step=self.step_count,mode='opaque_offset_diagnostic_plus1mm'))
                self.delay(60,is_save=True)
            self._finish()
            return
        self._perform_wipe_episode()
        # Persist terminal diagnostics even when an approach/grasp returns early.
        self._finish()

    def _perform_wipe_episode(self):
        grasp=self.foam_base+np.array([0,0,.031])
        for dz,tag in [(.07,'above_sponge'),(0.,'approach_sponge')]:
            if not self._move_center(grasp+[0,0,dz],tag):return
        self.move(self.atom.close_gripper(0.,depth_threshold=float(self.params.get('grasp_depth_mm',27.2))),tag='grasp_sponge',delay=False)
        if not self.plan_success:return
        for _ in range(16):
            if self._sponge_free:
                break
            self._step(is_save=True)
        if not self._sponge_free:
            self.failure='bilateral_grip_not_acquired'
            return
        self.delay(24)
        if not self._move_center(grasp+[0,0,.18],'lift_sponge'):return
        self.reference=read_rgb(self)
        import cv2
        for name,im in self.reference.items():cv2.imwrite(str(self.work/f'reference_{name}.png'),cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
        pose=self._robot_manager.get_ee_pose();center=self.sponge.vertex_positions.mean(axis=0)
        self.grip_offset=center-pose.p
        self.contact_q=pose.q.copy()
        height=.043
        for gap,tag in [(.040,'above_vase'),(.008,'approach_top')]:
            target,orientation=self._surface_target(height,-gap)
            if not self.move(self.atom.move_to_pose(Pose(target,orientation)),tag=tag,
                time_dilation_factor=.5,delay=False):return
        self.delay(12)
        self.reference=read_rgb(self)
        self.metadata['tactile_force_reference_step']=int(self.step_count)
        self.last_foam_center=None
        if self.controller=='marker_rgb':
            self.rgb_model=json.loads(Path(self.params['calibration']).read_text())
            contract=self.rgb_model.get('physical_contract',{})
            if (float(contract.get('gel_modulus_mpa',.10))!=float(self.params.get('gel_modulus_mpa',.10))
                    or contract.get('foam_core_modulus_mpa')!=self.params.get('foam_core_modulus_mpa')):
                raise ValueError('Vase RGB calibration does not match physical gel/core material')
        # A new reference/model must invalidate the same-step cached record;
        # otherwise the first servo iteration sees an old row without a
        # calibrated prediction and incorrectly reports missing RGB tracking.
        self._recorded_step=None
        self._record()
        indent=-.008
        levels=self.params.get('diagnostic_force_levels_N',[3.,7.,11.,7.]) if self.controller=='diagnostic_force' else [self.target]
        for level in levels:
            self.phase='press';stable=0
            for tick in range(0,1000,2):
                if tick%4==0:
                    row=self._record()
                    sensed=row.get('rgb_force_N') if self.controller=='marker_rgb' else row['normal_force_N']
                    if sensed is None:self.failure='rgb_tracking_lost';break
                    if row['normal_force_N']>float(self.params.get('overload_force_N',12.5)):self.failure='contact_overload';break
                    error=level-sensed;velocity=float(np.clip(error/2000,-.002,.002))
                    stable=stable+4 if abs(error)<.65 else 0
                    if stable>=48:break
                indent+=velocity*self.cfg.sim.dt*2
                if indent>.015:self.failure='contact_not_reached';break
                target,orientation=self._surface_target(height,indent)
                if not self._servo(target,orientation):break
            if self.failure or not self.plan_success:break
            if stable<48:self.failure='pressure_settle_timeout';break
        if self.failure or not self.plan_success:
            self._finish();return
        if self.params.get('vase_review_scenario')=='stationary_press':
            # A real bounded action example: hold the already acquired contact
            # without a tangential stroke. Any physical sliding still contributes
            # its ordinary dose; this branch never changes the soil or verdict.
            from ._force_task_utils import execute_joint_target
            manager=self._robot_manager
            held_joints=manager.robot.data.joint_pos[:,manager._arm_ids][0].clone()
            start=self.step_count
            clean_before=float(self.soil.fraction)
            self.phase='stationary_press_review'
            for _ in range(120):
                executed,accepted=execute_joint_target(self,arm=held_joints,ticks=2)
                if not executed or accepted:break
            self.metadata['review_stationary_press']=dict(
                start_step=start,end_step=self.step_count,
                cleaned_before=clean_before,cleaned_after=float(self.soil.fraction),
                action='same public8D joint target for at most2s; no tangential command')
            self._record();self._finish();return
        self.phase='wipe';direction=1;passes=0;lost=0
        lane_angles=tuple(float(x) for x in self.params.get('wipe_lane_angles_rad',[0.,.14]))
        if not lane_angles or any(abs(x)>.25 for x in lane_angles):
            raise ValueError('Wipe lanes must stay on the visible upper vase surface')
        for tick in range(0,int(self.params.get('max_wipe_steps',6600)),2):
            if tick%4==0:
                row=self._record()
                sensed=row.get('rgb_force_N') if self.controller=='marker_rgb' else row['normal_force_N']
                if sensed is None:self.failure='rgb_tracking_lost';break
                if row['normal_force_N']>float(self.params.get('overload_force_N',12.5)):self.failure='contact_overload';break
                lost=lost+4 if row['normal_force_N']<1 else 0
                if lost>120:self.failure='contact_lost';break
                if self.soil.fraction>=self.required_clean_fraction and passes>=1:break
                velocity=float(np.clip((self.target-sensed)/1000,-.006,.002))
            indent+=velocity*self.cfg.sim.dt*2
            if not -.015<indent<.015:self.failure='normal_travel_limit';break
            # Adjacent strokes remain on the upper-facing surface. A stiffer
            # pad can need both sides of the center stroke for full coverage.
            desired_angle=lane_angles[min(passes,len(lane_angles)-1)]
            turn=float(np.clip(desired_angle-self.lane_angle,-.05*self.cfg.sim.dt*2,.05*self.cfg.sim.dt*2))
            self.lane_angle+=turn
            lane_ready=abs(desired_angle-self.lane_angle)<.0005
            height+=direction*.0025*self.cfg.sim.dt*2*float(abs(self.target-sensed)<2. and lane_ready)
            if height>=.095 and direction>0:height=.095;direction=-1;passes+=1
            if height<=.038 and direction<0:height=.038;direction=1;passes+=1
            radius=float(np.interp(height,self.vase_data['profile_z'],self.vase_data['profile_r']))
            target,orientation=self._surface_target(height,indent)
            if not self._servo(target,orientation):break
        self._record()
        if self.soil.fraction<self.required_clean_fraction and self.failure is None:self.failure='soil_remaining'
        self._finish()

    def take_action(self,*args,**kwargs):
        if self._accepted_result is not None:
            return bool(self._accepted_result),bool(self._accepted_result)
        executed,success=super().take_action(*args,**kwargs)
        if success or self.failure:
            self._finish()
            return bool(executed and not self.failure),bool(self._accepted_result)
        return executed,success

    def _finish(self):
        if self._accepted_result is not None:return
        if not self.plan_success and self.failure is None:self.failure='motion_plan_failed'
        # Underpressure can be corrected while wiping; classify it only when
        # the bounded episode ends. A visually clean vase can still fail force
        # control, and its terminal reason must say so.
        if self.failure is None:
            if self.soil.fraction<self.required_clean_fraction:self.failure='soil_remaining'
            elif self.pressure_band_fraction<.95:self.failure='pressure_band_violation'
        self._accepted_result=bool(self.check_success())
        self._monitor_wiping=False
        self._set_phase(self.PHASE_TERMINAL,terminal_reason='success' if self._accepted_result else (self.failure or 'diagnostic'))
        if self.cfg.save_frequency>0 and self.mode!='eval_test':
            from ._force_task_utils import record_terminal_observation
            record_terminal_observation(self,'success' if self._accepted_result else (self.failure or 'incomplete'))
        # Freeze scoring before presentation moves. Their frames remain raw
        # observations, but TERMINAL excludes them from imitation actions.
        physical_clean=self.failure is None and self.soil.fraction>=self.required_clean_fraction and self.plan_success
        incomplete_review=(self.plan_success and self.failure in
            ('soil_remaining','pressure_band_violation','timeout','contact_lost'))
        if (self._accepted_result or (self.controller=='diagnostic_force' and physical_clean)
                or incomplete_review
                or (self.params.get('vase_review_scenario')=='stationary_press' and self.plan_success)):
            self.phase='reveal'
            self.metadata['terminal_surface_reveal']=dict(
                frozen_verdict='success' if self._accepted_result else self.failure,
                purpose='show actual remaining soil after the physical verdict; no cleaning credit')
            center=self._robot_manager.get_gripper_center_pose().p
            self._move_center(center+[0,0,.060],'reveal_clean_vase')
            if self.plan_success:self._move_center(np.array([.43,.12,.12]),'park_wiping_pad')
            self.delay(180,is_save=True)
        if self.failure in ('glaze_abraded','contact_overload') and self.plan_success:
            pose=self._robot_manager.get_ee_pose()
            self.move(self.atom.move_to_pose(Pose(pose.p+[0.,0.,.065],pose.q)),
                      tag='terminal_damage_review',time_dilation_factor=.5,delay=False)
            # Clear the wrist view after lifting. The vase axis projects along
            # image X; lateral world-Y motion exposes the contact patch beside
            # the held pad instead of leaving it hidden behind the pad.
            if self.plan_success:
                self.move(self.atom.move_to_pose(Pose(pose.p+[0.,.050,.065],pose.q)),
                          tag='terminal_damage_clear_view',time_dilation_factor=.5,delay=False)
            self.metadata['damage_review_motion']=dict(lift_m=.065,lateral_world_y_m=.050,
                phase='terminal',physical_result_frozen=True)
            self.delay(90,is_save=True)
        np.savez_compressed(self.work/'soil_final.npz',centers=self.soil.centers,
            initial=self.soil.initial,remaining=self.soil.remaining,area=self.soil.area,
            foam_points_m=self.sponge.vertex_positions,foam_rest_points_m=self.foam_data['points'])
        self.metadata['vase_wiping_final']=dict(failure=self.failure,cleaned_fraction=self.soil.fraction,
            required_clean_fraction=self.required_clean_fraction,
            peak_normal_force_N=self.max_force,controller=self.controller,
            pressure_band_N=self.params.get('pressure_band_N',[7.2,8.8]),pressure_band_fraction=self.pressure_band_fraction,
            pressure_scoring='fraction of physical sliding distance',wiping_distance_m=self.wiping_distance_m)
        self.metadata['trace_path']=str(self.work/'trace.jsonl')

    def check_success(self):
        if self._accepted_result is not None:return self._accepted_result
        return self.plan_success and self.failure is None and self.soil.fraction>=self.required_clean_fraction and self.max_force<=float(self.params.get('overload_force_N',12.5)) and getattr(self,'pressure_band_fraction',0.)>=.95

    def get_frame_shot(self,obs):
        return BaseTask.get_frame_shot(self,obs)

@configclass
class TaskCfg(BaseTaskCfg):
    step_lim=9600
    max_save_frames=2500
    video_size=(1120,320)

