"""Tighten a variable-thread bulb. Probing, geometry, damage and scoring live together."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import numpy as np
import transforms3d as t3d
from dataclasses import dataclass
from pathlib import Path
from ._base_task import *
from ._base_task import Pose
from ._force_task_utils import (
    ElasticResultant,
    ForceTaskScene,
    actor_contact_resultant,
    flow_rgb_features,
    image_features,
    predict_calibrated,
    read_rgb,
    sample_physics,
    task_parameters,
    tetra_faces,
    world_contact_vertices,
    write_tet_asset,
)


# Bulb damage

class BulbDamageStop(RuntimeError):
    """Unwind the scripted action immediately after an irreversible failure."""

SEAM_SCALE = np.array([.985, .985, .985])

def _clip_polyhedron(faces, normal, offset):
    """Clip a convex cell; preserve shared planar faces for conforming tets."""
    all_points=np.concatenate(faces)
    distance=all_points@normal-offset
    if distance.max()<=1e-12:return faces
    if distance.min()>=-1e-12:return []
    result=[];cap={}
    for face in faces:
        polygon=[]
        for a,b in zip(face,np.roll(face,-1,axis=0)):
            da,db=float(a@normal-offset),float(b@normal-offset)
            inside_a,inside_b=da<=1e-12,db<=1e-12
            if inside_a:polygon.append(a)
            if inside_a!=inside_b:
                point=a+(b-a)*da/(da-db)
                polygon.append(point);cap[tuple(np.round(point,12))]=point
            elif abs(da)<1e-12:
                cap[tuple(np.round(a,12))]=a
        unique=[]
        for point in polygon:
            if not unique or np.linalg.norm(point-unique[-1])>1e-12:unique.append(point)
        if len(unique)>1 and np.linalg.norm(unique[0]-unique[-1])<1e-12:unique.pop()
        if len(unique)>=3:result.append(np.asarray(unique))
    if len(cap)>=3:
        points=np.asarray(list(cap.values()));center=points.mean(0)
        u=points[0]-center;u/=np.linalg.norm(u)
        v=np.cross(normal,u);v/=np.linalg.norm(v)
        angle=np.arctan2((points-center)@v,(points-center)@u)
        result.append(points[np.argsort(angle)])
    return result

def _clip_tets(points,tets,planes):
    vertices=list(points.copy());indices={tuple(np.round(p,12)):i for i,p in enumerate(points)};cells=[]
    def vertex(point):
        key=tuple(np.round(point,12))
        if key not in indices:
            indices[key]=len(vertices);vertices.append(point.copy())
        return indices[key]
    xyz=points[tets]
    possible=np.ones(len(tets),bool);whole=np.ones(len(tets),bool)
    for normal,offset in planes:
        d=xyz@normal-offset
        possible &= d.min(axis=1)<1e-12
        whole &= d.max(axis=1)<=1e-12
    cells=tets[possible & whole].tolist()
    for source in tets[possible & ~whole]:
        poly=[points[source[list(face)]] for face in [(0,2,1),(0,1,3),(0,3,2),(1,2,3)]]
        for normal,offset in planes:
            poly=_clip_polyhedron(poly,normal,offset)
            if not poly:break
        if not poly:continue
        unique={tuple(np.round(p,12)):p for face in poly for p in face}
        center=np.mean(list(unique.values()),axis=0);middle=vertex(center)
        for face in poly:
            keys=[tuple(np.round(p,12)) for p in face]
            face=np.roll(face,-min(range(len(keys)),key=keys.__getitem__),axis=0)
            ids=[vertex(p) for p in face]
            for i in range(1,len(ids)-1):
                tri=face[[0,i,i+1]]
                det=float(np.linalg.det((tri-center).T))
                if abs(det)<=1e-20:continue
                a,b,c=ids[0],ids[i],ids[i+1]
                cells.append([middle,a,b,c] if det>0 else [middle,a,c,b])
    if not cells:return np.empty((0,3)),np.empty((0,4),dtype=np.int32)
    p=np.asarray(vertices);t=np.asarray(cells,dtype=np.int32)
    used,inverse=np.unique(t,return_inverse=True)
    return p[used],inverse.reshape(-1,4).astype(np.int32)

def apply_fragment_glass_material(path, opacity=1.0):
    """Use a visible frosted-glass surface on the existing physical shell."""
    from pxr import Gf,Sdf,Usd,UsdGeom,UsdShade
    stage=Usd.Stage.Open(str(path))
    material=UsdShade.Material.Define(stage,'/Object/Looks/FracturedGlass')
    shader=UsdShade.Shader.Define(stage,'/Object/Looks/FracturedGlass/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.89,.85,.70))
    shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.23)
    shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(0.)
    shader.CreateInput('opacity',Sdf.ValueTypeNames.Float).Set(float(opacity))
    shader.CreateInput('ior',Sdf.ValueTypeNames.Float).Set(1.48)
    shader.CreateInput('clearcoat',Sdf.ValueTypeNames.Float).Set(.8)
    shader.CreateInput('clearcoatRoughness',Sdf.ValueTypeNames.Float).Set(.08)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    mesh=UsdGeom.Mesh(stage.GetPrimAtPath('/Object/body'))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    stage.GetRootLayer().Save()

def mesh_volume(points,tets):
    return float(np.linalg.det(np.stack(
        [points[tets[:,i]]-points[tets[:,0]] for i in (1,2,3)],axis=-1)).sum()/6.)

def hollow_shell_fragments(points,tets):
    """Mass-matched hollow damage shapes for the intact collision proxy.

    Glass follows the original bulb's radial profile using conforming annular
    cells. The metal base retains unmodified original tetrahedra below the
    neck cut. Removed proxy-core mass is assigned to the glass shell.
    """
    from ._force_task_utils import tetra_faces
    points=np.asarray(points,float);tets=np.asarray(tets,np.int32)
    base_tets=tets[np.max(points[tets,2],axis=1)<.0409]
    used,inverse=np.unique(base_tets,return_inverse=True)
    base=(points[used],inverse.reshape(-1,4).astype(np.int32))
    pieces=[base];densities=[500.]
    glass_mass=(mesh_volume(points,tets)-mesh_volume(*base))*500.
    zs=np.array([.041,.046,.050,.055,.060,.065,.070,.075,.080,.084,.086,.087,.088])
    radii=np.array([.014,.0165,.018,.0192,.020,.0199,.0192,.0178,.015,.010,.005,.0025,0.])
    for sector in range(8):
        angles=np.linspace(sector*np.pi/4-.17,(sector+1)*np.pi/4-.17,13)
        vertices=[]
        for z,radius in zip(zs,radii):
            inner=max(0.,radius-.0015) if z<.087 else 0.
            for angle in angles:
                for rr in [inner,radius]:
                    vertices.append([rr*np.cos(angle),rr*np.sin(angle),z])
        p=np.asarray(vertices);cells=[]
        def idx(k,i,j):return (k*len(angles)+i)*2+j
        for k in range(len(zs)-1):
            for i in range(len(angles)-1):
                a,b,c,d=[idx(k,i+di,j) for di,j in [(0,0),(1,0),(0,1),(1,1)]]
                e,f,g,h=[idx(k+1,i+di,j) for di,j in [(0,0),(1,0),(0,1),(1,1)]]
                cells.extend([(a,b,d,h),(a,d,c,h),(a,c,g,h),(a,g,e,h),(a,e,f,h),(a,f,b,h)])
        t=np.asarray(cells,np.int32)
        _,first,inverse=np.unique(np.round(p,12),axis=0,return_index=True,return_inverse=True)
        p=p[first];t=inverse[t]
        det=np.linalg.det(np.stack([p[t[:,i]]-p[t[:,0]] for i in (1,2,3)],axis=-1))
        t=t[np.abs(det)>1e-18];det=det[np.abs(det)>1e-18]
        negative=det<0;t[negative,1],t[negative,2]=t[negative,2].copy(),t[negative,1].copy()
        used,inverse=np.unique(t,return_inverse=True);p=p[used];t=inverse.reshape(-1,4).astype(np.int32)
        faces=tetra_faces(t)
        edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
        _,counts=np.unique(edges,axis=0,return_counts=True)
        if not np.all(counts==2):raise ValueError("hollow glass shell must be closed and manifold")
        density=glass_mass/8./mesh_volume(p,t)
        if not 500.<density<6000.:raise ValueError("implausible effective glass fragment density")
        pieces.append((p,t));densities.append(density)
    return pieces,densities

def partition_bulb(points,tets,cut_height=.045):
    """Cut original tetrahedra at planar boundaries without volume erosion."""
    points=np.asarray(points,dtype=float);tets=np.asarray(tets,dtype=np.int32)
    pieces=[_clip_tets(points,tets,[(np.array([0.,0.,1.]),cut_height)])]
    for i in range(8):
        a=i*2*np.pi/8-.17;b=(i+1)*2*np.pi/8-.17
        planes=[(np.array([0.,0.,-1.]),-cut_height),
                (np.array([np.sin(a),-np.cos(a),0.]),0.),
                (np.array([-np.sin(b),np.cos(b),0.]),0.)]
        pieces.append(_clip_tets(points,tets,planes))
    def volume(p,t):
        return np.linalg.det(np.stack([p[t[:,i]]-p[t[:,0]] for i in (1,2,3)],axis=-1)).sum()/6.
    original=volume(points,tets);total=sum(volume(p,t) for p,t in pieces)
    if abs(total-original)>abs(original)*1e-5:
        raise ValueError(f"clipped bulb volume mismatch: {original} vs {total}")
    return pieces

def create_fragments(task):
    from pxr import Usd
    from ._base_task import Pose, UipcObjectCfg
    from ._force_task_utils import tetra_faces
    from ._force_task_utils import write_tet_asset
    source = Path(task.bulb_asset_path).resolve()
    stage = Usd.Stage.Open(str(source))
    mesh = next(p for p in stage.Traverse() if p.HasAttribute("tet_indices"))
    points = np.asarray(mesh.GetAttribute("tet_points").Get(), dtype=float)
    tets = np.asarray(mesh.GetAttribute("tet_indices").Get(), dtype=np.int32).reshape(-1, 4)
    task.bulb_fragments = []
    if task.params.get('glass_fragment_mesh_cache'):
        import hashlib
        cache=np.load(task.params['glass_fragment_mesh_cache'])
        if (str(cache['source_sha256'])!=hashlib.sha256(source.read_bytes()).hexdigest()
                or str(cache['model'])!=task.params.get('glass_fragment_model','solid_partition')
                or int(cache['geometry_version'])!=2):
            raise ValueError("bulb fragment cache does not match source/model geometry")
        pieces=[(cache[f'points_{i}'],cache[f'tets_{i}']) for i in range(9)]
        densities=cache['densities'].tolist()
    elif task.params.get('glass_fragment_model')=='hollow_shell':
        pieces,densities=hollow_shell_fragments(points,tets)
    else:
        pieces=partition_bulb(points,tets);densities=[500.]*len(pieces)
    densities=[rho/float(np.prod(SEAM_SCALE)) for rho in densities]
    task.bulb_fragment_model_info=dict(
        model=task.params.get('glass_fragment_model','solid_partition'),
        original_mass_kg=mesh_volume(points,tets)*500.,
        fragment_mass_after_clearance_kg=sum(mesh_volume(p,t)*rho*float(np.prod(SEAM_SCALE)) for (p,t),rho in zip(pieces,densities)),
        clearance_scale=SEAM_SCALE.tolist(),
        fragment_density_kg_m3=densities)
    for index, (p, t) in enumerate(pieces):
        path = write_tet_asset(task.work/"fragments"/f"bulb_{index}.usda",
                              p, t, tetra_faces(t),
                              (.42, .30, .10) if index == 0 else (.92, .76, .30))
        if index > 0 and task.params.get('glass_fragment_material','frosted_glass') == 'frosted_glass':
            apply_fragment_glass_material(path, opacity=task.params.get('glass_fragment_opacity',1.0))
        actor = task._actor_manager.add_from_usd_file(
            name=f"bulb_fragment_{index}", asset_path=path, visual_asset_path=path,
            pose=Pose([2., -.5-index*.15, .3], [1., 0., 0., 0.]),
            constitution_cfg=UipcObjectCfg.AffineBodyConstitutionCfg(m_kappa=100.),
            density=densities[index], show_physics_mesh=False, keep_constrained=True)
        from uipc import builtin,view
        geo=actor.uipc_meshes[0]
        view(geo.instances().find(builtin.is_constrained))[:]=1
        view(geo.instances().find(builtin.aim_transform))[:]=np.eye(4)
        task.bulb_fragments.append(actor)

def fracture(task, row):
    from ._base_task import Pose
    from ._force_task_utils import replace_squeezed_chip
    from ._force_task_utils import set_actor_visible
    from pxr import Gf, UsdGeom
    import omni.usd
    if task.fractured:
        return
    pose = task.bulb.get_pose()
    handoff = replace_squeezed_chip(
        task.uipc_sim, task.bulb, task.bulb_fragments,
        Pose([2., 1., .4], [1., 0., 0., 0.]),
        task._bulb_previous_geometry, task.cfg.sim.dt, seam_scale=SEAM_SCALE)
    stage = omni.usd.get_context().get_stage()
    set_actor_visible(stage, task.bulb, False)
    for fragment in task.bulb_fragments:
        for path in fragment.cfg.visual_prim_paths:
            for op in UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    op.Set(Gf.Vec3d(*(np.asarray(op.Get())*SEAM_SCALE)))
        set_actor_visible(stage, fragment, True)
    task._actor_manager.update(dt=0.)
    task.fractured = True
    task.metadata["bulb_fracture"] = dict(
        step=task.step_count, pose=pose.tolist(),
        threshold_per_pad_N=float(task.params.get("glass_break_force_N", 170.)),
        pad_normal_force_N=row["pad_normal_force_N"],
        pieces=len(task.bulb_fragments), handoff=handoff,
        geometry=task.bulb_fragment_model_info,
        model="load-threshold damage approximation; free fragments, no added kick")


# Bulb filament

FILAMENT_EMISSION_RGB=(56.,22.,5.2)

FILAMENT_LOCAL_LIGHT_INTENSITY=36.

def update_filament(task,pose=None):
    from pxr import Gf,UsdGeom
    if pose is None:pose=task.bulb.get_pose()
    for root,translate,rotate in getattr(task,"_filament_roots",[]):
        translate.Set(Gf.Vec3d(*pose.p))
        rotate.Set(Gf.Quatf(float(pose.q[0]),Gf.Vec3f(*map(float,pose.q[1:]))))
        UsdGeom.Imageable(root).GetVisibilityAttr().Set("invisible" if task.fractured else "inherited")

def set_filament_lit(task,lit):
    from pxr import Gf,Sdf,UsdShade,UsdLux,UsdGeom
    import omni.usd
    if lit and (task.phase_id!=task.PHASE_TERMINAL or not task._accepted_result):
        raise RuntimeError("Only successful terminal acceptance lights the filament")
    stage=omni.usd.get_context().get_stage()
    update_filament(task)
    for env in task.scene.env_prim_paths:
        shader=UsdShade.Shader(stage.GetPrimAtPath(env+"/bulb_filament/Looks/Tungsten/Surface"))
        if not shader:continue
        shader.CreateInput("emissiveColor",Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*FILAMENT_EMISSION_RGB) if lit else Gf.Vec3f(0.))
        light=UsdLux.SphereLight.Define(stage,env+"/bulb_acceptance_light")
        light.CreateRadiusAttr(.0015);light.CreateIntensityAttr(FILAMENT_LOCAL_LIGHT_INTENSITY if lit else 0.)
        light.CreateColorAttr(Gf.Vec3f(1.,.65,.28))
        xform=UsdGeom.Xformable(light)
        op=next((o for o in xform.GetOrderedXformOps() if o.GetOpType()==UsdGeom.XformOp.TypeTranslate),None)
        if op is None:op=xform.AddTranslateOp()
        op.Set(Gf.Vec3d(*task.bulb.get_pose().add_bias([0.,0.,.064]).p))
    if lit:
        task.metadata["bulb_light"]=dict(step=task.step_count,phase="terminal",accepted=True,
            source="internal coiled filament",glass_emission=False,action_time_light=False,
            emissive_rgb=list(FILAMENT_EMISSION_RGB),
            local_light_intensity=FILAMENT_LOCAL_LIGHT_INTENSITY,
            appearance_version="brighter_filament_v2")


# Bulb damage

def light_after_submission(task):
    set_filament_lit(task,True)

def reset_lighting(task):
    set_filament_lit(task,False)


# Bulb filament

def create_filament(task):
    from pxr import Gf,Sdf,UsdGeom,UsdShade
    import omni.usd
    stage=omni.usd.get_context().get_stage()
    task._filament_roots=[]
    for env in task.scene.env_prim_paths:
        path=env+"/bulb_filament"
        root=UsdGeom.Xform.Define(stage,path)
        translate=root.AddTranslateOp();rotate=root.AddOrientOp()
        task._filament_roots.append((root,translate,rotate))
        def material(name,color,metal=.0):
            mat=UsdShade.Material.Define(stage,path+"/Looks/"+name)
            shader=UsdShade.Shader.Define(stage,str(mat.GetPath())+"/Surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor",Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput("roughness",Sdf.ValueTypeNames.Float).Set(.35)
            shader.CreateInput("metallic",Sdf.ValueTypeNames.Float).Set(metal)
            shader.CreateInput("emissiveColor",Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.))
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),"surface")
            return mat
        coilmat=material("Tungsten",(.10,.08,.06),.55)
        supportmat=material("Support",(.19,.19,.18),.8)
        t=np.linspace(0,1,241)
        coil=np.column_stack((-.008+.016*t,.00065*np.cos(t*16*np.pi),.064+.00065*np.sin(t*16*np.pi)))
        for name,points,width,mat in [
            ("coil",coil,.00020,coilmat),
            ("support_left",[[-.003,0,.041],[-.003,0,.054],[-.008,0,.0639]],.00030,supportmat),
            ("support_right",[[.003,0,.041],[.003,0,.054],[.008,0,.0639]],.00030,supportmat)]:
            curve=UsdGeom.BasisCurves.Define(stage,path+"/"+name)
            curve.CreateTypeAttr("linear");curve.CreateCurveVertexCountsAttr([len(points)])
            curve.CreatePointsAttr([Gf.Vec3f(*map(float,p)) for p in points]);curve.CreateWidthsAttr([width])
            curve.SetWidthsInterpolation("constant")
            UsdShade.MaterialBindingAPI.Apply(curve.GetPrim()).Bind(mat)
    update_filament(task,pose=task.bulb.init_pose)


# Bulb observer contract

def check_bulb_observer_contract(parameters, model):
    if parameters.get('controller')!='marker_rgb':
        return
    contract=model.get('physical_contract',{})
    keys=('gel_modulus_mpa','grip_depth_mm','friction_eps_velocity_m_s',
          'newton_velocity_tol_m_s','newton_max_iter','thread_friction')
    mismatches=[key for key in keys if key not in contract or key not in parameters
                or not math.isclose(float(contract[key]),float(parameters[key]),rel_tol=0.,abs_tol=1e-12)]
    if mismatches:
        raise ValueError('RGB bulb observer does not match physical parameters: '+', '.join(mismatches)+
            '. Use a matching validated model; diagnostic_force is explicitly a physical verification mode.')


# Bulb probe estimation

def estimate_thread_direction(records, *, angle_deg=4., min_lead_m=.0035, max_lead_m=.013, lead_estimator="opposite_difference"):
    if lead_estimator not in ("opposite_difference","forward_response"):
        raise ValueError("unknown tactile lead estimator")
    if not math.isfinite(angle_deg) or angle_deg<=0:
        raise ValueError("probe angle must be positive and finite")
    if not (0<min_lead_m<max_lead_m and math.isfinite(max_lead_m)):
        raise ValueError("invalid public lead bounds")
    if len(records)!=2 or {r.get("world_yaw_sign") for r in records}!={-1,1}:
        raise ValueError("need one response for each world yaw direction")
    advances=[float(r["inferred_thread_advance_m"]) for r in records]
    if not all(math.isfinite(v) for v in advances):
        return dict(resolved=False,reason="nonfinite_response",estimated_lead_m=None)
    separation=abs(advances[0]-advances[1])
    lead=(max(advances) if lead_estimator=="forward_response" else separation*.5)*360./angle_deg
    selected=records[advances.index(max(advances))]["world_yaw_sign"]
    result=dict(resolved=False,world_yaw_sign=None,candidate_world_yaw_sign=int(selected),estimated_lead_m=lead,
                unclipped_estimated_lead_m=lead,signal_separation_m=separation,
                public_lead_bounds_m=[min_lead_m,max_lead_m],
                lead_estimator=("positive inferred advance after reciprocal direction test; no clipping"
                                if lead_estimator=="forward_response" else
                                "half difference of opposite axial responses; no clipping"),
                opposite_difference_lead_m=separation*.5*360./angle_deg)
    if lead_estimator=="forward_response" and not (min(advances)<-3e-6 and max(advances)>3e-6):
        result["reason"]="nonreciprocal_probe_response"
    elif separation<.000006:
        result["reason"]="insufficient_opposite_response"
    elif not min_lead_m<=lead<=max_lead_m:
        result["reason"]="implausible_lead_response"
    else:
        result.update(resolved=True,reason="resolved",world_yaw_sign=int(selected))
    return result

def rotation_probe_descent(value):
    value=float(value)
    if not math.isfinite(value) or not 0.<=value<=.00015:
        raise ValueError("rotational probe descent must be within0..0.15mm")
    return value


# Bulb probe lifecycle

@dataclass
class ReversibleThreadProbe:
    track_grasp_origin: bool = False
    origin_degrees: float | None = None
    origin_advance_m: float | None = None
    origin_locked: bool = False

    def observe_grasp_origin(self, world_degrees, advance_m, *, bilateral_grip):
        """Track axial settling before the first angular excursion, once only.

        Scoring uses physical pose/contact, never expert phase or requested
        thread direction. The angular reference is fixed at first grasp;
        axial settling is tracked only while still within the home angle.
        Once rotation leaves that region, neither reference can change,
        including during backoff, retries, or regrasp.
        """
        if not self.track_grasp_origin or self.origin_locked:
            return False
        angle=float(world_degrees);advance=float(advance_m)
        if not math.isfinite(angle) or not math.isfinite(advance):
            raise ValueError("Non-finite physical probe origin")
        if self.origin_degrees is None:
            if not bilateral_grip:return False
            self.origin_degrees=angle
        if abs(angle-self.origin_degrees)>self.home_degrees:
            self.origin_locked=True
            return True
        if bilateral_grip:self.origin_advance_m=advance
        return False

    minimum_degrees: float = 2.
    maximum_degrees: float = 6.
    home_degrees: float = .6
    home_advance_m: float = .0002
    positive_complete: bool = False
    negative_complete: bool = False
    excursion_min: float = 0.
    excursion_max: float = 0.
    formal_reference_degrees: float | None = None
    formal_reference_advance_m: float | None = None

    @property
    def complete(self):
        return self.positive_complete and self.negative_complete

    def update(self, world_degrees, advance_m):
        if self.complete:return True
        if self.track_grasp_origin:
            if self.origin_degrees is None or self.origin_advance_m is None:return False
            angle=float(world_degrees)-self.origin_degrees
            local_advance=float(advance_m)-self.origin_advance_m
        else:
            angle=float(world_degrees)
            local_advance=float(advance_m)
        self.excursion_min=min(self.excursion_min,angle)
        self.excursion_max=max(self.excursion_max,angle)
        at_home=abs(angle)<=self.home_degrees and abs(local_advance)<=self.home_advance_m
        if at_home:
            if (self.minimum_degrees<=self.excursion_max<=self.maximum_degrees
                    and self.excursion_min>=-self.home_degrees):
                self.positive_complete=True
            if (-self.maximum_degrees<=self.excursion_min<=-self.minimum_degrees
                    and self.excursion_max<=self.home_degrees):
                self.negative_complete=True
            self.excursion_min=self.excursion_max=0.
        if self.complete:
            self.formal_reference_degrees=float(world_degrees)
            self.formal_reference_advance_m=float(advance_m)
        return self.complete

    def formal_motion(self, world_degrees, advance_m):
        if not self.complete or self.formal_reference_degrees is None:
            return 0.,0.
        return (float(world_degrees)-self.formal_reference_degrees,
                float(advance_m)-self.formal_reference_advance_m)


# Bulb probing

def _move_probe(task, start, yaw, descent, ticks=48):
    from ._base_task import Pose
    before=task._robot_manager.get_gripper_center_pose()
    destination=start.p-np.array([0.,0.,descent])
    qend=t3d.quaternions.qmult(t3d.quaternions.axangle2quat([0.,0.,1.],yaw),start.q)
    for tick in range(2,ticks+1,2):
        a=tick/ticks
        q=t3d.quaternions.qmult(t3d.quaternions.axangle2quat([0.,0.,1.],
            task._get_actor_yaw_delta_deg(Pose(destination,qend),before)*np.pi/180*a),before.q)
        if not task._ik_to_center(Pose(before.p+(destination-before.p)*a,q)):return False
    return True

def axial_image_displacement(task):
    """Signed pad shear from the same raw-image basis as the safe axial probe."""
    response,tracking=image_features(task._thread_shear_reference,read_rgb(task))
    equivalent=.00015*float(np.dot(response,task._axial_image_basis)/task._axial_image_strength)
    if not np.isfinite(equivalent):
        raise ValueError("non-finite axial tactile response")
    return equivalent,tracking

def probe_both_directions(task):
    """Estimate tightening sign and lead from reversible axial shear response.

    Neither the randomized thread specification nor physical resultants are
    inputs. The fixed small probe motions and raw-image response are logged.
    """
    start=task._robot_manager.get_gripper_center_pose()
    reference=read_rgb(task)
    depth=.00015
    rotation_descent=rotation_probe_descent(task.params.get("probe_rotation_descent_m",depth))
    task.phase="probe_axial_compliance"
    if not _move_probe(task,start,0.,depth):return False
    task.delay(8,is_save=True)
    basis,details=image_features(reference,read_rgb(task))
    strength=float(np.dot(basis,basis))
    if not _move_probe(task,start,0.,0.):return False
    task.delay(12,is_save=True)
    if not np.isfinite(strength) or strength<.0025:
        task.failure="probe_tactile_response_missing";return False
    records=[]
    angle=np.deg2rad(4.)
    for sign in (-1,1):
        task.phase="probe_clockwise" if sign<0 else "probe_counterclockwise"
        reference=read_rgb(task)
        if not _move_probe(task,start,sign*angle,rotation_descent):return False
        task.delay(8,is_save=True)
        response,tracking=image_features(reference,read_rgb(task))
        axial_equivalent=depth*float(np.dot(response,basis)/strength)
        inferred_advance=rotation_descent-axial_equivalent
        records.append(dict(world_yaw_sign=sign,probe_angle_deg=4.,
                            commanded_descent_m=rotation_descent,axial_equivalent_m=axial_equivalent,
                            inferred_thread_advance_m=inferred_advance,
                            image_feature_energy=float(np.dot(response,response)),tracking=tracking,
                            step=task.step_count,physical_diagnostic=dict(task._measure())))
        task.phase="probe_backoff"
        if not _move_probe(task,start,0.,0.):return False
        task.delay(12,is_save=True)
    # Greater positive axial advance under the same small downward command
    # identifies the accommodating direction. Wrong-direction resistance alone
    # is not a task failure.
    decision=estimate_thread_direction(records,angle_deg=4.,
        lead_estimator=task.params.get("probe_lead_estimator","opposite_difference"))
    task.metadata["thread_probe"]=records
    task.metadata["thread_probe_decision"]=dict(decision,
        axial_image_basis=basis.tolist(),axial_probe_depth_m=depth,
        rotation_probe_descent_m=rotation_descent,
        inputs="raw tactile image flow and issued safe probe motions",
        returned_gripper_pose=task._robot_manager.get_gripper_center_pose().tolist(),
        requested_return_pose=start.tolist(),completed_step=task.step_count)
    # Preserve the decision before a long formal rollout, including unresolved
    # attempts. This diagnostic sidecar is never an observation or policy input.
    import json
    def json_numeric(value):
        if isinstance(value,np.ndarray):return value.tolist()
        if isinstance(value,np.generic):return value.item()
        raise TypeError(type(value).__name__)
    (task.work/"thread_probe_diagnostics.json").write_text(json.dumps(dict(
        probes=records,decision=task.metadata["thread_probe_decision"]),
        indent=2,default=json_numeric))
    if not decision["resolved"]:
        task.failure="thread_direction_unresolved"
        return False
    task.controller_spin_sign=int(decision["world_yaw_sign"])
    task.controller_lead_m=float(decision["estimated_lead_m"])
    task._axial_image_basis=basis
    task._axial_image_strength=strength
    task._thread_shear_reference=read_rgb(task)
    return True


# Bulb threads

@dataclass(frozen=True)
class ThreadSpec:
    handedness: int
    lead_m: float
    initial_depth_m: float = .014
    seat_top_m: float = .022

    def __post_init__(self):
        if self.handedness not in (-1, 1):
            raise ValueError("Thread handedness must be -1 or +1")
        if not .0035 <= self.lead_m <= .013:
            raise ValueError("Lead outside supported physical mesh resolution")
        if not .010 <= self.initial_depth_m <= .020:
            raise ValueError("Start must leave substantial real screw travel")

    @property
    def initial_yaw_rad(self):
        return -self.handedness*2*np.pi*self.initial_depth_m/self.lead_m

    @property
    def nominal_free_travel_m(self):
        return .055-.006-self.initial_depth_m-self.seat_top_m

    def signed_progress_turns(self, world_yaw_deg):
        return -self.handedness*float(world_yaw_deg)/360.

    def expected_advance(self, world_yaw_deg):
        return self.signed_progress_turns(world_yaw_deg)*self.lead_m

    def cache_key(self):
        return hashlib.sha256(json.dumps(self.__dict__,sort_keys=True).encode()).hexdigest()[:16]

def thread_profile_half_width(spec, *, preserve_flank_width=False):
    """Keep the axial flank width of the established two-turn thread.

    A fixed angular width makes short-lead flanks steeper and their normal
    gap smaller than the unchanged contact barrier. The widened matched
    ridge/groove preserves physical clearance; it does not alter lead.
    """
    base=np.deg2rad(58.)
    return max(base,base*.00615/spec.lead_m) if preserve_flank_width else base

def generate_variant(spec, output_dir, *, preserve_boundary=False, preserve_flank_width=False):
    """Generate one matched pair; never alter the legacy two-turn source files."""
    import tetgen
    from pxr import Gf,Sdf,Usd,UsdGeom,UsdShade,Vt
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    manifest=output_dir/"manifest.json"
    expected={"geometry_version":3 if preserve_flank_width else (2 if preserve_boundary else 1),**spec.__dict__}
    if preserve_flank_width:
        expected.update(profile_mode="minimum_axial_flank_width",profile_half_width_rad=thread_profile_half_width(spec,preserve_flank_width=True))
    if manifest.exists():
        existing=json.loads(manifest.read_text())
        if existing.get("spec")==expected and all((output_dir/n).is_file() for n in ("bulb.usda","socket.usda")):
            return str(output_dir/"bulb.usda"),str(output_dir/"socket.usda")
        raise ValueError("Existing thread cache differs from requested specification")
    generator_path=Path(__file__).resolve().parents[1]/"scripts/generate_screw_light_bulb_assets.py"
    loader=importlib.util.spec_from_file_location("bulb_mesh_source",generator_path)
    source=importlib.util.module_from_spec(loader);loader.loader.exec_module(source)
    source.PITCH=spec.lead_m
    source.SIDES=64
    source.ANGLES=np.linspace(0,2*np.pi,source.SIDES,endpoint=False)
    if preserve_flank_width:
        half_width=thread_profile_half_width(spec,preserve_flank_width=True)
        def clearance_profile(angle):
            wrapped=np.arctan2(np.sin(angle),np.cos(angle))
            return np.clip(1.-np.abs(wrapped)/half_width,0.,1.)
        source.wrapped_thread_profile=clearance_profile
    def bulb_radii(z):
        phase=spec.handedness*source.ANGLES-2*np.pi*z/spec.lead_m
        profile=source.wrapped_thread_profile(phase)
        envelope=min(float(np.clip((z-source.THREAD_START_Z)/.0015,0,1)),
                     float(np.clip((source.THREAD_END_Z-z)/.0015,0,1)))
        return source.THREAD_CORE_RADIUS+(source.THREAD_MAJOR_RADIUS-source.THREAD_CORE_RADIUS)*profile*envelope
    def socket_radii(z):
        phase=spec.handedness*source.ANGLES-2*np.pi*z/spec.lead_m+2*np.pi*(.055-.006)/spec.lead_m
        return source.SOCKET_MINOR_RADIUS+(source.SOCKET_GROOVE_RADIUS-source.SOCKET_MINOR_RADIUS)*source.wrapped_thread_profile(phase)
    source.bulb_thread_radii=bulb_radii;source.socket_thread_radii=socket_radii
    # Preserve at least the source z resolution. At the finest lead this gives
    # more than 12 samples per pitch, while both contact and visuals share it.
    records={}
    colors={"bulb_contact_metal":(.08,.09,.09),"bulb_thread_brass":(.58,.34,.08),
            "bulb_neck_metal":(.25,.28,.29),"bulb_glass_warm":(.94,.97,1.),
            "socket_ceramic":(.72,.76,.78),"socket_rim_brass":(.62,.38,.09),
            "socket_thread_brass":(.30,.20,.06)}
    for kind,builder in (("bulb",source.build_bulb),("socket",source.build_socket)):
        mesh=builder();mesh.validate(kind)
        p=np.asarray(mesh.vertices,np.float64);f=np.asarray(mesh.faces,np.int32)
        result=tetgen.TetGen(p,f).tetrahedralize(quality=False,nobisect=True) if preserve_boundary else tetgen.TetGen(p,f).tetrahedralize()
        tp=np.asarray(result[0],np.float64);tt=np.asarray(result[1],np.int32).reshape(-1,4)
        d=np.linalg.det(tp[tt[:,1:]]-tp[tt[:,:1]])/6
        negative=d<0
        if negative.any():tt[negative,1],tt[negative,2]=tt[negative,2].copy(),tt[negative,1].copy()
        if not np.all(np.abs(d)>1e-18):raise ValueError("Degenerate thread tetrahedron")
        faces={}
        for a,b,c,d0 in tt:
            for face in ((b,c,d0),(a,d0,c),(a,b,d0),(a,c,b)):
                key=tuple(sorted(map(int,face)))
                if key in faces:faces[key]=None
                else:faces[key]=tuple(map(int,face))
        tf=np.array([v for v in faces.values() if v is not None],np.int32)
        stage=Usd.Stage.CreateInMemory();root=UsdGeom.Xform.Define(stage,"/Object")
        stage.SetDefaultPrim(root.GetPrim());UsdGeom.SetStageUpAxis(stage,"Z");UsdGeom.SetStageMetersPerUnit(stage,1.)
        visual=UsdGeom.Mesh.Define(stage,"/Object/body")
        visual.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(p.astype(np.float32)))
        visual.CreateFaceVertexCountsAttr([3]*len(f));visual.CreateFaceVertexIndicesAttr(f.ravel().tolist())
        visual.CreateSubdivisionSchemeAttr("none");visual.CreateDoubleSidedAttr(True)
        for name,points in (("tet_points",tp),("tet_surf_points",tp)):
            visual.GetPrim().CreateAttribute(name,Sdf.ValueTypeNames.Double3Array).Set(Vt.Vec3dArray.FromNumpy(points))
        visual.GetPrim().CreateAttribute("tet_indices",Sdf.ValueTypeNames.IntArray).Set(tt.ravel().tolist())
        visual.GetPrim().CreateAttribute("tet_surf_indices",Sdf.ValueTypeNames.IntArray).Set(tf.ravel().tolist())
        materials=np.asarray(mesh.face_materials)
        for material_name in sorted(set(mesh.face_materials)):
            material=UsdShade.Material.Define(stage,"/Object/Looks/"+material_name)
            shader=UsdShade.Shader.Define(stage,str(material.GetPath())+"/Surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor",Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colors[material_name]))
            glass="glass" in material_name
            shader.CreateInput("roughness",Sdf.ValueTypeNames.Float).Set(.08 if glass else .3)
            shader.CreateInput("metallic",Sdf.ValueTypeNames.Float).Set(0. if glass or "ceramic" in material_name else .65)
            if glass:
                shader.CreateInput("opacity",Sdf.ValueTypeNames.Float).Set(.16)
                shader.CreateInput("ior",Sdf.ValueTypeNames.Float).Set(1.48)
                shader.CreateInput("clearcoat",Sdf.ValueTypeNames.Float).Set(1.)
                shader.CreateInput("clearcoatRoughness",Sdf.ValueTypeNames.Float).Set(.06)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),"surface")
            subset=UsdGeom.Subset.Define(stage,"/Object/body/"+material_name)
            subset.CreateElementTypeAttr("face");subset.CreateFamilyNameAttr("materialBind")
            subset.CreateIndicesAttr(np.flatnonzero(materials==material_name).tolist())
            UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(material)
        output=output_dir/(kind+".usda");stage.GetRootLayer().Export(str(output))
        records[kind]={"vertices":len(p),"tets":len(tt),"physical_surface_faces":len(tf),"visual_faces":len(f),
                       "sha256":hashlib.sha256(output.read_bytes()).hexdigest(),"volume_m3":float(np.abs(d).sum())}
    manifest.write_text(json.dumps({"spec":expected,"meshes":records},indent=2))
    return str(output_dir/"bulb.usda"),str(output_dir/"socket.usda")

def sample_thread_spec(parameters):
    rng=np.random.default_rng(int(parameters.get("physics_seed",0))+71823)
    hand=int(parameters.get("thread_handedness",rng.choice([-1,1])))
    nominal=float(parameters.get("nominal_turns",rng.uniform(1.12,2.88)))
    if not 1.0 <= nominal <= 3.0:raise ValueError("Nominal turns must be in [1,3]")
    # Nominal terminal travel includes the existing contact barrier and seat
    # compliance. The scorer always uses actual signed angle and axial motion.
    lead=float(parameters.get("thread_lead_m",.0123/nominal))
    return ThreadSpec(hand,lead,float(parameters.get("initial_depth_m",.014)),
                      float(parameters.get("seat_top_m",.022)))


# Bulb visual

def update_bulb_surface(task,pose=None):
    from pxr import Gf,UsdGeom
    if not hasattr(task,"_bulb_surface_points"):return
    if pose is None:pose=task.bulb.get_pose()
    # Bulb is an affine rigid actor here. Static source-matched vertices and a
    # rigid transform give the same surface without rebuilding its RTX BLAS on
    # every rendered observation. Fracture still replaces it with free pieces.
    for root,translate,rotate in task._bulb_surface_roots:
        translate.Set(Gf.Vec3d(*map(float,pose.p)))
        rotate.Set(Gf.Quatf(float(pose.q[0]),Gf.Vec3f(*map(float,pose.q[1:]))))
        UsdGeom.Imageable(root).GetVisibilityAttr().Set("invisible" if task.fractured else "inherited")

def create_bulb_surface(task):
    from pxr import Gf,Sdf,Usd,UsdGeom,UsdShade,Vt
    from ._force_task_utils import _material
    import omni.usd
    source=Usd.Stage.Open(str(task.bulb_asset_path))
    source_mesh=UsdGeom.Mesh(source.GetPrimAtPath("/Object/body"))
    points=np.asarray(source_mesh.GetPointsAttr().Get(),float)
    faces=np.asarray(source_mesh.GetFaceVertexIndicesAttr().Get(),int).reshape(-1,3)
    stage=omni.usd.get_context().get_stage()
    task._bulb_surface_meshes=[];task._bulb_surface_points=points
    task._bulb_surface_roots=[]
    colors={"bulb_contact_metal":(.08,.09,.09),"bulb_thread_brass":(.58,.34,.08),
            "bulb_neck_metal":(.25,.28,.29)}
    task._bulb_visual_audit=[]
    for env in task.scene.env_prim_paths:
        root=env+"/bulb_surface"
        rootprim=UsdGeom.Xform.Define(stage,root)
        task._bulb_surface_roots.append((rootprim,rootprim.AddTranslateOp(),rootprim.AddOrientOp()))
        # Author the MDL network directly. Scene construction should not
        # activate Kit editor extensions, issue MovePrim or select materials.
        glass=UsdShade.Material.Define(stage,root+"/Looks/Glass")
        shader=UsdShade.Shader.Define(stage,root+"/Looks/Glass/Shader")
        shader.CreateImplementationSourceAttr().Set("sourceAsset")
        shader.SetSourceAsset(Sdf.AssetPath("OmniGlass.mdl"),"mdl")
        shader.SetSourceAssetSubIdentifier("OmniGlass","mdl")
        shader.CreateInput("glass_color",Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.97,.985,1.))
        shader.CreateInput("glass_ior",Sdf.ValueTypeNames.Float).Set(1.48)
        shader.CreateInput("depth",Sdf.ValueTypeNames.Float).Set(.001)
        shader.CreateInput("thin_walled",Sdf.ValueTypeNames.Bool).Set(True)
        shader.CreateInput("frosting_roughness",Sdf.ValueTypeNames.Float).Set(.12)
        shader.CreateInput("reflection_color",Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.,1.,1.))
        shader.CreateOutput("out",Sdf.ValueTypeNames.Token)
        glass.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(),"out")
        glass.CreateVolumeOutput("mdl").ConnectToSource(shader.ConnectableAPI(),"out")
        glass.CreateDisplacementOutput("mdl").ConnectToSource(shader.ConnectableAPI(),"out")
        for subset in UsdGeom.Subset.GetAllGeomSubsets(source_mesh):
            name=subset.GetPrim().GetName()
            indices=np.asarray(subset.GetIndicesAttr().Get(),int)
            selected=faces[indices]
            used,inverse=np.unique(selected,return_inverse=True)
            surface_points=points[used]
            selected=inverse.reshape(-1,3)
            mesh=UsdGeom.Mesh.Define(stage,root+"/"+name)
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(surface_points.astype(np.float32)))
            mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.array(
                [surface_points.min(0),surface_points.max(0)],np.float32)))
            mesh.CreateFaceVertexCountsAttr([3]*len(selected))
            mesh.CreateFaceVertexIndicesAttr(selected.ravel().tolist())
            mesh.CreateSubdivisionSchemeAttr("none");mesh.CreateDoubleSidedAttr(True)
            import trimesh
            normals=trimesh.Trimesh(surface_points,selected,process=False).vertex_normals.copy()
            mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
            mesh.SetNormalsInterpolation("vertex")
            if "glass" in name:mat=glass
            else:mat=_material(stage,root+"/Looks/"+name,colors[name],.28,.65)
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
            task._bulb_surface_meshes.append((mesh,normals))
            task._bulb_visual_audit.append(dict(name=name,face_count=len(selected),
                material=str(mat.GetPath()),source_asset=str(task.bulb_asset_path)))
    # The original imported visual is replaced with the identical source
    # surface, avoiding stale Fabric attributes and ambiguous subset shading.
    for path in task.bulb.cfg.visual_prim_paths:
        UsdGeom.Imageable(stage.GetPrimAtPath(path)).MakeInvisible()
    update_bulb_surface(task,pose=task.bulb.init_pose)


# Screw light bulb

THREAD_PITCH = 0.010


# Force task probe

def bulb_grid(task):
    """Hold physical axial load at each small-yaw pose before collecting RGB."""
    from .utils.transforms import Pose
    import transforms3d as t3d
    manager=task._robot_manager
    origin=manager.get_gripper_center_pose()
    forces=list(task.params.get('probe_axial_levels_N',[12.,24.]))
    angles=list(task.params.get('probe_angles_deg',[0.,8.,-8.,0.]))
    if any(not 5<=v<=40 for v in forces) or any(abs(v)>12 for v in angles):
        raise ValueError("Bulb grid loads/angles outside bounded diagnostic range")
    offset=0.;angle=0.;completed=[]
    task.metadata['diagnostic_protocol']={'type':'axial/yaw grid','axial_levels_N':forces,
        'angles_deg':angles,'steady_force_tolerance_N':.35,'hold_s':.5,
        'axial_measurement':'negative z of bulb grip contact resultant, world frame',
        'control':'privileged grip axial force and small yaw; never an RGB success episode'}
    for f_index,force in enumerate(forces):
        for a_index,degrees in enumerate(angles):
            target_angle=np.deg2rad(degrees);stable=0
            task.phase=f'diagnostic {force:g} N / {degrees:+g} deg'
            for tick in range(840):
                if tick%4==0:
                    task.probe_command={'probe_force_target_N':force,'probe_yaw_command_deg':degrees,
                        'probe_grid_index':f_index*len(angles)+a_index,
                        'probe_z_correction_m':offset}
                    row=task._record()
                    grip_axial=-float(row['bulb_grip_contact_resultant']['force_N'][2])
                    row['probe_grip_axial_N']=grip_axial
                    if abs(grip_axial)>80 or row['seat_force_N']>180 or row['torque_Nm']>task.target_torque*2.5:
                        task.failure='diagnostic_overload';return True
                    if row['center_drift_m']>.004 or row['tilt_deg']>10:
                        task.failure='diagnostic_alignment_lost';return True
                    error=grip_axial-force
                    z_speed=float(np.clip(error*.00015,-.001,.001))
                    if abs(error)<.12:z_speed=0.
                    stable=stable+4 if abs(error)<.35 and abs(angle-target_angle)<1e-7 else 0
                    if stable>=60:
                        completed.append({'force_N':force,'angle_deg':degrees,'step':task.step_count,
                                          'actual_grip_axial_N':grip_axial,'seat_force_N':row['seat_force_N'],
                                          'torque_Nm':row['torque_Nm'],
                                          'grip_torque_Nm':row['bulb_grip_contact_resultant']['torque_Nm'][2]})
                        break
                angle+=float(np.clip(target_angle-angle,-.001,.001))
                offset=float(np.clip(offset+z_speed*task.cfg.sim.dt,-.003,.003))
                q=t3d.quaternions.qmult(t3d.quaternions.axangle2quat([0.,0.,1.],angle),origin.q)
                center=Pose(origin.p+[0.,0.,THREAD_PITCH*angle/(2*np.pi)+offset],q)
                if not task._ik_to_center(center):return True
            if stable<60:
                task.failure='diagnostic_force_settle_timeout'
                task.metadata['diagnostic_completed']=False
                task.metadata['diagnostic_levels']=completed
                return True
    task.metadata['diagnostic_completed']=True
    task.metadata['diagnostic_levels']=completed
    task._record()
    return True


# Screw light bulb

BULB_THREAD_END_Z = 0.034

BULB_THREAD_MAJOR_RADIUS = 0.0140

GRASP_LOCAL_Z = 0.058

GRASP_PRE_DISTANCE = 0.050

GRIPPER_ADAPTIVE_CLOSE_TARGET = 0.0

GRIPPER_OPEN_PERCENT = 0.85

GRIPPER_RELEASE_PERCENT = 0.80

IDENTITY_Q = (1.0, 0.0, 0.0, 0.0)

INITIAL_PILOT_INSERTION_DEPTH = 0.006

SCREW_TURN_ANGLE = np.pi / 2

SOCKET_HEIGHT = 0.055

SUCCESS_CENTER_DRIFT_TOLERANCE = 0.002

SUCCESS_INSERTION_TOLERANCE = 0.001

SUCCESS_INSERTION_DEPTH = max(
    0.0,
    BULB_THREAD_END_Z
    - INITIAL_PILOT_INSERTION_DEPTH
    - SUCCESS_INSERTION_TOLERANCE,
)

SUCCESS_PITCH_ERROR_TOLERANCE = 0.003

SUCCESS_ROTATION_DEG = 900.0

SUCCESS_SOCKET_TRANSLATION_TOLERANCE = 0.0005

SUCCESS_THREAD_TOP_CLEARANCE = SUCCESS_INSERTION_TOLERANCE

SUCCESS_TILT_TOLERANCE_DEG = 6.0

@configclass
class TaskCfg(BaseTaskCfg):
    cameras = [
        CameraCfg(
            name="head",
            prim_path="/World/envs/env_.*/Camera",
            offset=CameraCfg.OffsetCfg(
                pos=(0.72, 0.16, 0.16),
                rot=(0.437426, 0.300596, 0.512602, 0.675369),
                convention="opengl",
            ),
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=1.6,
                focus_distance=1.0,
                horizontal_aperture=2.4,
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
    step_lim = 7000
    max_save_frames = 4000
    xense_bulb_adaptive_grasp_depth_threshold: float | None = 27.5
    xense_bulb_adaptive_grasp_require_both_contacts: bool | None = True

WRIST_TURN_STEPS = 120

class LegacyBulbTask(BaseTask):
    def __init__(
        self,
        cfg: TaskCfg,
        mode: Literal["collect", "eval"] = "collect",
        render_mode: str | None = None,
        **kwargs,
    ):
        cfg.sim.physics_material.dynamic_friction = 2.5
        cfg.sim.physics_material.static_friction = 2.5
        cfg.uipc_sim.contact.default_friction_ratio = 2.5
        super().__init__(cfg=cfg, mode=mode, render_mode=render_mode, **kwargs)

    @staticmethod
    def _is_xense_cfg(cfg):
        return getattr(cfg, "tactile_sensor_type", "") in (
            "xensews",
            "xensews_robotiq",
        )

    def load_robot_and_sensors(self, cfg: BaseTaskCfg):
        cfg = super().load_robot_and_sensors(cfg)
        if self._is_xense_cfg(cfg):
            joint_pos = dict(cfg.robot.robot.init_state.joint_pos)
            joint_pos = apply_xense_wrist_y_alignment(joint_pos)
            cfg.robot.robot.init_state.joint_pos.update(joint_pos)
        return cfg






    def pre_move(self):
        self.delay(10)
        self.move(
            self.atom.open_gripper(GRIPPER_OPEN_PERCENT),
            tag="open_gripper_for_bulb",
            delay=False,
        )

    def _make_bulb_pose(
        self,
        socket_pose: Pose,
        yaw: float,
        insertion_depth: float,
    ) -> Pose:
        bulb_bottom_z = (
            socket_pose.p[2]
            + SOCKET_HEIGHT
            - INITIAL_PILOT_INSERTION_DEPTH
            - insertion_depth
        )
        pose = Pose(
            [socket_pose.p[0], socket_pose.p[1], bulb_bottom_z],
            IDENTITY_Q,
        )
        return pose.add_rotation([0.0, 0.0, yaw], coord="local")

    def _record_initial_bulb_pose(self):
        self.initial_bulb_pose = self.bulb.get_pose()
        self.initial_socket_pose = self.socket.get_pose()
        self._cumulative_rotation_signed_deg = 0.0
        self.metadata["initial_bulb_pose"] = self.initial_bulb_pose.tolist()
        self.metadata["initial_socket_pose"] = self.initial_socket_pose.tolist()

    def _approach_bulb(self):
        self._record_initial_bulb_pose()
        bulb_pose = self.bulb.get_pose()
        grasp_position = bulb_pose.add_bias([0.0, 0.0, GRASP_LOCAL_Z]).p
        # Keep the current wrist orientation for the approach. A constructed
        # exact top-down grasp can put GelSight/Panda near a wrist singularity
        # and causes CuRobo planning failures for this bulb location.
        grasp_q = self._robot_manager.get_gripper_center_pose().q
        pre_grasp_pose = Pose(
            grasp_position + np.array([0.0, 0.0, GRASP_PRE_DISTANCE]),
            grasp_q,
        )
        grasp_pose = Pose(
            grasp_position,
            grasp_q,
        )
        self.metadata["bulb_grasp_local_z"] = float(GRASP_LOCAL_Z)
        self.metadata["bulb_pre_grasp_pose"] = pre_grasp_pose.tolist()
        self.metadata["bulb_grasp_pose"] = grasp_pose.tolist()
        self.metadata["bulb_grasp_pre_distance"] = float(GRASP_PRE_DISTANCE)

        self.move(
            self.atom.move_to_pose(
                self._robot_manager.gripper_center_to_ee(pre_grasp_pose)
            ),
            tag="move_above_light_bulb",
            time_dilation_factor=0.5,
            delay=False,
        )
        self.move(
            self.atom.move_to_pose(
                self._robot_manager.gripper_center_to_ee(grasp_pose)
            ),
            tag="approach_light_bulb",
            time_dilation_factor=0.5,
            delay=False,
        )

    def _close_bulb(self, turn_idx: int):
        depth_threshold = self.get_xense_adaptive_grasp_depth_threshold(
            "xense_bulb_adaptive_grasp_depth_threshold"
        )
        require_both_contacts = self.get_xense_adaptive_grasp_require_both_contacts(
            "xense_bulb_adaptive_grasp_require_both_contacts"
        )
        self.move(
            self.atom.close_gripper(pos=GRIPPER_ADAPTIVE_CLOSE_TARGET),
            tag=f"close_light_bulb_{turn_idx}",
            delay=False,
            gripper_depth_threshold=depth_threshold,
            gripper_require_both_contacts=require_both_contacts,
        )
        self.metadata["gripper_close_percent"] = float(GRIPPER_ADAPTIVE_CLOSE_TARGET)
        self.metadata["gripper_adaptive_depth_threshold"] = (
            None if depth_threshold is None else float(depth_threshold)
        )
        self.metadata["gripper_adaptive_require_both_contacts"] = require_both_contacts
        self.metadata["gripper_close_target_gap_estimate"] = float(
            2.0 * self._robot_manager.gripper_max_qpos * GRIPPER_ADAPTIVE_CLOSE_TARGET
        )
        self.settle_xense_after_close(is_save=False)
        self.record_xense_grasp_debug(f"xense_after_close_light_bulb_{turn_idx}", self.bulb)
        self.bulb.remove_animate(force=True)
        self._actor_manager.update(dt=0.0)
        self.delay(2, is_save=False)
        self.metadata[f"bulb_released_for_turn_{turn_idx}"] = True
        if turn_idx == 1:
            self.metadata["bulb_released_after_tactile_grasp"] = True


    def _release_bulb(self, turn_idx: int):
        self.move(
            self.atom.open_gripper(GRIPPER_RELEASE_PERCENT),
            tag=f"release_light_bulb_{turn_idx}",
            delay=False,
        )

    def _rotate_wrist_joint(
        self,
        delta: float,
        steps: int = WRIST_TURN_STEPS,
        tag: str = "rotate_wrist_joint",
        is_save: bool = True,
    ):
        if self.plan_success is False:
            return False

        self.atom_id += 1
        self.atom_tag = tag
        arm_ids = self._robot_manager._arm_ids
        start_qpos = self._robot_manager.robot.data.joint_pos[0, arm_ids].clone()
        target_qpos = start_qpos.clone()
        target_qpos[-1] += delta

        self.metadata[f"{tag}_start_arm_qpos"] = start_qpos.detach().cpu().tolist()
        self.metadata[f"{tag}_target_arm_qpos"] = target_qpos.detach().cpu().tolist()
        self.metadata[f"{tag}_delta_rad"] = float(delta)
        self.metadata[f"{tag}_delta_deg"] = float(np.rad2deg(delta))
        self.metadata[f"{tag}_steps"] = int(steps)

        last_qpos = start_qpos
        for step_idx in range(1, steps + 1):
            alpha = step_idx / steps
            qpos = start_qpos + (target_qpos - start_qpos) * alpha
            vel = (qpos - last_qpos) / self.cfg.sim.dt
            self._robot_manager.set_arm(qpos, vel)
            self._step(is_save)
            last_qpos = qpos
        self._update_render()
        return True



    def _return_gripper_yaw(self, turn_idx: int):
        self._rotate_wrist_joint(
            -SCREW_TURN_ANGLE,
            tag=f"return_gripper_yaw_for_regrasp_{turn_idx}",
        )


    @staticmethod
    def _get_actor_yaw_delta_deg(actor_pose: Pose, initial_pose: Pose) -> float:
        init_mat = initial_pose.to_transformation_matrix()
        curr_mat = actor_pose.to_transformation_matrix()
        init_x = init_mat[:3, 0].copy()
        curr_x = curr_mat[:3, 0].copy()
        init_x[2] = 0.0
        curr_x[2] = 0.0
        init_x /= np.linalg.norm(init_x) + 1e-8
        curr_x /= np.linalg.norm(curr_x) + 1e-8
        return float(
            np.rad2deg(
                np.arctan2(
                    np.dot(np.cross(init_x, curr_x), np.array([0.0, 0.0, 1.0])),
                    np.dot(init_x, curr_x),
                )
            )
        )

    def _get_turn_result(self):
        bulb_pose = self.bulb.get_pose()
        socket_pose = self.socket.get_pose()
        initial_pose = getattr(self, "initial_bulb_pose", bulb_pose)
        initial_socket_pose = getattr(self, "initial_socket_pose", socket_pose)
        initial_relative_position = initial_pose.p - initial_socket_pose.p
        relative_position = bulb_pose.p - socket_pose.p
        relative_position_delta = relative_position - initial_relative_position
        bulb_position_delta = bulb_pose.p - initial_pose.p
        socket_position_delta = socket_pose.p - initial_socket_pose.p
        yaw_wrapped_deg = self._get_actor_yaw_delta_deg(bulb_pose, initial_pose)
        yaw_signed_deg = float(
            getattr(self, "_cumulative_rotation_signed_deg", yaw_wrapped_deg)
        )
        yaw_delta_deg = abs(yaw_signed_deg)
        expected_insertion_depth = THREAD_PITCH * yaw_delta_deg / 360.0
        actual_insertion_depth = max(0.0, -relative_position_delta[2])
        bulb_mat = bulb_pose.to_transformation_matrix()
        socket_mat = socket_pose.to_transformation_matrix()
        bulb_in_socket_mat = np.linalg.inv(socket_mat) @ bulb_mat
        thread_top_center_in_socket = (
            bulb_in_socket_mat @ np.array([0.0, 0.0, BULB_THREAD_END_Z, 1.0])
        )[:3]
        thread_top_vertical_radius = BULB_THREAD_MAJOR_RADIUS * np.linalg.norm(
            bulb_in_socket_mat[2, :2]
        )
        bulb_thread_top_above_socket_rim = float(
            thread_top_center_in_socket[2]
            + thread_top_vertical_radius
            - SOCKET_HEIGHT
        )
        bulb_z_axis = bulb_mat[:3, 2]
        socket_z_axis = socket_mat[:3, 2]
        tilt_cosine = np.clip(
            np.dot(bulb_z_axis, socket_z_axis)
            / (
                np.linalg.norm(bulb_z_axis) * np.linalg.norm(socket_z_axis)
                + 1e-8
            ),
            -1.0,
            1.0,
        )
        tilt_deg = float(np.rad2deg(np.arccos(tilt_cosine)))
        return {
            "bulb_pose": bulb_pose.tolist(),
            "socket_pose_current": socket_pose.tolist(),
            "bulb_center_delta": float(np.linalg.norm(bulb_position_delta)),
            "bulb_center_xy_delta": float(
                np.linalg.norm(relative_position_delta[:2])
            ),
            "bulb_vertical_delta": float(relative_position_delta[2]),
            "bulb_insertion_depth": float(actual_insertion_depth),
            "bulb_expected_insertion_depth_from_yaw": float(expected_insertion_depth),
            "bulb_insertion_depth_error": float(
                expected_insertion_depth - actual_insertion_depth
            ),
            "bulb_thread_top_above_socket_rim": (
                bulb_thread_top_above_socket_rim
            ),
            "bulb_yaw_wrapped_deg": yaw_wrapped_deg,
            "bulb_yaw_signed_deg": yaw_signed_deg,
            "bulb_yaw_delta_deg": yaw_delta_deg,
            "bulb_tilt_relative_to_socket_deg": tilt_deg,
            "socket_translation_delta": float(np.linalg.norm(socket_position_delta)),
        }

    def _record_turn_result(self, result, suffix=""):
        for key, value in result.items():
            if isinstance(value, list):
                self.metadata[f"{key}{suffix}"] = value
            else:
                self.metadata[f"{key}{suffix}"] = float(value)

    def check_success(self):
        result = self._get_turn_result()
        self._record_turn_result(result)
        rotation_success = result["bulb_yaw_delta_deg"] >= SUCCESS_ROTATION_DEG
        insertion_success = result["bulb_insertion_depth"] >= SUCCESS_INSERTION_DEPTH
        full_thread_success = (
            result["bulb_thread_top_above_socket_rim"]
            <= SUCCESS_THREAD_TOP_CLEARANCE
        )
        pitch_success = (
            abs(result["bulb_insertion_depth_error"])
            <= SUCCESS_PITCH_ERROR_TOLERANCE
        )
        centering_success = (
            result["bulb_center_xy_delta"] <= SUCCESS_CENTER_DRIFT_TOLERANCE
        )
        tilt_success = (
            result["bulb_tilt_relative_to_socket_deg"]
            <= SUCCESS_TILT_TOLERANCE_DEG
        )
        socket_stable = (
            result["socket_translation_delta"]
            <= SUCCESS_SOCKET_TRANSLATION_TOLERANCE
        )
        success = all(
            (
                rotation_success,
                insertion_success,
                full_thread_success,
                pitch_success,
                centering_success,
                tilt_success,
                socket_stable,
            )
        )
        self.metadata["screw_light_bulb_rotation_success"] = bool(rotation_success)
        self.metadata["screw_light_bulb_insertion_success"] = bool(insertion_success)
        self.metadata["screw_light_bulb_full_thread_success"] = bool(
            full_thread_success
        )
        self.metadata["screw_light_bulb_pitch_success"] = bool(pitch_success)
        self.metadata["screw_light_bulb_centering_success"] = bool(centering_success)
        self.metadata["screw_light_bulb_tilt_success"] = bool(tilt_success)
        self.metadata["screw_light_bulb_socket_stable"] = bool(socket_stable)
        self.metadata["screw_light_bulb_success"] = bool(success)
        return bool(success)


# Tension strap geometry

def box_mesh(width, depth, height, counts=(5,5,5)):
    nx,ny,nz=counts
    pts=np.array([[x,y,z] for z in np.linspace(0,height,nz)
                  for x in np.linspace(-width/2,width/2,nx)
                  for y in np.linspace(-depth/2,depth/2,ny)])
    tet=[]
    def idx(k,x,y): return k*nx*ny+x*ny+y
    for k in range(nz-1):
        for x in range(nx-1):
            for y in range(ny-1):
                a,b,c,d=[idx(k,x+dx,y+dy) for dx,dy in [(0,0),(1,0),(0,1),(1,1)]]
                e,f,g,h=[idx(k+1,x+dx,y+dy) for dx,dy in [(0,0),(1,0),(0,1),(1,1)]]
                tet.extend([(a,b,d,h),(a,d,c,h),(a,c,g,h),(a,g,e,h),(a,e,f,h),(a,f,b,h)])
    tet=np.asarray(tet,dtype=np.int32)
    return pts,tet,tetra_faces(tet)


# Bulb tightening

class Task(LegacyBulbTask):
    def __init__(self,cfg,**kwargs):
        self.params=task_parameters(cfg)
        self.params.setdefault('public_tactile_grasp',True)
        self.params.setdefault('torque_tolerance_Nm',.012)
        for key,value in sample_physics('bulb_tightening',self.params.get('physics_seed',0)).items():
            self.params.setdefault(key,value)
        if self.params.get('controller')=='marker_rgb':
            check_bulb_observer_contract(self.params,json.loads(Path(self.params['calibration']).read_text()))
        self.work=Path(self.params['workspace'])
        self.work.mkdir(parents=True,exist_ok=True)
        self.controller=self.params.get('controller','diagnostic_force')
        self.thread_spec=sample_thread_spec(self.params)
        self.initial_depth=self.thread_spec.initial_depth_m
        self.controller_spin_sign=None;self.controller_lead_m=None
        rng=np.random.default_rng(int(self.params.get('physics_seed',0))+4319)
        offset=np.asarray(self.params.get('socket_xy_offset_m',rng.uniform(-.02,.02,2)),float)
        if offset.shape!=(2,) or not np.all(np.isfinite(offset)) or np.any(np.abs(offset)>.02):
            raise ValueError('Socket XY displacement must stay within +/- 2 cm')
        self.socket_position=np.r_[np.array([.55,0.])+offset,.002]
        self.seat_top=float(self.params.get('seat_top_m',.022))
        self.target_torque=float(self.params.get('target_torque_Nm',.120))
        self.torque_tolerance=float(self.params.get('torque_tolerance_Nm',.012))
        self.phase='setup'; self.measured={}; self.trace=[]
        self.reference=None; self.failure=None; self.held_steps=0
        self.cumulative_yaw=0.
        self._monitor_bulb=False; self._expert_active=False; self.fractured=False
        self._accepted_result=None;self._submission_requested=False
        self._submission_anchor=None;self._submission_stationary_steps=0
        cfg.video_size=(1120,320)
        if self.params.get('enable_glass_rendering',True):
            cfg.sim.render.enable_translucency=True
            cfg.sim.render.enable_reflections=True
            settings=dict(cfg.sim.render.carb_settings or {})
            # Fractional cutout misorders the opaque GelPad/housing with RTX
            # glass enabled. Keep normal depth ordering; OmniGlass and the
            # filament retain the enabled translucency/reflection rendering.
            settings['/rtx/raytracing/fractionalCutoutOpacity']=False
            # Trace refraction instead of accepting image-reprojection hits
            # across the millimetre-scale gap between bulb and opaque housing.
            settings['/rtx/translucency/worldEps']=0.0
            cfg.sim.render.carb_settings=settings
        cfg.uipc_sim.contact.eps_velocity=float(self.params.get('friction_eps_velocity_m_s',.001))
        if 'newton_velocity_tol_m_s' in self.params:
            cfg.uipc_sim.newton.velocity_tol=float(self.params['newton_velocity_tol_m_s'])
        if 'newton_max_iter' in self.params:
            cfg.uipc_sim.newton.max_iter=int(self.params['newton_max_iter'])
        from ._force_task_utils import configure_final_task
        configure_final_task(cfg, self.params, max_policy_seconds=180)
        super().__init__(cfg,**kwargs)
        self.video_handler.fps=120/cfg.video_frequency
        self.video_handler.encoder_threads=2

    def _setup_scene(self):
        from ._force_task_utils import configure_gel
        configure_gel(self.cfg.robot.tactiles,self.params)
        super()._setup_scene()
        self.presentation=ForceTaskScene(self,'bulb')
        create_bulb_surface(self)
        from pxr import Gf,UsdGeom
        import omni.usd
        for env in self.scene.env_prim_paths:
            mount=UsdGeom.Xformable(omni.usd.get_context().get_stage().GetPrimAtPath(env+'/force_task_presentation'))
            mount.AddTranslateOp().Set(Gf.Vec3d(*(self.socket_position-np.array([.55,0.,.002]))))
        from ._force_task_utils import set_actor_visible
        import omni.usd
        for fragment in getattr(self,'bulb_fragments',[]):
            set_actor_visible(omni.usd.get_context().get_stage(),fragment,False)
        if 'thread_friction' in self.params:
            contacts=self.uipc_sim.scene.contact_tabular()
            body=contacts.create('tightening_bulb_body')
            thread=contacts.create('tightening_socket_thread')
            seat=contacts.create('tightening_compliant_seat')
            resistance=self.cfg.uipc_sim.contact.default_contact_resistance*1e9
            contacts.insert(body,contacts.default_element(),friction_rate=2.5,resistance=resistance)
            contacts.insert(body,thread,friction_rate=float(self.params['thread_friction']),resistance=resistance)
            contacts.insert(body,seat,friction_rate=2.5,resistance=resistance)
            for actor,element in [(self.bulb,body),(self.socket,thread),(self.seat,seat)]:
                for mesh in actor.uipc_meshes:element.apply_to(mesh)

    def _update_render(self):
        if hasattr(self,'bulb'):
            pose=self.bulb.get_pose() if getattr(self,'step_count',0)>0 else self.bulb.init_pose
            update_filament(self,pose=pose)
            update_bulb_surface(self,pose=pose)
        super()._update_render()

    def create_actors(self):
        socket_pose=Pose(self.socket_position,[1.,0.,0.,0.])
        mesh_mode=self.params.setdefault('thread_mesh_mode','source_boundary')
        if mesh_mode not in ('refined','source_boundary'):
            raise ValueError('Unknown thread mesh mode')
        boundary=mesh_mode=='source_boundary'
        preserve_flank=bool(self.params.get('preserve_thread_flank_width',False))
        if preserve_flank and not boundary:
            raise ValueError('Flank-clearance variant requires matching source-boundary meshes')
        version='final_v3' if preserve_flank else ('final_v2' if boundary else 'final_v1')
        asset_dir=Path('assets/objects/task_assets/screw_bulb')/version/self.thread_spec.cache_key()
        self.bulb_asset_path,self.socket_asset_path=generate_variant(
            self.thread_spec,asset_dir,preserve_boundary=boundary,preserve_flank_width=preserve_flank)
        bulb_pose=self._make_bulb_pose(socket_pose,self.thread_spec.initial_yaw_rad,self.initial_depth)
        self.socket=self._actor_manager.add_from_usd_file(
            name='lamp_socket',asset_path=str(Path(self.socket_asset_path).resolve()),
            pose=socket_pose,density=1e6,
            visual_asset_path=str(Path(self.socket_asset_path).resolve()),
            show_physics_mesh=False,keep_constrained=True)
        self.bulb=self._actor_manager.add_from_usd_file(
            name='light_bulb',asset_path=str(Path(self.bulb_asset_path).resolve()),
            pose=bulb_pose,density=5e2,
            visual_asset_path=str(Path(self.bulb_asset_path).resolve()),
            show_physics_mesh=False,keep_constrained=True)
        p,t,f=box_mesh(.016,.016,self.seat_top-.012,(5,5,7))
        self.seat_rest_local=p
        self.seat_tets=t
        self.seat_base=self.socket_position+np.array([0.,0.,.012])
        path=write_tet_asset(self.work/'hidden_seat.usda',p,t,f,(.16,.17,.18))
        modulus=float(self.params.get('seat_modulus_mpa',.12))
        self.seat=self._actor_manager.add_from_usd_file(
            name='compliant_seat', asset_path=path,pose=Pose(self.seat_base,[1.,0.,0.,0.]),
            constitution_cfg=UipcObjectCfg.StableNeoHookeanCfg(youngs_modulus=modulus,poisson_rate=.40),
            density=1100.,show_physics_mesh=True,keep_constrained=True)
        E=modulus*1e6; nu=.4
        self.seat_model=ElasticResultant(p+self.seat_base,t,E/(2*(1+nu)),E*nu/((1+nu)*(1-2*nu)))
        self.seat_anchor=p[:,2]<.0018
        self.seat_upper=p[:,2]>=(self.seat_top-.012)/2
        self.seat_targets=p+self.seat_base
        if self.params.get('glass_damage_enabled',True):
            create_fragments(self)
        create_filament(self)

    def reset(self,*args,**kwargs):
        if self.fractured:raise RuntimeError('Create a new bulb process after glass fracture')
        self._monitor_bulb=False
        ret=super().reset(*args,**kwargs)
        self._record_initial_bulb_pose()
        self.previous_pose=self.bulb.get_pose()
        self.grip_peak_N=0.; self.fractured=False; self._accepted_result=None
        self._submission_requested=False;self._submission_anchor=None
        self._submission_stationary_steps=0
        self._last_measure_step=None;self._reset_grip_released=False
        self._motion_sample_step=self.step_count;self._motion_sample_pose=self.bulb.get_pose()
        self._angular_speed=0.;self._axial_speed=0.
        self._image_tracker=None;self._tracker_step=None;self._tracker_reference=None
        self._monitor_bulb=True
        return ret

    def _reset_actors(self):
        if int(self.cfg.seed)!=int(self.params.get('physics_seed',0)):
            raise ValueError('Create one force-task process per seed so physical rest geometry/materials are sampled before UIPC initialization')
        self.phase='setup'; self.trace=[]; self.reference=None
        self.failure=None; self.held_steps=0; self.policy_hold=0; self.cumulative_yaw=0.
        self._torque_filter_value=None; self._torque_filter_step=None
        reset_lighting(self)
        for fragment in getattr(self,'bulb_fragments',[]):
            fragment.set_pose(fragment.init_pose)
        socket_pose=Pose(self.socket_position,[1.,0.,0.,0.])
        # Advancing down requires negative world yaw for the matched right-hand thread.
        yaw=self.thread_spec.initial_yaw_rad
        bulb_pose=self._make_bulb_pose(socket_pose,yaw,self.initial_depth)
        self.socket.set_pose(socket_pose)
        self.bulb.set_pose(bulb_pose)
        self.seat.set_vertex_targets(self.seat_targets,self.seat_anchor)
        self._reset_socket_pose=socket_pose
        self._reset_bulb_pose=bulb_pose
        self.metadata['bulb_parameters']=dict(self.params)
        self.metadata['bulb_visual_surface']=getattr(self,'_bulb_visual_audit',[])
        self.metadata['backend_energy_volume_scale']=6.0
        self.metadata['diagnostic_only']=self.controller!='marker_rgb'
        self.metadata['randomization']='sample physical parameters, then apply explicit config overrides; see bulb_parameters'
        self.metadata['constraint_contract']='socket and lower seat fixed; bulb free after first grasp including regrasp'
        self.metadata['remaining_geometric_turns']=self.thread_spec.nominal_free_travel_m/self.thread_spec.lead_m
        self.metadata['thread_physical_spec']=dict(self.thread_spec.__dict__)
        self.metadata['socket_xy_offset_m']=(self.socket_position[:2]-[.55,0.]).tolist()
        self._probe_seen_positive=False;self._probe_seen_negative=False;self._probe_physical_complete=False
        self._physical_probe=ReversibleThreadProbe(track_grasp_origin=True)
        self._physical_trace_path=self.work/'physical_trace.jsonl'
        self._physical_trace_path.write_text('')
        self.metadata['bulb_physical_trace_path']=str(self._physical_trace_path)
        self.controller_spin_sign=None;self.controller_lead_m=None

    def _release_reset_constraints(self):
        self.seat.set_vertex_targets(self.seat_targets,self.seat_anchor)

    def build_instruction(self):
        return f"Safely probe both directions, back off, then screw the bulb in and hold {self.target_torque:g} newton metres of tightening torque."


    def _measure(self):
        if getattr(self,'_last_measure_step',None)==self.step_count:return self.measured
        pose=self.bulb.get_pose()
        # Estimate motion on a fixed cadence; trace reads must not shorten the
        # velocity interval or compare a newly rendered pose with a stale one.
        if self.step_count%2==0 and self.step_count>self._motion_sample_step:
            elapsed=(self.step_count-self._motion_sample_step)*self.cfg.sim.dt
            self._angular_speed=abs(self._get_actor_yaw_delta_deg(pose,self._motion_sample_pose))/elapsed
            self._axial_speed=abs(float(pose.p[2]-self._motion_sample_pose.p[2]))/elapsed
            self._motion_sample_pose=pose;self._motion_sample_step=self.step_count
        seat=self.seat_model.measure(self.seat.vertex_positions,self.seat_upper,center=self.seat_base)
        actual_yaw=self._get_actor_yaw_delta_deg(pose,self.previous_pose)
        self.cumulative_yaw+=actual_yaw
        self.previous_pose=pose
        torque=np.zeros(3)
        grip=np.zeros(3)
        pad_normal={};pad_resultant={}
        for name,sensor in self._tactile_manager.tactiles.items():
            f=sensor._get_contact_force().detach().cpu().numpy().reshape(-1,3)
            p=sensor.gelpad.data.nodal_pos_w.detach().cpu().numpy().reshape(-1,3)
            # Independent contact-gradient diagnostic; dt^2 conversion cross-checked with FEM seat.
            torque+=np.cross(p-pose.p,-f).sum(axis=0)/self.cfg.sim.dt**2
            resultant=-f.sum(axis=0)/self.cfg.sim.dt**2
            grip+=resultant
            radial=pose.p-p.mean(axis=0);radial[2]=0.
            radial/=max(np.linalg.norm(radial),1e-12)
            pad_normal[name]=abs(float(np.dot(resultant,radial)))
            pad_resultant[name]=resultant.tolist()
        self.grip_peak_N=max(getattr(self,'grip_peak_N',0.),max(pad_normal.values(),default=0.))
        seat_torque=abs(float(seat['elastic_torque_Nm'][2]))
        seat_compression=max(0.,float(seat['elastic_force_N'][2]))
        initial=self.initial_bulb_pose
        advance=float(initial.p[2]-pose.p[2])
        expected=self.thread_spec.expected_advance(self.cumulative_yaw)
        if not self._probe_physical_complete:
            locked=self._physical_probe.observe_grasp_origin(
                self.cumulative_yaw,advance,
                bilateral_grip=(self._reset_grip_released
                    and min(pad_normal.values(),default=0.)>5.))
            if locked:
                self.metadata['thread_probe_grasp_origin']=dict(
                    step=self.step_count,world_rotation_deg=self._physical_probe.origin_degrees,
                    advance_m=self._physical_probe.origin_advance_m,
                    source='physical bilateral grasp; axial settling before first angular excursion',
                    frozen_once=True,expert_phase_used=False)
            if self._physical_probe.update(self.cumulative_yaw,advance):
                self._probe_physical_complete=True
                self.metadata['thread_probe_physical_return_step']=self.step_count
                self.metadata['formal_motion_reference']=dict(
                    step=self.step_count,world_rotation_deg=self.cumulative_yaw,
                    advance_m=advance,source='independently recognized physical probe return')
                self.metadata['thread_probe_physical_limits']=dict(
                    minimum_excursion_deg=2.,maximum_excursion_deg=6.,
                    return_angle_deg=.6,return_advance_m=.0002,
                    recognition='each physical direction must separately return safely; wrong direction alone never fails')
        formal_angle,formal_advance=self._physical_probe.formal_motion(self.cumulative_yaw,advance)
        tilt=float(np.degrees(np.arccos(np.clip(pose.R[2,2],-1,1))))
        m=dict(torque_Nm=seat_torque,signed_torque_Nm=float(seat['elastic_torque_Nm'][2]),seat_force_N=seat_compression,
            contact_torque_Nm=torque.tolist(),contact_force_N=grip.tolist(),
            rotation_deg=self.thread_spec.signed_progress_turns(self.cumulative_yaw)*360,
            signed_world_rotation_deg=self.cumulative_yaw,
            formal_net_turns=self.thread_spec.signed_progress_turns(formal_angle),
            formal_advance_m=formal_advance,
            probe_complete=self._probe_physical_complete,advance_m=advance,
            phase_error_m=abs(advance-expected),tilt_deg=tilt,
            center_drift_m=float(np.linalg.norm(pose.p[:2]-initial.p[:2])),
            maximum_stretch=seat['maximum_stretch'],
            seat_compression_m=float(self.seat_top+self.socket_position[2]-pose.p[2]),
            bulb_z_m=float(pose.p[2]))
        m['angular_speed_deg_s']=self._angular_speed
        m['axial_speed_m_s']=self._axial_speed
        m['pad_normal_force_N']=pad_normal
        m['pad_contact_resultant_N']=pad_resultant
        m['peak_pad_normal_force_N']=self.grip_peak_N
        m['gripper_qpos_m']=float(self._robot_manager.get_gripper_qpos())
        m['grip_depth_mm']=self._tactile_manager.get_min_depth().detach().cpu().numpy().reshape(-1).tolist()
        m.update(getattr(self,'probe_command',{}))
        self.measured=m
        self._last_measure_step=self.step_count
        return m

    def _tracked_image_features(self,current=None):
        from ._force_task_utils import tracked_flow_rgb_features
        from ._force_task_utils import MarkerFlowTracker
        if self._tracker_reference is not self.reference:
            self._image_tracker=MarkerFlowTracker(self.reference)
            self._tracker_reference=self.reference
            self._tracker_step=None
        if self._tracker_step!=self.step_count:
            current=read_rgb(self) if current is None else current
            self._tracker_result=tracked_flow_rgb_features(self.reference,current,self._image_tracker)
            self._tracker_step=self.step_count
        return self._tracker_result

    def _record(self):
        current=read_rgb(self) if self.reference is not None else None
        row=dict(step=self.step_count,phase=self.phase,**self._measure())
        row['physical_hold_steps']=self.held_steps
        if getattr(self,'bulb_grip_mask',None) is not None:
            points=world_contact_vertices(self.bulb)
            surface=self.bulb.vertices
            row['contact_vertex_bbox_error_m']=float(np.max(np.abs(
                np.r_[points.min(0),points.max(0)]-np.r_[surface.min(0),surface.max(0)])))
            row['bulb_grip_contact_resultant']=actor_contact_resultant(self,self.bulb,self.bulb_grip_mask,self.bulb.get_pose().p)
        row['seat_contact_resultant']=actor_contact_resultant(self,self.seat,self.seat_upper,self.seat_base)

        if self.reference is not None:
            try:
                extractor=(getattr(self,'calibration',None) or {}).get('feature_extractor',
                    self.params.get('image_feature_extractor','flow'))
                if extractor=='tracked_flow_rgb_grid':
                    features,details=self._tracked_image_features(current)
                else:
                    fn=flow_rgb_features if extractor=='flow_rgb_grid' else image_features
                    features,details=fn(self.reference,current)
                row['image_feature_extractor']=extractor
                row['image_features']=features.tolist(); row['image_tracking']=details
            except ValueError as e: row['image_error']=str(e)
            if self.step_count-getattr(self,'_last_saved_rgb_step',-100000)>=int(self.params.get('record_images_every',1000000000)):
                import cv2
                for n,im in current.items():
                    cv2.imwrite(str(self.work/f'{self.step_count:05d}_{n}.png'),cv2.cvtColor(im,cv2.COLOR_RGB2BGR))
                self._last_saved_rgb_step=self.step_count
        if self.controller=='marker_rgb' and getattr(self,'calibration',None) and 'image_features' in row:
            raw_torque=predict_calibrated(self.calibration,row['image_features'])
            row['controller_raw_torque_Nm']=raw_torque
            row['controller_torque_Nm']=raw_torque
            if self.phase=='hold':
                previous=getattr(self,'_torque_filter_value',None)
                previous_step=getattr(self,'_torque_filter_step',None)
                if previous is None:
                    self._torque_filter_value=raw_torque
                elif previous_step!=self.step_count:
                    alpha=1.-np.exp(-((self.step_count-previous_step)*self.cfg.sim.dt)/.15)
                    self._torque_filter_value=previous+alpha*(raw_torque-previous)
                self._torque_filter_step=self.step_count
                row['controller_torque_Nm']=float(self._torque_filter_value)
            row['controller_seat_force_N']=predict_calibrated(self.calibration['seat_force'],row['image_features'])
        self.trace.append(row)
        with (self.work/'trace.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        return row

    def _step(self,is_save=True):
        active=getattr(self,'_monitor_bulb',False) and not self.fractured
        if active and hasattr(self,'bulb_fragments'):
            from ._force_task_utils import capture_geometry_state
            self._bulb_previous_geometry=capture_geometry_state(self.uipc_sim)
        previous_step=self.step_count
        log_every=int(self.params.get('solver_diagnostics_every',0) if active else
                      self.params.get('reset_solver_diagnostics_every',0))
        log_solver=log_every>0 and (self.step_count+1)%log_every==0
        if log_solver:
            from uipc import Logger
            Logger.set_level(Logger.Info)
            print(f'BULB_SOLVER_DIAGNOSTIC_BEGIN step={self.step_count+1}',flush=True)
        try:
            super()._step(is_save=is_save)
        finally:
            if log_solver:
                Logger.set_level(Logger.Warn)
                print(f'BULB_SOLVER_DIAGNOSTIC_END step={self.step_count}',flush=True)
        if self.step_count==previous_step:return
        if active:
            row=self._measure()
            if not self._reset_grip_released:
                depths=np.asarray(row['grip_depth_mm'])
                if (np.all(depths<=float(self.params.get('grip_depth_mm',27.2))+.1)
                        and min(row['pad_normal_force_N'].values(),default=0.)>5.):
                    self.bulb.remove_animate(force=True)
                    self._actor_manager.update(dt=0.)
                    self._reset_grip_released=True
                    self.metadata['bilateral_grip_release_step']=self.step_count
                    self.metadata['bilateral_grip_release_depth_mm']=depths.tolist()
            if not self.failure:
                if (self.params.get('glass_damage_enabled',True)
                        and max(row['pad_normal_force_N'].values(),default=0.)
                        >float(self.params.get('glass_break_force_N',170.))):
                    self.failure='glass_crushed'
                    fracture(self,row)
                elif row['torque_Nm']>float(self.params.get('over_torque_Nm',.150)):
                    self.failure='over_tightened'
                elif row['seat_force_N']>180.:
                    self.failure='axial_overload'
                elif (self._reset_grip_released
                        and (row['center_drift_m']>.004 or row['tilt_deg']>10.)):
                    self.failure='grasp_or_alignment_lost'
            good=(self.failure is None
                  and row['signed_torque_Nm']*self.thread_spec.handedness>0.
                  and abs(row['torque_Nm']-self.target_torque)<=self.torque_tolerance
                  and row['seat_force_N']>3. and row['phase_error_m']<.002
                  and row['angular_speed_deg_s']<3. and row['axial_speed_m_s']<.00015)
            self.held_steps=self.held_steps+1 if good else 0
            if (self.failure is None and self.phase_id==self.PHASE_POLICY
                    and (self.step_count-self.policy_start_step)*self.cfg.sim.dt>=self.cfg.final_policy_timeout_seconds
                    and not self.check_success()):
                self.failure='timeout'
            if self.step_count%4==0 or self.failure:
                physical_keys=('signed_world_rotation_deg','formal_net_turns','formal_advance_m','advance_m','phase_error_m',
                    'torque_Nm','signed_torque_Nm','seat_force_N','center_drift_m','tilt_deg',
                    'peak_pad_normal_force_N','angular_speed_deg_s','axial_speed_m_s')
                trace_row={key:row[key] for key in physical_keys}
                trace_row.update(step=self.step_count,phase=self.phase,
                    probe_complete=self._probe_physical_complete,held_steps=self.held_steps,
                    failure=self.failure)
                with self._physical_trace_path.open('a') as stream:
                    stream.write(json.dumps(trace_row)+'\n')
            if self.failure and self._expert_active:
                raise BulbDamageStop(self.failure)
        extractor=(getattr(self,'calibration',None) or {}).get('feature_extractor',
            self.params.get('image_feature_extractor','flow'))
        if self.reference is not None and extractor=='tracked_flow_rgb_grid' and self.step_count%4==0:
            try:self._tracked_image_features()
            except ValueError:pass  # _record reports image validity to the controller.
        if getattr(self,'_grip_probe_active',False) and self.step_count%4==0:
            row=self._record()
            if max(row['pad_normal_force_N'].values(),default=0.)>150.:
                self.failure='diagnostic_grip_force_limit'
                self.plan_success=False

    def take_action(self,action,*args,**kwargs):
        # ACT retains the Insert_USB eight-joint-action contract. Its END
        # gesture is to keep a gripped command still for 1.5 s. Submission
        # depends only on commands/cadence, never on the hidden torque band.
        if self._accepted_result is not None:
            return bool(self._accepted_result),bool(self._accepted_result)
        before=self.step_count
        executed,_=super().take_action(action,*args,**kwargs)
        command=np.asarray(action.detach().cpu() if hasattr(action,'detach') else action,
                           dtype=float).reshape(-1)
        if self._reset_grip_released and command.size==8:
            tolerance=np.r_[np.full(7,.005),.0001]
            anchor=self._submission_anchor
            if anchor is None or np.any(np.abs(command-anchor)>tolerance):
                self._submission_anchor=command.copy()
                self._submission_stationary_steps=0
            else:
                self._submission_stationary_steps+=max(0,self.step_count-before)
        else:
            self._submission_anchor=None;self._submission_stationary_steps=0
        requested=self._submission_stationary_steps>=180
        if requested or self.failure:
            self.metadata['bulb_submission_source']='stationary_joint_command' if requested else 'physical_failure'
            self.metadata['bulb_submission_stationary_steps']=self._submission_stationary_steps
            self._finish_bulb_episode()
            self.eval_success=bool(self._accepted_result)
            return bool(executed and not self.failure),bool(self._accepted_result)
        return executed,False

    def check_early_stop(self):
        return self.failure is not None or self._accepted_result is False

    def _finish_bulb_episode(self):
        if self._accepted_result is not None:
            return
        self._submission_requested=True
        self.metadata.setdefault('bulb_submission_source','expert_submitted_end')
        self._accepted_result=bool(self.check_success())
        if not self._accepted_result and self.failure is None:
            self.failure='submission_outside_acceptance'
        self.metadata['bulb_accepted']=self._accepted_result
        self.metadata['bulb_failure']=self.failure
        self.metadata['bulb_final']=dict(self.measured)
        self.metadata['bulb_physical_verdict_step']=self.step_count
        self.metadata['bulb_hold_steps']=self.held_steps
        self.metadata['bulb_trace_path']=str(self.work/'trace.jsonl')
        self._set_phase(self.PHASE_TERMINAL,terminal_reason='success' if self._accepted_result else (self.failure or 'not_accepted'))
        self._monitor_bulb=False
        if self._accepted_result:
            light_after_submission(self)
        if self.cfg.save_frequency>0 and self.mode!='eval_test':
            from ._force_task_utils import record_terminal_observation
            record_terminal_observation(self,'success' if self._accepted_result else (self.failure or 'incomplete'))
        manager=self._robot_manager
        joints=manager.robot.data.joint_pos[:,manager._arm_ids][0]
        manager.set_arm(joints,torch.zeros_like(joints),force=True)
        qpos=float(manager.get_gripper_qpos())
        manager.set_gripper(qpos,force=True)
        review_start=self.step_count
        if self.plan_success:
            self.phase='terminal_review'
            if self._accepted_result:
                # Reveal the powered bulb only after the submitted result is
                # frozen and terminal frames have been excluded from actions.
                for k in range(90):
                    manager.set_gripper(min(.039,qpos+(k+1)*.00025),force=True)
                    self._step(is_save=True)
                origin=manager.get_gripper_center_pose()
                revealed=True
                for k in range(150):
                    center=Pose(origin.p+[0.,0.,.060*(k+1)/150.],origin.q)
                    if not self._ik_to_center(center):
                        revealed=False;break
                self.metadata['bulb_terminal_reveal_completed']=revealed
                joints=manager.robot.data.joint_pos[:,manager._arm_ids][0].clone()
                manager.set_arm(joints,torch.zeros_like(joints),force=True)
                for _ in range(60):self._step(is_save=True)
            else:
                for k in range(180 if self.fractured else 120):
                    if self.fractured:
                        manager.set_gripper(min(qpos+.012,k*.0001+qpos),force=True)
                    self._step(is_save=True)
        self.metadata['bulb_terminal_review_frames']=int((self.step_count-review_start)/self.cfg.video_frequency)

    def _run_grip_probe(self):
        self.metadata['diagnostic_only']=True
        self.metadata['diagnostic_protocol']='Progressive physical grip depths; no screw motion or success claim'
        self._grip_probe_active=True
        self.reference=self.unloaded_reference
        self.calibration=None
        completed=[]
        for depth in self.params.get('probe_grip_depths_mm',[27.2,26.8,26.4]):
            self.phase=f'diagnostic_grip_{depth:g}'
            self.move(self.atom.close_gripper(0.),tag=f'grip_depth_{depth:g}',
                delay=False,gripper_require_both_contacts=True,gripper_depth_threshold=float(depth))
            if not self.plan_success:break
            self.bulb.remove_animate(force=True)
            self.delay(120,is_save=True)
            if not self.plan_success:break
            completed.append(dict(depth_mm=depth,step=self.step_count,
                normal_force_N=self.measured['pad_normal_force_N'],
                actual_depth_mm=self.measured['grip_depth_mm']))
        self._grip_probe_active=False
        self.metadata['diagnostic_levels']=completed
        self.metadata['diagnostic_completed']=len(completed)==len(self.params.get('probe_grip_depths_mm',[27.2,26.8,26.4]))
        self.metadata['bulb_final']=self.measured
        self.metadata['bulb_failure']=self.failure
        self.metadata['bulb_trace_path']=str(self.work/'trace.jsonl')

    def _position_empty_wrist(self):
        from ._force_task_utils import execute_joint_target
        manager=self._robot_manager
        joints=manager.robot.data.joint_pos[:,manager._arm_ids][0].clone()
        target=float(self.controller_spin_sign)*1.65
        limits=manager.robot.data.soft_joint_pos_limits[0,manager._arm_ids[-1]]
        if not float(limits[0])+.1<target<float(limits[1])-.1:
            self.failure='wrist_range_not_available';return False
        begin=self.step_count
        self.atom_id+=1;self.atom_tag='position_empty_wrist'
        for tick in range(2,121,2):
            alpha=tick/120
            alpha=alpha*alpha*(3-2*alpha)
            command=joints.clone()
            command[-1]=float(joints[-1])+(target-float(joints[-1]))*alpha
            executed,accepted=execute_joint_target(self,arm=command,ticks=2)
            if not executed or accepted:return False
        self.metadata.setdefault('empty_wrist_positions',[]).append(dict(
            start_step=begin,end_step=self.step_count,joint7_target_rad=target,
            gripper_depth_mm=self._read_bulb_depths(),
            object_was_unconstrained=True))
        return True

    def _read_bulb_depths(self):
        return self._tactile_manager.get_min_depth().detach().cpu().numpy().reshape(-1).tolist()

    def _release_bulb(self,turn_idx):
        if not self.params.get('efficient_wrist_motion',False):
            return super()._release_bulb(turn_idx)
        from ._force_task_utils import execute_joint_target
        manager=self._robot_manager
        initial=float(manager.get_gripper_qpos())
        target=min(.039,initial+.004)
        self.atom_id+=1;self.atom_tag=f'release_for_wrist_{turn_idx}'
        for tick in range(2,17,2):
            alpha=tick/16;alpha=alpha*alpha*(3-2*alpha)
            executed,accepted=execute_joint_target(self,gripper=initial+(target-initial)*alpha,ticks=2)
            if not executed or accepted:return
        # Actual unloaded tactile depth determines the clearance. A bulb is
        # never pinned during this opening or the following wrist return.
        for _ in range(40):
            if min(self._read_bulb_depths())>=33.5:break
            target=min(.039,target+.0002)
            executed,accepted=execute_joint_target(self,gripper=target,ticks=2)
            if not executed or accepted:return
        if min(self._read_bulb_depths())<33.5:
            self.failure='regrasp_clearance_not_acquired'
        self.metadata.setdefault('efficient_releases',[]).append(dict(
            step=self.step_count,initial_qpos_m=initial,open_qpos_m=target,
            final_depth_mm=self._read_bulb_depths()))

    def _close_bulb(self,index,force_recenter=False):
        if index>1 or force_recenter:
            pose=self.bulb.get_pose()
            center=pose.add_bias([0.,0.,.058]).p
            grasp=Pose(center,self._robot_manager.get_gripper_center_pose().q)
            self.move(self.atom.move_to_pose(self._robot_manager.gripper_center_to_ee(grasp)),
                      tag=f'recenter_regrasp_{index}',time_dilation_factor=.5,delay=False)
        if self.params.get('public_tactile_grasp',False):
            if not self._close_bulb_public(index):
                raise BulbDamageStop(self.failure or 'public_gripper_close_failed')
        else:
            self.move(self.atom.close_gripper(0.),tag=f'bilateral_bulb_grasp_{index}',
                      delay=False,gripper_require_both_contacts=True,
                       gripper_depth_threshold=float(self.params.get('grip_depth_mm',27.2)))
        for _ in range(16):
            if self._reset_grip_released:
                break
            self._step(is_save=True)
        if not self._reset_grip_released:
            self.failure='bilateral_grip_not_acquired'
            raise BulbDamageStop(self.failure)
        self.metadata[f'grasp_{index}_center']=self._robot_manager.get_gripper_center_pose().tolist()

    def _close_bulb_public(self,index):
        from ._force_task_utils import next_grip_qpos
        from ._force_task_utils import execute_joint_target
        manager=self._robot_manager
        log=dict(grasp_index=index,depth_target_mm=float(self.params.get('grip_depth_mm',27.2)),
            action_repeat=2,minimum_step_m=5e-6,maximum_step_m=1e-4,
            inputs='existing bilateral raw tactile depth and current jaw position',samples=[])
        self.metadata.setdefault('public_grasp_control',[]).append(log)
        for _ in range(400):
            depths=self._tactile_manager.get_min_depth().detach().cpu().numpy()
            current=float(manager.get_gripper_qpos())
            try:
                target=next_grip_qpos(depths,current,log['depth_target_mm'])
            except ValueError:
                self.failure='invalid_grasp_tactile';return False
            log['samples'].append(dict(step=self.step_count,qpos_m=current,
                depths_mm=np.asarray(depths,float).reshape(-1).tolist(),target_qpos_m=target))
            if target is None:
                log['stop_reason']='bilateral_depth_target';return True
            if target==current:
                self.failure='gripper_travel_limit';return False
            executed,accepted=execute_joint_target(self,gripper=target,ticks=2)
            if not executed or self._accepted_result is not None:
                log['stop_reason']=self.failure or 'public_terminal'
                return bool(accepted)
        # Never fall through to a fully-closed target after exhausting a loop.
        self.failure='gripper_close_budget_exceeded'
        log['stop_reason']=self.failure
        return False

    def _ik_to_center(self,center):
        manager=self._robot_manager
        target=manager.gripper_center_to_ee(center)
        pos,q=manager.get_ee_pose_tensor()
        joints=manager.robot.data.joint_pos[:,manager._arm_ids]
        cmd=torch.tensor(np.r_[target.p,target.q],dtype=torch.float32,device=self.device).reshape(1,7)
        manager._ik_controller.set_command(cmd)
        goal=manager._ik_controller.compute(pos,q,manager.jacobian_b[:,:,manager._arm_ids],joints)
        delta=goal-joints
        limits=manager.robot.data.soft_joint_pos_limits[:,manager._arm_ids]
        if (not bool(torch.all(torch.isfinite(goal))) or float(torch.abs(delta).max())>.08
                or bool(torch.any(goal<limits[...,0])) or bool(torch.any(goal>limits[...,1]))):
            self.failure='screw_IK_limit'
            return False
        if self.phase_id==self.PHASE_POLICY:
            from ._force_task_utils import execute_joint_target
            executed,accepted=execute_joint_target(self,arm=goal[0],ticks=2)
            if not executed and not accepted:return False
        else:
            manager.set_arm(goal[0],torch.zeros_like(goal[0]),force=True)
            self._step(is_save=True);self._step(is_save=True)
        # The renderer runs every two ticks. Refresh robot kinematics between
        # renders so the next IK velocity is based on the last physical step,
        # matching the established chip servo rather than a stale joint state.
        if self.last_render!=self.step_count:
            manager.robot.update(dt=self.cfg.sim.dt)
        return self.plan_success

    def _return_gripper_yaw(self,index):
        if self.params.get('efficient_wrist_motion',False):
            return self._position_empty_wrist()
        start=self._robot_manager.get_gripper_center_pose()
        for tick in range(2,241,2):
            q=t3d.quaternions.qmult(
                t3d.quaternions.axangle2quat([0.,0.,1.],-self.controller_spin_sign*np.pi/2*tick/240),start.q)
            if not self._ik_to_center(Pose(start.p,q)):
                return

    def _screw_segment(self,index):
        manager=self._robot_manager
        start=manager.get_gripper_center_pose()
        stopped=False
        axial_correction=0.
        axial_trace=[]
        segment_lead=self.controller_lead_m
        segment_angle=np.pi if self.params.get('efficient_wrist_motion',False) else np.pi/2
        command_scale=1.
        skipped_ticks=0.
        seating_scale_trace=[]
        for tick in range(2,241,2):
            if self._accepted_result is not None:return True
            skipped_ticks+=2*(1-command_scale)
            angle=segment_angle*(tick-skipped_ticks)/240
            q=t3d.quaternions.qmult(
                t3d.quaternions.axangle2quat([0.,0.,1.],self.controller_spin_sign*angle),start.q)
            center=Pose(start.p-[0.,0.,segment_lead*angle/(2*np.pi)+axial_correction],q)
            if not self._ik_to_center(center): return False
            if tick%4==0:
                row=self._record()
                if row['torque_Nm']>self.target_torque*2.5 or row['seat_force_N']>180:
                    self.failure='overload'; return False
                if row['center_drift_m']>.004 or row['tilt_deg']>10:
                    self.failure='grasp_or_alignment_lost'; return False
                if self.controller=='marker_rgb':
                    if 'image_features' not in row:
                        self.failure='marker_tracking_lost'; return False
                    sensed=predict_calibrated(self.calibration,row['image_features'])
                else: sensed=row['torque_Nm']
                seat_sensed=row['seat_force_N'] if self.controller!='marker_rgb' else predict_calibrated(self.calibration['seat_force'],row['image_features'])
                reached=(row['rotation_deg']>=float(self.params['reference_rotation_deg'])
                         if self.controller=='angle_reference'
                         else sensed>=self.target_torque and seat_sensed>3.)
                if self.params.get('bulb_review_scenario')=='over_tightened':
                    reached=False  # Deliberate continued motion; actual overload monitor still stops it.
                if reached:
                    stopped=True; break
                if self.params.get('progressive_seating_speed',False):
                    # Smaller commands after sensed seating load appears;
                    # physical overload thresholds remain independent.
                    command_scale=.2 if sensed>self.target_torque/3 or seat_sensed>2. else 1.
                    seating_scale_trace.append(dict(step=self.step_count,scale=command_scale,
                        sensed_torque_Nm=float(sensed),sensed_seat_force_N=float(seat_sensed)))
                if self.params.get('axial_tactile_feedback',True):
                    try:
                        shear,tracking=axial_image_displacement(self)
                    except ValueError:
                        self.failure='axial_tactile_tracking_lost';return False
                    # Positive shear means the jaws lead the bulb downward.
                    # Correct that shear gradually; no thread geometry or
                    # physical object displacement enters this controller.
                    adjustment=float(np.clip(-shear*(4*self.cfg.sim.dt/.15),-.00002,.00002))
                    axial_correction=float(np.clip(axial_correction+adjustment,-.0015,.0015))
                    axial_trace.append(dict(step=self.step_count,shear_m=shear,
                        descent_correction_m=axial_correction))
        self.metadata.setdefault('axial_tactile_control',[]).append(dict(
            segment=index,initial_estimated_lead_m=segment_lead,trace=axial_trace,
            final_descent_correction_m=axial_correction,
            issued_rotation_rad=float(angle),seating_command_scale_trace=seating_scale_trace,
            inputs='raw tactile images and issued joint/pose commands'))
        if not stopped and not self.failure:
            self.controller_lead_m=float(np.clip(segment_lead+axial_correction*2*np.pi/max(angle,1e-9),.0035,.013))
        if stopped:
            if self.controller=='angle_reference':
                return self._hold_angle_reference()
            if self.controller=='diagnostic_torque_probe':
                return self._torque_probe()
            if self.controller=='diagnostic_load_grid':
                return bulb_grid(self)
            self.phase='hold'
            hold_start=manager.get_gripper_center_pose()
            correction=0.
            stable=0
            omega=0.
            frozen_joints=None
            for k in range(0,1200,2):
                if self._accepted_result is not None:return True
                if k%4==0:
                    row=self._record()
                    good=(abs(row['torque_Nm']-self.target_torque)<=self.torque_tolerance
                          and row['seat_force_N']>3. and row['phase_error_m']<.002)
                    if row['torque_Nm']>self.target_torque*2.5 or row['seat_force_N']>180:
                        self.failure='overload_in_hold';return False
                    if self.controller=='marker_rgb':
                        if 'image_features' not in row:
                            self.failure='marker_tracking_lost';return False
                        estimate=row['controller_torque_Nm']
                        seated=predict_calibrated(self.calibration['seat_force'],row['image_features'])>3.
                    else:
                        estimate=row['torque_Nm'];seated=row['seat_force_N']>3.
                    error=self.target_torque-estimate
                    omega=float(np.clip(error*8.,-.15,.15))
                    if abs(error)<float(self.params.get('hold_control_deadband_Nm',self.torque_tolerance*.75)):
                        omega=0.
                    stable=stable+4 if seated and omega==0. else 0
                    if stable>=int(self.params.get('policy_hold_steps',132)):
                        self.metadata['bulb_policy_hold_steps']=stable
                        # Continue the identical command until take_action
                        # recognizes the public180-tick stationary END.
                correction+=omega*self.cfg.sim.dt*2
                q=t3d.quaternions.qmult(
                    t3d.quaternions.axangle2quat([0.,0.,1.],self.controller_spin_sign*correction),hold_start.q)
                center=Pose(hold_start.p-[0.,0.,self.controller_lead_m*correction/(2*np.pi)],q)
                if omega==0.:
                    if frozen_joints is None:
                        frozen_joints=manager.robot.data.joint_pos[:,manager._arm_ids][0].clone()
                    from ._force_task_utils import execute_joint_target
                    executed,accepted=execute_joint_target(self,arm=frozen_joints,ticks=2)
                    if self._accepted_result is not None:return True
                    if not executed:return False
                else:
                    frozen_joints=None
                    if not self._ik_to_center(center):return False
            self.failure='controller_hold_timeout'
            return True
        return False

    def _hold_angle_reference(self):
        """Perfect visible-pose comparator: hold gripper pose, no load correction."""
        self.phase='hold'
        target=self._robot_manager.get_gripper_center_pose()
        hold_steps=int(self.params.get('policy_hold_steps',180))
        for k in range(hold_steps):
            if not self._ik_to_center(target):return False
            if k%4==3:
                row=self._record()
                good=(abs(row['torque_Nm']-self.target_torque)<=self.torque_tolerance
                      and row['seat_force_N']>3. and row['phase_error_m']<.002)
                if row['torque_Nm']>self.target_torque*2.5 or row['seat_force_N']>180:
                    self.failure='overload_in_reference_hold';return False
        self.metadata['bulb_policy_hold_steps']=hold_steps
        return True

    def _torque_probe(self):
        """Privileged diagnostic: separate yaw excitation from axial preload."""
        manager=self._robot_manager
        origin=manager.get_gripper_center_pose()
        offset=0.
        target=float(self.params.get('probe_axial_target_N',18.))
        amplitude=np.deg2rad(float(self.params.get('probe_amplitude_deg',5.)))
        settle_steps=360
        period_steps=480
        total_steps=settle_steps+3*period_steps
        self.metadata['diagnostic_protocol']={
            'axial_target_N':target,'yaw_amplitude_deg':float(np.rad2deg(amplitude)),
            'cycles':3,'period_s':period_steps*self.cfg.sim.dt,
            'control':'privileged axial force with commanded small yaw; not final RGB control'}
        for tick in range(total_steps):
            self.phase='diagnostic_axial_hold' if tick<settle_steps else 'diagnostic_yaw_probe'
            angle=0. if tick<settle_steps else amplitude*np.sin(
                2*np.pi*(tick-settle_steps)/period_steps)
            if tick%4==0:
                self.probe_command={'probe_force_target_N':target,
                    'probe_yaw_command_deg':float(np.rad2deg(angle)),
                    'probe_z_correction_m':offset}
                row=self._record()
                if row['seat_force_N']>180 or row['torque_Nm']>self.target_torque*2.5:
                    self.failure='diagnostic_overload';return True
                if row['center_drift_m']>.004 or row['tilt_deg']>10:
                    self.failure='diagnostic_alignment_lost';return True
                # Raising the gripper relieves axial seat compression.
                speed=float(np.clip((row['seat_force_N']-target)*.00004,-.0004,.0004))
                if abs(row['seat_force_N']-target)<.25:speed=0.
            offset=float(np.clip(offset+speed*self.cfg.sim.dt,-.0015,.0015))
            q=t3d.quaternions.qmult(
                t3d.quaternions.axangle2quat([0.,0.,1.],angle),origin.q)
            center=Pose(origin.p+[0.,0.,self.controller_lead_m*angle/(2*np.pi)+offset],q)
            if not self._ik_to_center(center):return True
        self.metadata['diagnostic_completed']=True
        self._record()
        return True

    def _play_once(self):
        self._expert_active=True
        try:
            self._perform_bulb_episode()
        except BulbDamageStop:
            pass
        finally:
            self._expert_active=False
        if self.controller not in ('diagnostic_grip_probe','diagnostic_torque_probe','diagnostic_load_grid'):
            self._finish_bulb_episode()

    def _submit_review_end(self):
        """Submit a deliberately partial action using the ordinary public END."""
        from ._force_task_utils import execute_joint_target
        manager=self._robot_manager
        joints=manager.robot.data.joint_pos[:,manager._arm_ids][0].clone()
        start=self.step_count
        self.phase='review_partial_submission'
        self.metadata['review_partial_submission']=dict(
            scenario=self.params.get('bulb_review_scenario'),
            start_step=start,action='stationary public joint target; ordinary END scorer')
        for _ in range(100):
            executed,accepted=execute_joint_target(self,arm=joints,ticks=2)
            if self._accepted_result is not None or not executed or accepted:
                break
        self.metadata['review_partial_submission']['verdict_step']=self.metadata.get('bulb_physical_verdict_step')
        self.metadata['review_partial_submission']['return_after_terminal_review_step']=self.step_count

    def _perform_bulb_episode(self):
        self._approach_bulb()
        self.previous_pose=self.bulb.get_pose()
        local=(world_contact_vertices(self.bulb)-self.previous_pose.p)@self.previous_pose.R
        self.bulb_grip_mask=local[:,2]>.045
        self.cumulative_yaw=0.
        self.calibration=None
        self.unloaded_reference=read_rgb(self)
        import cv2
        for name,image in self.unloaded_reference.items():
            cv2.imwrite(str(self.work/f'unloaded_{name}.png'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
        if self.controller=='diagnostic_grip_probe':
            return self._run_grip_probe()
        if self.controller=='diagnostic_crush':
            self.reference=self.unloaded_reference
            self.phase='crush_probe'
            self._close_bulb(1)
            self.delay(120,is_save=True)
            self.failure='crush_threshold_not_reached'
            return
        template=(self.unloaded_reference if self.controller!='marker_rgb'
                  and self.params.get('reference_state','episode_unloaded')=='episode_unloaded' else None)
        self.metadata['rgb_reference_state']='episode_unloaded' if template is not None else 'grasp'
        if self.controller=='marker_rgb':
            model_path=Path(self.params['calibration'])
            self.calibration=json.loads(model_path.read_text())
            check_bulb_observer_contract(self.params,self.calibration)
            if self.calibration.get('reference_state')=='episode_unloaded':
                template=self.unloaded_reference
            elif self.calibration.get('reference_images'):
                import cv2
                template={}
                for name,relative in self.calibration['reference_images'].items():
                    image=cv2.imread(str(model_path.parent/relative))
                    if image is None:raise ValueError('Missing recorded tactile calibration reference')
                    template[name]=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            self.metadata['rgb_reference_state']=self.calibration.get('reference_state','grasp')
        extractor=(self.calibration or {}).get('feature_extractor',self.params.get('image_feature_extractor','flow'))
        if extractor=='tracked_flow_rgb_grid':
            self.reference=template if template is not None else self.unloaded_reference
            self._tracked_image_features()
        for i in range(1,15):
            if not self.plan_success or self._accepted_result is not None: break
            self._close_bulb(i)
            self.delay(20)
            self.reference=template if template is not None else read_rgb(self)
            if i==1:
                if not probe_both_directions(self):break
                if self.params.get('bulb_review_scenario')=='probe_only':
                    self.failure='diagnostic_probe_completed_without_formal_screw';break
                if self.params.get('bulb_review_scenario')=='wrong_direction':
                    self.controller_spin_sign *= -1
                    self.metadata['review_wrong_direction_override_after_safe_probe']=True
                if self.params.get('efficient_wrist_motion',False):
                    self.metadata['probe_grasp_center']=self.metadata.get('grasp_1_center')
                    self.phase='prepare_formal_wrist'
                    self._release_bulb(0)
                    if self.failure or not self._position_empty_wrist():break
                    self._close_bulb(1,force_recenter=True)
                    self.delay(12,is_save=True)
            self.phase='screw'
            self._thread_shear_reference=read_rgb(self)
            self._record()
            if self._screw_segment(i) or self.failure: break
            if i==1 and self.params.get('bulb_review_scenario') in ('wrong_direction','under_tightened'):
                self._submit_review_end()
                break
            if i<14:
                self.phase='regrasp'
                self._release_bulb(i)
                self._return_gripper_yaw(i)
                self._record()
        if self._accepted_result is not None:
            return  # Preserve the physical END verdict before reveal motions.
        self._record()
        if self.failure is None and self.controller not in ('diagnostic_torque_probe','diagnostic_load_grid'):
            m=self.measured
            if abs(m.get('torque_Nm',0)-self.target_torque)>self.torque_tolerance:
                self.failure='torque_outside_acceptance'
            elif self.held_steps<132:
                self.failure='physical_hold_duration_not_met'
            elif m.get('phase_error_m',1)>=.002 or m.get('center_drift_m',1)>=.002 or m.get('tilt_deg',90)>=6:
                self.failure='pose_outside_acceptance'
        self.metadata['bulb_final']=self.measured
        self.metadata['bulb_hold_steps']=self.held_steps
        self.metadata['bulb_failure']=self.failure
        self.metadata['bulb_trace_path']=str(self.work/'trace.jsonl')

    def check_success(self):
        if self._accepted_result is not None:
            return self._accepted_result
        if not self._submission_requested:return False
        m=self.measured
        if self.controller in ('diagnostic_torque_probe','diagnostic_load_grid','diagnostic_grip_probe','diagnostic_crush'):
            return False
        return (self.failure is None and self.plan_success and self.held_steps>=132
                and m.get('signed_torque_Nm',0)*self.thread_spec.handedness>0.
                and self._probe_physical_complete and 1.<=m.get('formal_net_turns',0)<=3.
                and m.get('formal_advance_m',0)>=self.thread_spec.nominal_free_travel_m-.002
                and abs(m.get('torque_Nm',0)-self.target_torque)<=self.torque_tolerance
                and m.get('phase_error_m',1)<.002 and m.get('center_drift_m',1)<.002
                and m.get('tilt_deg',90)<6 and m.get('seat_force_N',0)>3.)

    def get_frame_shot(self, obs):
        return BaseTask.get_frame_shot(self, obs)

