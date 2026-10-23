"""Spawner primitives: mechanism only, no task numbers.

Quaternion convention throughout is XYZW (isaaclab 3.0).
"""

# isaaclab's generated .pyi leave its member types partially Unknown (ProxyArray and
# spawner/actuator surfaces), and pxr is a Boost namespace package with no root stub;
# the boundary calls below are runtime-verified.

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast

import isaaclab.sim as sim_utils
import numpy.typing as npt
import torch
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim.schemas import RigidBodyBaseCfg
from isaaclab.sim.spawners.lights import LightCfg
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from .spec import FixedJoint, Joint, PrismaticJoint, RevoluteJoint

_IDENTITY_XYZW = (0.0, 0.0, 0.0, 1.0)


def rigid_object_cfg(
    usd_path: str,
    prim_path: str,
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rot: tuple[float, float, float, float] = _IDENTITY_XYZW,
    *,
    kinematic: bool = False,
    density: float | None = None,
) -> RigidObjectCfg:
    """Build a rigid-object cfg for a dynamic body or a kinematic visual proxy.

    ``rigid_props`` carries a :class:`PhysxRigidBodyCfg` fragment: isaaclab's USD
    spawner only authors ``RigidBodyAPI`` when such a fragment is present, so passing
    it is what gives the actor dynamics. A proxy is kinematic, gravity-free, and has
    collisions disabled so it can follow UIPC without contacting the PhysX robot.
    ``density`` authors a PhysX mass fragment on the spawned body; leave it ``None``
    to keep the USD's own mass properties.
    """
    mass_props = (
        None
        if density is None
        else sim_utils.MassCfg(density=density)
    )
    mass_props_create_if_missing = density is not None
    if kinematic:
        return RigidObjectCfg(
            prim_path=prim_path,
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd_path,
                mass_props=mass_props,
                mass_props_create_if_missing=mass_props_create_if_missing,
                rigid_props=PhysxRigidBodyCfg(disable_gravity=True),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
        )
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            mass_props=mass_props,
            mass_props_create_if_missing=mass_props_create_if_missing,
            rigid_props=PhysxRigidBodyCfg(disable_gravity=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
    )


def spawn_rigid(
    usd_path: str,
    prim_path: str,
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rot: tuple[float, float, float, float] = _IDENTITY_XYZW,
    *,
    kinematic: bool = False,
    density: float | None = None,
) -> RigidObject:
    """Spawn a USD rigid body, or a kinematic visual proxy when ``kinematic``."""
    cfg = rigid_object_cfg(
        usd_path,
        prim_path,
        pos,
        rot,
        kinematic=kinematic,
        density=density,
    )
    # The stub marks the base class abstract; the runtime class is concrete.
    rigid = cast(RigidObject, cast(Any, RigidObject)(cfg))
    if kinematic:
        make_proxy_kinematic(prim_path)
    return rigid


def make_proxy_kinematic(prim_path: str, *, stage: Usd.Stage | None = None) -> None:
    """Make a spawned body kinematic and disable collision under it.

    ``UsdFileCfg`` creates the rigid-body API through ``PhysxRigidBodyCfg``; the
    kinematic and collision-enabled flags live on the solver-common USD schemas,
    so apply them directly after spawn and before ``SimulationContext.reset()``.
    """
    stage = stage or sim_utils.get_current_stage()
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        raise RuntimeError(f"proxy prim {prim_path!r} was not spawned")
    body_api = UsdPhysics.RigidBodyAPI.Apply(root)
    body_api.CreateKinematicEnabledAttr().Set(True)
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_api = UsdPhysics.CollisionAPI(prim)
            collision_api.CreateCollisionEnabledAttr().Set(False)


def disable_named_collisions(
    root_prim_path: str,
    name: str | None = None,
    *,
    prefix: str | None = None,
    stage: Usd.Stage | None = None,
) -> int:
    """Disable collision on every matching prim under ``root_prim_path``.

    Pass ``name`` for an exact prim name or ``prefix`` for a name prefix.  The
    GsMini USD carries invisible ``pad_proxy`` cubes for the PhysX A path; in the
    UIPC B-prime path the soft pads own contact, so those cubes must not block
    finger closure.  The XenseWS tactile adapters are sensor mounts rather than
    jaw geometry, so the Robotiq embodiment disables them by prefix.  Prims are
    kept (A path untouched when not called).
    """
    if (name is None) == (prefix is None):
        raise ValueError("pass exactly one of name or prefix")
    stage = stage or sim_utils.get_current_stage()
    root = stage.GetPrimAtPath(root_prim_path)
    if not root.IsValid():
        raise RuntimeError(f"collision root {root_prim_path!r} is missing")
    count = 0
    for prim in Usd.PrimRange(root):
        prim_name = prim.GetName()
        if name is not None:
            matches = prim_name == name
        else:
            matches = prim_name.startswith(prefix or "")
        if not matches or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
        count += 1
    return count


def sync_kinematic_proxy(
    proxy: RigidObject, pose_xyzw: npt.ArrayLike, *, device: torch.device | str
) -> None:
    """Write an authoritative pose to a kinematic proxy and clear its velocity."""
    pose = torch.as_tensor(pose_xyzw, dtype=torch.float32, device=device).unsqueeze(0)
    if pose.shape != (1, 7):
        raise ValueError(f"proxy pose must be (7,), got {tuple(pose.shape[1:])}")
    proxy.write_root_pose_to_sim(pose)
    proxy.write_root_velocity_to_sim(
        torch.zeros((1, 6), dtype=torch.float32, device=device)
    )


def spawn_static(
    usd_path: str,
    prim_path: str,
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rot: tuple[float, float, float, float] = _IDENTITY_XYZW,
) -> None:
    """Spawn a USD prop as static geometry (no rigid body) — e.g. the plate.

    Some prop USDs author a *dynamic* ``RigidBodyAPI`` (``kinematicEnabled=0``);
    spawned as-is such a "static" fixture falls under gravity and carries every actor
    resting on it down through the world. Pin any authored rigid body kinematic so
    fixtures stay put; props with no rigid body are unaffected (the legacy writer
    touches only prims that already carry the schema).
    """
    sim_utils.spawn_from_usd(
        prim_path,
        sim_utils.UsdFileCfg(
            usd_path=usd_path,
            rigid_props=RigidBodyBaseCfg(kinematic_enabled=True),
        ),
        translation=pos,
        orientation=rot,
    )


def create_joint(spec: Joint) -> None:
    """Author the USD physics joint a :class:`Joint` spec describes.

    Runs at build time, before ``SimulationContext.reset()``. When the parent prim
    carries no ``RigidBodyAPI`` (a static prop), body0 stays empty — PhysX attaches
    the joint to the world — but the anchor still resolves through the parent's stage
    pose, so the spec's parent-relative frame convention holds for static fixtures.
    """
    stage = sim_utils.get_current_stage()
    child_prim = _stage_prim(stage, spec.child)
    child_pos_w, child_rot_w = _world_pose(child_prim)
    body0_target: str | None = None
    if spec.parent is None:
        anchor_pos_w, anchor_rot_w = spec.pos, spec.rot
        local0_pos, local0_rot = spec.pos, spec.rot
    else:
        parent_prim = _stage_prim(stage, spec.parent)
        parent_pos_w, parent_rot_w = _world_pose(parent_prim)
        anchor_pos_w = _add3(parent_pos_w, _quat_rotate(parent_rot_w, spec.pos))
        anchor_rot_w = _quat_mul(parent_rot_w, spec.rot)
        if _is_rigid_body(parent_prim):
            body0_target = spec.parent
            local0_pos, local0_rot = spec.pos, spec.rot
        else:
            local0_pos, local0_rot = anchor_pos_w, anchor_rot_w
    inv_child = _quat_conjugate(child_rot_w)
    local1_pos = _quat_rotate(inv_child, _sub3(anchor_pos_w, child_pos_w))
    local1_rot = _quat_mul(inv_child, anchor_rot_w)

    if isinstance(spec, FixedJoint):
        usd_type = "PhysicsFixedJoint"
    elif isinstance(spec, RevoluteJoint):
        usd_type = "PhysicsRevoluteJoint"
    elif isinstance(spec, PrismaticJoint):
        usd_type = "PhysicsPrismaticJoint"
    else:
        raise TypeError(f"unsupported joint spec {type(spec).__name__!r}")
    prim = stage.DefinePrim(spec.prim_path, usd_type)

    if body0_target is not None:
        _set_rel(prim, "physics:body0", body0_target)
    _set_rel(prim, "physics:body1", spec.child)
    _set_attr(
        prim, "physics:localPos0", Sdf.ValueTypeNames.Point3f, Gf.Vec3f(*local0_pos)
    )
    _set_attr(
        prim, "physics:localRot0", Sdf.ValueTypeNames.Quatf, _gf_quatf(local0_rot)
    )
    _set_attr(
        prim, "physics:localPos1", Sdf.ValueTypeNames.Point3f, Gf.Vec3f(*local1_pos)
    )
    _set_attr(
        prim, "physics:localRot1", Sdf.ValueTypeNames.Quatf, _gf_quatf(local1_rot)
    )

    if isinstance(spec, FixedJoint):
        return
    _set_attr(prim, "physics:axis", Sdf.ValueTypeNames.Token, spec.axis, uniform=True)
    if spec.limits is not None:
        low, high = spec.limits
        if isinstance(spec, RevoluteJoint):
            # USD physics authors angular limits in degrees; the spec carries radians.
            low, high = math.degrees(low), math.degrees(high)
        _set_attr(prim, "physics:lowerLimit", Sdf.ValueTypeNames.Float, low)
        _set_attr(prim, "physics:upperLimit", Sdf.ValueTypeNames.Float, high)
    if spec.damping > 0.0:
        drive = "angular" if isinstance(spec, RevoluteJoint) else "linear"
        UsdPhysics.DriveAPI.Apply(prim, drive)
        _set_attr(
            prim, f"drive:{drive}:physics:stiffness", Sdf.ValueTypeNames.Float, 0.0
        )
        _set_attr(
            prim,
            f"drive:{drive}:physics:damping",
            Sdf.ValueTypeNames.Float,
            spec.damping,
        )


def _stage_prim(stage: Usd.Stage, prim_path: str) -> Usd.Prim:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"joint endpoint {prim_path!r} is not on the stage")
    return prim


def _is_rigid_body(prim: Usd.Prim) -> bool:
    return prim.HasAPI(UsdPhysics.RigidBodyAPI)


def _world_pose(
    prim: Usd.Prim,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """A spawned prim's world (pos, quat XYZW), extracted to plain floats once."""
    xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    t = xf.ExtractTranslation()
    q = xf.ExtractRotationQuat()
    qi = q.GetImaginary()
    return (
        (float(t[0]), float(t[1]), float(t[2])),
        (float(qi[0]), float(qi[1]), float(qi[2]), float(q.GetReal())),
    )


def _gf_quatf(quat_xyzw: tuple[float, float, float, float]) -> Gf.Quatf:
    """Gf.Quatf from XYZW; the Gf constructor takes (real, imaginary)."""
    x, y, z, w = quat_xyzw
    return Gf.Quatf(w, Gf.Vec3f(x, y, z))


def _set_rel(prim: Usd.Prim, name: str, target: str) -> None:
    rel = prim.CreateRelationship(name, False)
    rel.SetTargets([Sdf.Path(target)])


def _set_attr(
    prim: Usd.Prim,
    name: str,
    type_name: Sdf.ValueTypeName,
    value: Any,
    *,
    uniform: bool = False,
) -> None:
    """Author a schema-builtin attribute; pxr's Boost stubs type every call ``None``."""
    attr = prim.CreateAttribute(
        name,
        type_name,
        False,
        Sdf.VariabilityUniform if uniform else Sdf.VariabilityVarying,
    )
    attr.Set(value)


def _add3(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub3(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _quat_conjugate(
    q: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return (-q[0], -q[1], -q[2], q[3])


def _quat_mul(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _quat_rotate(
    q: tuple[float, float, float, float], v: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Rotate ``v`` by unit quat ``q`` (XYZW).

    ``v + 2w cross(qv, v) + 2 cross(qv, cross(qv, v))`` with ``qv`` the vector part.
    """
    x, y, z, w = q
    cx, cy, cz = (y * v[2] - z * v[1], z * v[0] - x * v[2], x * v[1] - y * v[0])
    dx, dy, dz = (y * cz - z * cy, z * cx - x * cz, x * cy - y * cx)
    return (
        v[0] + 2.0 * (w * cx + dx),
        v[1] + 2.0 * (w * cy + dy),
        v[2] + 2.0 * (w * cz + dz),
    )


def spawn_articulation(cfg: ArticulationCfg) -> Articulation:
    """Spawn an articulated actor (drawer, gear, robot); pose/PD come from ``cfg``."""
    # The stub marks the base class abstract; the runtime class is concrete.
    return cast(Articulation, cast(Any, Articulation)(cfg))


def apply_init_state(articulation: Articulation) -> None:
    """Write the cfg's ``init_state`` into the live sim and hold it.

    isaaclab 3.0 does not auto-apply ``init_state`` on ``sim.reset()`` (joints stay
    at the USD-authored pose); it only seeds ``default_joint_pos``. Write the pose and
    set it as the actuator target so the PD holds it. Run after ``sim.reset()``.
    """
    data = articulation.data
    joint_pos = data.default_joint_pos.torch
    articulation.write_joint_position_to_sim_index(position=joint_pos)
    joint_vel = data.default_joint_vel.torch
    articulation.write_joint_velocity_to_sim_index(velocity=joint_vel)
    articulation.actuators.target_command.set_position_index(value=joint_pos)
    articulation.write_data_to_sim()


def spawn_camera(cfg: CameraCfg) -> Camera:
    """Create a camera.

    Must run before ``SimulationContext.reset()``: a camera's render product is
    destroyed when the timeline stops, so a camera made after reset never renders.
    """
    return Camera(cfg)


def spawn_light(cfg: LightCfg, prim_path: str = "/World/light") -> None:
    """Spawn a light (DomeLight by default); color/intensity come from ``cfg``."""
    # The stub unions ``str`` into func (configclass export machinery); it is the
    # spawner callable at runtime.
    func = cast(Callable[[str, LightCfg], None], cfg.func)
    func(prim_path, cfg)
