"""The tool contract of a robot asset, as pure data.

A scripted expert only ever asks the asset two things: *where is the grasp point*
and *how do the fingers turn a command into a pad gap*.  Both are properties of
the robot USD, not of a gripper model:

* the fixed ``tool_center`` frame is the pad midpoint (the fingers open
  symmetrically, so the midpoint does not move with the opening);
* ``customLayerData.openvitac`` declares the finger joints, the pad extents and
  the pad-face gap at the closed limit;
* the finger joints' own frames (``physics:localRot0``) give the closing axis.

:func:`read_tool_geometry` turns those into a :class:`ToolGeometry` for one
asset.  That is the only place in the package that touches ``pxr``, and it
imports it lazily, so ``openvitac_spec`` (and everything that imports it on a
machine without a simulator) still loads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .types import Vec3

# Root-layer dictionary the robot assets declare their grasp contract in.
CONTRACT_KEY = "openvitac"


@dataclass(frozen=True)
class ToolGeometry:
    """Grasp geometry of the tool riding one arm, in the arm's EE-link frame.

    Every field is asset data; nothing here is a gripper-model constant, so a
    new tool needs no code change -- only a ``tool_center`` prim and a contract
    dictionary in its USD.
    """

    # Asset: ``ArmSpec.ee_link``.  The body frame every offset below is expressed
    # in, i.e. the frame the IK target poses are written in.
    ee_link: str
    # Asset: pose of the ``tool_center`` prim minus the EE link's pose, metres in
    # the EE frame.  Constant while the arm moves (both prims are rigid on it).
    center_in_ee: Vec3
    # Asset: the last finger joint's ``physics:localRot0`` X axis in the EE frame,
    # unit, pointing out of the mirrored finger's pad face.  The pair closes on
    # +/- this axis; it is the "pad normal" a face alignment rotates.
    pad_normal_in_ee: Vec3
    # Contract ``pad_extent_m``: the pad body's width, thickness along the closing
    # axis, and height in metres, as the asset declares them.  Informational
    # (coverage/collision); the close law uses ``pad_gap_closed``.
    pad_extent: Vec3
    # Contract ``close_gap_m``: pad-face gap in metres with every finger at the
    # closed limit (``q_closed``).  Measured on the asset, so it already carries
    # the pad thickness and any mounting stand-off.
    pad_gap_closed: float
    # Contract ``open_travel_m``: how far one pad moves, in metres, between the
    # two limits.  Mirrored jaws move one pad each, so the pad-face gap changes
    # by ``finger_count * open_travel_m`` across the whole finger range -- for a
    # prismatic jaw that equals the joint span, for a linkage that swings its
    # pads it is the arc length and is shorter.
    open_travel: float
    # Asset: number of driven finger joints (the contract's ``finger_joints``).
    finger_count: int
    # Asset: (lower, upper) command limit of one finger joint, in the joint's own
    # unit -- metres for a prismatic jaw, radians for a revolute one (USD authors
    # revolute limits in degrees; ``read_tool_geometry`` converts).
    finger_q: tuple[float, float]
    # Contract ``close_gap_at``: which limit ``close_gap_m`` was measured at,
    # ``"lower"`` (the default, every prismatic jaw here) or ``"upper"``.
    closed_at_upper: bool = False

    def __post_init__(self) -> None:
        if not self.ee_link:
            raise ValueError("ToolGeometry.ee_link must be non-empty")
        if len(self.pad_extent) != 3:
            raise ValueError(f"pad_extent must have 3 components: {self.pad_extent}")
        if self.pad_gap_closed < 0.0:
            raise ValueError(f"pad_gap_closed must be >= 0: {self.pad_gap_closed}")
        if self.open_travel < 0.0:
            raise ValueError(f"open_travel must be >= 0: {self.open_travel}")
        if self.finger_count < 1:
            raise ValueError(f"finger_count must be >= 1: {self.finger_count}")
        if self.finger_q[0] >= self.finger_q[1]:
            raise ValueError(f"finger_q must be (lower, upper): {self.finger_q}")
        norm = math.sqrt(sum(c * c for c in self.pad_normal_in_ee))
        if abs(norm - 1.0) > 1e-6:
            raise ValueError(f"pad_normal_in_ee must be unit, |n| = {norm}")

    @property
    def q_closed(self) -> float:
        """Finger command at which the pads are closed."""
        return self.finger_q[1] if self.closed_at_upper else self.finger_q[0]

    @property
    def q_open(self) -> float:
        """Finger command at which the pads are furthest apart."""
        return self.finger_q[0] if self.closed_at_upper else self.finger_q[1]

    @property
    def pad_gap_per_travel(self) -> float:
        """Pad-face gap change per unit of finger command (mirrored jaws)."""
        return self.finger_count * self.open_travel / (self.q_open - self.q_closed)

    def pad_gap(self, q: float) -> float:
        """Pad-face gap in metres with every finger commanded to ``q``."""
        return self.pad_gap_closed + self.pad_gap_per_travel * (q - self.q_closed)

    def finger_command_for_gap(self, gap: float) -> float:
        """Inverse of :meth:`pad_gap`: command putting the pad faces ``gap`` apart."""
        q = self.q_closed + (gap - self.pad_gap_closed) / self.pad_gap_per_travel
        return min(max(q, self.finger_q[0]), self.finger_q[1])


def _quat_xyzw(q: Any) -> tuple[float, float, float, float]:
    """``Gf.Quatf``/``Gf.Quatd`` -> XYZW floats (USD authors them WXYZ)."""
    imag = q.GetImaginary()
    return (float(imag[0]), float(imag[1]), float(imag[2]), float(q.GetReal()))


def read_tool_geometry(usd_path: str | Path, *, ee_link: str) -> ToolGeometry:
    """Read the grasp geometry declared by the robot asset at ``usd_path``.

    ``ee_link`` names the body inside the asset's default prim whose frame the
    offsets are expressed in (``ArmSpec.ee_link``).  The result does not depend
    on the asset's authored joint pose: the tool frame and the pads are rigid on
    the hand and the fingers, and the closing axis comes from the joint frames.

    Raises ``FileNotFoundError`` for a missing asset and ``ValueError`` when the
    asset declares no contract or an incomplete one, so a hand-written tool
    cannot silently fall back to another gripper's numbers.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics  # local: keeps the import pure

    path = Path(usd_path)
    if not path.is_file():
        raise FileNotFoundError(f"robot asset not found: {path}")
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"not a readable USD stage: {path}")
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        raise ValueError(f"{path} declares no defaultPrim")
    root = str(default_prim.GetPath())

    data = stage.GetRootLayer().customLayerData or {}
    contract = data.get(CONTRACT_KEY)
    if not isinstance(contract, dict):
        raise ValueError(
            f"{path}: no customLayerData[{CONTRACT_KEY!r}] contract. A robot asset "
            "used by a scripted expert must declare tool_center, pad_frames, "
            "pad_extent_m, finger_joints and close_gap_m."
        )

    def require(key: str) -> Any:
        try:
            return contract[key]
        except KeyError:
            raise ValueError(f"{path}: contract is missing {key!r}") from None

    def resolve(rel: str) -> str:
        """Contract prim paths are absolute in the shipped assets; accept both."""
        for candidate in (rel, f"{root}/{rel.lstrip('/')}"):
            if stage.GetPrimAtPath(candidate):
                return candidate
        raise ValueError(f"{path}: contract prim {rel!r} does not exist")

    cache = UsdGeom.XformCache()
    ee_prim = stage.GetPrimAtPath(f"{root}/{ee_link}")
    if not ee_prim:
        raise ValueError(f"{path}: ee_link prim {root}/{ee_link} does not exist")
    tool_prim = stage.GetPrimAtPath(resolve(str(require("tool_center"))))
    m_ee = cache.GetLocalToWorldTransform(ee_prim)
    m_tool = cache.GetLocalToWorldTransform(tool_prim)
    # pxr transforms are row-vector: "tool in the EE frame" is M_tool * M_ee^-1.
    center = (m_tool * m_ee.GetInverse()).ExtractTranslation()

    joint_names = list(require("finger_joints"))
    if not joint_names:
        raise ValueError(f"{path}: contract lists no finger_joints")
    axes: list[Vec3] = []
    limits: list[tuple[float, float]] = []
    revolute = False
    for name in joint_names:
        prims = [
            prim
            for prim in stage.Traverse()
            if prim.GetName() == name and prim.IsA(UsdPhysics.Joint)
        ]
        if len(prims) != 1:
            raise ValueError(f"{path}: expected one joint prim named {name!r}")
        joint = prims[0]
        is_prismatic = joint.IsA(UsdPhysics.PrismaticJoint)
        is_revolute = joint.IsA(UsdPhysics.RevoluteJoint)
        if not (is_prismatic or is_revolute):
            raise NotImplementedError(
                f"{path}: finger joint {name!r} is {joint.GetTypeName()}; only a "
                "sliding (prismatic) or rotating (revolute) jaw moves one pad"
            )
        if is_revolute != revolute and limits:
            raise NotImplementedError(
                f"{path}: finger joints mix sliding and rotating jaws"
            )
        revolute = is_revolute
        lower = joint.GetAttribute("physics:lowerLimit").Get()
        upper = joint.GetAttribute("physics:upperLimit").Get()
        if lower is None or upper is None:
            raise ValueError(f"{path}: joint {name!r} declares no travel limits")
        if revolute:
            # USD authors revolute limits in degrees; every consumer of this
            # geometry speaks radians.
            lower, upper = math.radians(lower), math.radians(upper)
        limits.append((float(lower), float(upper)))
        body0 = joint.GetRelationship("physics:body0").GetTargets()
        if not body0:
            raise ValueError(f"{path}: joint {name!r} declares no physics:body0")
        m_body0 = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(str(body0[0])))
        # Joint frame in its parent body, then in world, then in the EE frame.
        # ``SetRotate`` replaces the matrix, so the translation is never set.
        x, y, z, w = _quat_xyzw(joint.GetAttribute("physics:localRot0").Get())
        m_joint = Gf.Matrix4d().SetRotate(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
        axis = (m_joint * m_body0 * m_ee.GetInverse()).ExtractRotationMatrix()
        token = str(joint.GetAttribute("physics:axis").Get() or "X").upper()
        # Row i of a row-vector rotation matrix is the image of basis vector i, so
        # the joint axis token picks the row.
        row = axis.GetRow({"X": 0, "Y": 1, "Z": 2}[token])
        norm = math.sqrt(row[0] ** 2 + row[1] ** 2 + row[2] ** 2)
        if norm < 1e-6:
            raise ValueError(f"{path}: joint {name!r} has a degenerate axis")
        axes.append((row[0] / norm, row[1] / norm, row[2] / norm))

    extent = tuple(float(v) for v in require("pad_extent_m"))
    if len(extent) != 3:
        raise ValueError(f"{path}: pad_extent_m must have 3 components: {extent}")
    pad_names = list(require("pad_frames"))
    if len(pad_names) != 2:
        raise ValueError(f"{path}: pad_frames must list two pads: {pad_names}")
    # A sliding jaw moves its pad along the joint axis, so the closing axis is
    # that axis.  A rotating one swings its pad around the joint: the closing
    # axis is then the one between the two pad frames.
    if revolute:
        pad_origins = [
            (
                cache.GetLocalToWorldTransform(stage.GetPrimAtPath(resolve(p)))
                * m_ee.GetInverse()
            ).ExtractTranslation()
            for p in pad_names
        ]
        delta = pad_origins[0] - pad_origins[1]
        norm = math.sqrt(delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2)
        if norm < 1e-6:
            raise ValueError(f"{path}: the two pad frames coincide")
        pad_normal = (delta[0] / norm, delta[1] / norm, delta[2] / norm)
    else:
        pad_normal = axes[-1]

    spans = {round(hi - lo, 9) for lo, hi in limits}
    if len(spans) != 1:
        raise ValueError(f"{path}: finger joints have different travel: {limits}")
    last = limits[-1]
    span = last[1] - last[0]
    declared_travel = float(require("open_travel_m"))
    if not revolute and abs(declared_travel - span) > 1e-4:
        # A sliding jaw moves its pad one metre per metre of command, so the
        # declared pad travel is the joint span.  A linkage swings its pad along
        # an arc: the travel is metrically shorter and no such identity holds.
        raise ValueError(
            f"{path}: open_travel_m={declared_travel} disagrees with the joint "
            f"limits {last} (span {span:.6f})"
        )
    closed_at = str(contract.get("close_gap_at", "lower")).lower()
    if closed_at not in ("lower", "upper"):
        raise ValueError(f"{path}: close_gap_at must be 'lower' or 'upper'")
    return ToolGeometry(
        ee_link=ee_link,
        center_in_ee=(float(center[0]), float(center[1]), float(center[2])),
        pad_normal_in_ee=pad_normal,
        pad_extent=(extent[0], extent[1], extent[2]),
        pad_gap_closed=float(require("close_gap_m")),
        open_travel=declared_travel,
        finger_count=len(joint_names),
        finger_q=last,
        closed_at_upper=closed_at == "upper",
    )
