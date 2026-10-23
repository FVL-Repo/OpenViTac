"""Translate a pure :class:`SceneSpec` into the tactile runtime's :class:`TactileCfg`.

This is the only sim-side place that decides which sensor binding becomes a pad.
The old sensor-to-robot ``gripper_kind`` switch and the left/right derivation live
only in the deprecated compatibility adapter; new scene specs carry explicit
bindings, one pad per binding.
"""

from __future__ import annotations

from collections.abc import Iterable

from openvitac_spec import SceneSpec, SensorBinding
from openvitac_tactile.config import (
    ObjectCfg,
    PadCfg,
    PadFrameCfg,
    PadGeometryCfg,
    PadMaterialCfg,
    ResetCfg,
    Side,
    TactileCfg,
)

from .assets import asset
from .tactile_bridge import objects_from_actors


def _side(binding: SensorBinding) -> Side:
    # UIPC pad frames still use the left/right sign convention internally.  A
    # non-side pad key is representable in the spec, but the current worker would
    # have to learn a generic frame sign; fail loud until that lands.
    if binding.pad_key not in ("left", "right"):
        raise NotImplementedError(
            f"sensor pad key {binding.pad_key!r} is not 'left'/'right'; the UIPC "
            "pad frame sign is still side-based"
        )
    return binding.pad_key


def _pad(binding: SensorBinding) -> PadCfg:
    sensor = binding.sensor
    is_gsmini = sensor.kind in ("gsmini", "neote")
    return PadCfg(
        side=_side(binding),
        # The bridge uses the pad name as the observation/data key.
        name=binding.pad_key,
        gelpad_prim_path=binding.gelpad_prim_path,
        frame=PadFrameCfg(
            body_name=binding.body_name,
            body_prim_path=binding.body_prim_path,
            pos=(0.0, 0.0, 0.0),
            rot_xyzw=(0.0, 0.0, 0.0, 1.0),
            back_axis="z" if is_gsmini else "y",
            back_is_min=True if is_gsmini else None,
            attach_radius_m=binding.mount.attach_radius_m,
        ),
        geometry=(
            PadGeometryCfg(
                kind="tet_mesh",
                mesh_path=asset("embodiment/gsmini_gelpad_tet.npz"),
            )
            if is_gsmini
            else PadGeometryCfg()
        ),
        material=PadMaterialCfg(
            youngs_mpa=sensor.youngs_mpa,
            constraint_strength=sensor.strength,
        ),
        image_width=sensor.width,
        image_height=sensor.height,
        marker_rows=sensor.marker_rows,
        marker_cols=sensor.marker_cols,
        render_kinds=tuple(
            kind
            for kind in binding.streams
            if kind in ("rgb", "marker_rgb", "depth")
        ),
    )


def tactile_plan_from_scene(
    spec: SceneSpec,
    *,
    objects: Iterable[ObjectCfg] | None = None,
    include_uipc_fingers: bool = False,
) -> TactileCfg:
    """Build the tactile config from ``spec.sensors`` and ``spec.actors``.

    ``objects`` defaults to the dynamic UIPC/PhysX actors in the scene.  Settling is
    the strictest binding protocol (max stable/max steps) so a mixed pair does not
    let one pad publish before it settles.
    """
    pads = tuple(_pad(binding) for binding in spec.sensors)
    settle = [binding.sensor.settle for binding in spec.sensors]
    stable = [p.stable_steps for p in settle if p is not None]
    max_steps = [p.max_steps for p in settle if p is not None]
    return TactileCfg(
        enabled=bool(pads),
        pads=pads,
        objects=tuple(
            objects_from_actors(spec.actors) if objects is None else objects
        ),
        include_uipc_fingers=include_uipc_fingers,
        reset=ResetCfg(
            settle_stable_steps=max(stable) if stable else None,
            settle_max_steps=max(max_steps) if max_steps else 0,
        ),
    )
