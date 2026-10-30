"""insert_block selection/success/expert tests without Isaac."""

from __future__ import annotations

import itertools
import logging
import math
import unittest
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from openvitac_spec import GripperCapability
from openvitac_tasks import Action
from openvitac_tasks.motion import (
    HAND_TO_GEL_FACE,
    hand_for_pad_midpoint,
    pad_close_q,
    pad_midpoint,
)
from openvitac_tasks.tasks.insert_block import (
    _BLOCK_WIDTH_BY_ACTOR,
    _BLOCK_XY_NOISE,
    _BOX_SIZE,
    _CLOSE_Q_BY_BLOCK,
    _GRASP_HEIGHT,
    _INSERT_DEPTH,
    _RELEASE_RETRACT,
    _SETTLE_STEPS,
    InsertBlock,
)

_BOX = (0.48, -0.12, 0.0026)
_ACTOR = {
    "cube": "block_blue_cube",
    "half_cylinder": "block_blue_half_cylinder",
    "hexagon": "block_red_hexagonal_prism",
}
_BLOCKS = {
    "block_blue_cube": ((0.45, 0.15, 0.004), (0.0, 0.0, 0.0, 1.0)),
    "block_blue_half_cylinder": ((0.37, 0.08, 0.004), (0.0, 0.0, 0.0, 1.0)),
    "block_red_hexagonal_prism": ((0.36, 0.24, 0.004), (0.0, 0.0, 0.0, 1.0)),
}
_HOLES = {
    "cube": (-0.035, 0.03),
    "half_cylinder": (0.034, -0.018117),
    "hexagon": (-0.026, -0.036),
}
_DESCRIPTIONS = {
    "cube": "blue cube",
    "half_cylinder": "blue half cylinder",
    "hexagon": "red hexagonal prism",
}


class _Gripper:
    open_qpos = 0.04
    close_qpos = 0.0

    @property
    def capabilities(self) -> frozenset[GripperCapability]:
        return frozenset({"binary_close"})

    def __init__(self, world: _World) -> None:
        self._world = world
        self._qpos = 0.02

    def set_target(self, qpos: float) -> None:
        world = self._world
        # The task commands the shape's width-derived close q, not the limit.
        if qpos == self.open_qpos:
            if self._qpos != self.open_qpos:
                world.events.append(("open", 1))
                # The real block falls through the top plate to the box floor.
                if world.holding is not None:
                    x, y, _ = world.poses[world.holding][0]
                    world.poses[world.holding] = (
                        (x, y, _BOX[2] + 0.006),
                        (0.0, 0.0, 0.0, 1.0),
                    )
                world.holding = None
        elif qpos != self._qpos:
            world.events.append(("close", qpos))
            world.holding = world.actor
        self._qpos = qpos

    def joint_pos(self) -> float:
        return self._qpos


class _Arm:

    def joint_pos(self) -> Any:
        return None

    def gripper_contact(self, *, force_threshold: float = 1.0) -> tuple[bool, bool]:
        return (False, False)

    def __init__(self, world: _World) -> None:
        self._world = world

    @property
    def gripper(self) -> _Gripper:
        return self._world._gripper

    def ee_pose(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        return self._world.ee, self._world.quat

    def set_ee_target(
        self,
        pos: tuple[float, float, float],
        quat: tuple[float, float, float, float],
    ) -> None:
        world = self._world
        if world.holding is not None:
            bx, by, bz = world.poses[world.holding][0]
            world.poses[world.holding] = (
                (
                    bx + pos[0] - world.ee[0],
                    by + pos[1] - world.ee[1],
                    bz + pos[2] - world.ee[2],
                ),
                world.poses[world.holding][1],
            )
        world.ee = pos
        world.quat = quat
        world.events.append(("move", pos))


class _World:

    @property
    def logger(self) -> logging.LoggerAdapter[logging.Logger]:
        return logging.LoggerAdapter(logging.getLogger(__name__))

    def __init__(self, target: str) -> None:
        self.target = target
        self.actor = _ACTOR.get(target, "block_blue_cube")
        self.poses = {name: pose for name, pose in _BLOCKS.items()}
        self.poses["box_with_holes_yellow"] = (_BOX, (0.0, 0.0, 0.0, 1.0))
        self.ee = (0.30, 0.0, 0.30)
        self.quat = (0.0, 0.0, 0.0, 1.0)
        self.holding: str | None = None
        self.steps = 0
        self.events: list[tuple[Any, ...]] = []
        self._gripper = _Gripper(self)
        self._arm = _Arm(self)

    def apply(self, action: Action) -> None:
        self.arm.set_ee_target(action.ee_pos, action.ee_quat)
        self._gripper.set_target(action.gripper_qpos)
        self.steps += 1

    @property
    def arm(self) -> _Arm:
        return self._arm

    def expert_param(self, name: str) -> str:
        if name == "target_block":
            return self.target
        raise KeyError(name)

    def object_pose(
        self, prim_name: str
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        try:
            return self.poses[prim_name]
        except KeyError:
            raise KeyError(prim_name) from None


def _run(world: _World) -> list[str]:
    """Drive the registered state machine and record its phase sequence."""
    machine = InsertBlock.expert
    state = machine.init()
    phases: list[str] = []
    while True:
        result = machine.step(world, state)
        if result is None:
            return phases
        state, action = result
        phases.append(state.phase.name)
        world.apply(action)


def _run_until_success(world: _World) -> tuple[str, bool]:
    """Drive the machine like the collector; return the first-true phase."""
    machine = InsertBlock.expert
    state = machine.init()
    verdicts = InsertBlock.success(world)
    while True:
        result = machine.step(world, state)
        if result is None:
            raise AssertionError("policy ended before the success gate")
        state, action = result
        world.apply(action)
        if next(verdicts):
            return str(state.phase.name), world.holding is None


def _fires(verdicts: Iterator[bool], ticks: int = _SETTLE_STEPS) -> bool:
    """Advance at most ``ticks`` verdicts; whether the settle gate fires."""
    return any(itertools.islice(verdicts, ticks))


class InsertBlockTests(unittest.TestCase):
    def test_randomizer_samples_all_targets_and_keeps_blocks_on_table(self) -> None:
        seen: set[str] = set()
        base = {
            actor.prim_path.rsplit("/", 1)[-1]: actor
            for actor in InsertBlock.actors
        }
        for seed in range(10):
            randomization = InsertBlock.setup(seed)
            target = randomization.metadata["target_block"]
            assert isinstance(target, str)
            seen.add(target)
            self.assertEqual(
                randomization.instruction,
                f"Insert the {_DESCRIPTIONS[target]} into the matching hole in "
                "the yellow box.",
            )
            self.assertNotIn(
                "/World/actors/box_with_holes_yellow", randomization.actor_poses
            )
            for block in _BLOCKS:
                pose = randomization.actor_poses[f"/World/actors/{block}"]
                self.assertAlmostEqual(pose.position[2], _BLOCKS[block][0][2])
                self.assertLessEqual(
                    abs(pose.position[0] - base[block].pos[0]), _BLOCK_XY_NOISE
                )
                self.assertLessEqual(
                    abs(pose.position[1] - base[block].pos[1]), _BLOCK_XY_NOISE
                )
        self.assertEqual(seen, set(_ACTOR))

    def test_close_position_per_shape(self) -> None:
        # 1 mm gel overtravel per side; measured pad-face gap 7.1276 mm at q=0.
        self.assertAlmostEqual(
            _CLOSE_Q_BY_BLOCK["block_blue_cube"], 0.0154362, places=7
        )
        self.assertAlmostEqual(
            _CLOSE_Q_BY_BLOCK["block_blue_half_cylinder"], 0.0079362, places=7
        )
        self.assertAlmostEqual(
            _CLOSE_Q_BY_BLOCK["block_red_hexagonal_prism"], 0.0110242, places=7
        )
        for target, actor in _ACTOR.items():
            world = _World(target)
            _run(world)
            close_q = next(
                event[1] for event in world.events if event[0] == "close"
            )
            self.assertAlmostEqual(
                float(close_q), _CLOSE_Q_BY_BLOCK[actor], places=12
            )

    def test_success_requires_cavity_in_box_frame(self) -> None:
        retract_z = (
            _BOX[2] + _BOX_SIZE + HAND_TO_GEL_FACE + _RELEASE_RETRACT - _INSERT_DEPTH
        )
        for target, actor in _ACTOR.items():
            world = _World(target)
            world.poses[actor] = (
                (_BOX[0], _BOX[1], _BOX[2] + 0.006),
                (0.0, 0.0, 0.0, 1.0),
            )
            world.ee = (_BOX[0], _BOX[1], retract_z)
            world.arm.gripper.set_target(world.arm.gripper.open_qpos)
            self.assertTrue(_fires(InsertBlock.success(world)))

            # Missed the box and rests on the table beside it.
            world.poses[actor] = (
                (_BOX[0] + 0.10, _BOX[1], _BOX[2]),
                (0.0, 0.0, 0.0, 1.0),
            )
            self.assertFalse(_fires(InsertBlock.success(world)))

            # Still sitting on the box top.
            world.poses[actor] = (
                (_BOX[0], _BOX[1], _BOX[2] + 0.15),
                (0.0, 0.0, 0.0, 1.0),
            )
            self.assertFalse(_fires(InsertBlock.success(world)))

    def test_success_uses_rotated_box_frame(self) -> None:
        world = _World("cube")
        # 90 deg about z: box-local (x, y) maps to world (-y, x).
        world.poses["box_with_holes_yellow"] = (
            _BOX,
            (0.0, 0.0, 0.7071067811865476, 0.7071067811865476),
        )
        local = (0.02, 0.01, 0.006)
        world.poses["block_blue_cube"] = (
            (_BOX[0] - local[1], _BOX[1] + local[0], _BOX[2] + local[2]),
            (0.0, 0.0, 0.0, 1.0),
        )
        world.ee = (
            _BOX[0] - local[1],
            _BOX[1] + local[0],
            _BOX[2] + _BOX_SIZE + HAND_TO_GEL_FACE + _RELEASE_RETRACT - _INSERT_DEPTH,
        )
        world.arm.gripper.set_target(world.arm.gripper.open_qpos)
        self.assertTrue(_fires(InsertBlock.success(world)))

    def test_expert_inserts_each_target_into_its_hole(self) -> None:
        for target, actor in _ACTOR.items():
            world = _World(target)
            phases = _run(world)

            close = next(
                i for i, event in enumerate(world.events) if event[0] == "close"
            )
            open_ = next(
                i for i, event in enumerate(world.events) if event[0] == "open"
            )
            self.assertLess(close, open_)
            self.assertLess(phases.index("CLOSE"), phases.index("OPEN"))
            self.assertEqual(phases[-1], "SETTLE")
            bx, by, bz = world.poses[actor][0]
            self.assertAlmostEqual(bx, _BOX[0] + _HOLES[target][0], places=6)
            self.assertAlmostEqual(by, _BOX[1] + _HOLES[target][1], places=6)
            self.assertAlmostEqual(bz, _BOX[2] + 0.006, places=6)
            self.assertTrue(_fires(InsertBlock.success(world)))

    def test_unknown_target_fails_loud(self) -> None:
        world = _World("banana")
        with self.assertRaises(ValueError):
            _run(world)
        with self.assertRaises(ValueError):
            next(InsertBlock.success(world))

    def test_first_true_is_after_release_retract_and_settle(self) -> None:
        world = _World("cube")
        phase, released = _run_until_success(world)
        self.assertEqual(phase, "SETTLE")
        self.assertTrue(released)
        box_top = world.poses["box_with_holes_yellow"][0][2] + _BOX_SIZE
        self.assertGreaterEqual(
            world.ee[2],
            box_top + HAND_TO_GEL_FACE + _RELEASE_RETRACT - _INSERT_DEPTH,
        )

    def test_expert_respects_step_budget(self) -> None:
        world = _World("half_cylinder")
        _run(world)
        self.assertLess(world.steps, 2000)


if __name__ == "__main__":
    unittest.main()


def _cross_section(points: Any, faces: Any, z: float) -> Any:
    """Intersection points of a triangle mesh with the plane z=const."""
    import numpy as np

    out = []
    for tri in faces:
        v = points[tri]
        d = v[:, 2] - z
        if (d > 0).all() or (d < 0).all():
            continue
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if (d[a] > 0) != (d[b] > 0):
                t = d[a] / (d[a] - d[b])
                out.append(v[a][:2] + t * (v[b][:2] - v[a][:2]))
    return np.array(out)


class InsertBlockWidthTests(unittest.TestCase):
    """The close law must use the cross-section at the pad plane, not the bbox.

    The three shipped blocks are z-extruded prisms, so their width across the
    closing axis (local y) is constant over the pad's face; this test samples
    the usd at three planes inside that face and pins the constants.  A shape
    whose width varies with height would fail here, because a bbox width would
    then over- or under-close.
    """

    def test_close_width_is_the_cross_section_at_the_pad_plane(self) -> None:
        import numpy as np
        from pxr import Usd, UsdGeom

        assets = Path(__file__).resolve().parents[3] / "assets"
        for shape, actor in _ACTOR.items():
            with self.subTest(shape=shape):
                stage = Usd.Stage.Open(
                    str(assets / "insert_block" / f"{actor}.usd")
                )
                mesh = next(
                    p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)
                )
                m = UsdGeom.Mesh(mesh)
                P = np.array(
                    [[q[0], q[1], q[2]] for q in m.GetPointsAttr().Get()]
                )
                idx = np.array(
                    m.GetFaceVertexIndicesAttr().Get(), dtype=np.int64
                )
                counts = np.array(
                    m.GetFaceVertexCountsAttr().Get(), dtype=np.int64
                )
                tris: list[tuple[int, int, int]] = []
                k = 0
                for c in counts:
                    f = idx[k : k + c]
                    k += c
                    for i in range(1, c - 1):
                        tris.append((int(f[0]), int(f[i]), int(f[i + 1])))
                F = np.array(tris, dtype=np.int64)
                # 15..30 mm spans the pad face; the pad centre sits in it.
                for z in (0.015, 0.020, 0.030):
                    section = _cross_section(P, F, z)
                    self.assertGreater(len(section), 0)
                    width = float(np.ptp(section[:, 1]))
                    self.assertAlmostEqual(
                        width,
                        _BLOCK_WIDTH_BY_ACTOR[actor],
                        delta=1e-5,
                        msg=f"{shape}: width at z={z} differs from the "
                        "pinned value",
                    )
                self.assertAlmostEqual(
                    _CLOSE_Q_BY_BLOCK[actor],
                    pad_close_q(_BLOCK_WIDTH_BY_ACTOR[actor]),
                    places=12,
                )


class InsertBlockGraspPlaneTests(unittest.TestCase):
    """The pad pair's mid-plane must sit on the block's grasp mid-plane.

    The pad plane is aligned to the block's +-y face, so the hand's finger axis
    is tilted; placing the hand by a world-z sum left the pads ~17.5 mm off in x
    and ~9.6 mm above the mid-plane, one pad seated and the other pressing air.
    ``hand_for_pad_midpoint`` rotates the pad offset with the hand instead.
    """

    def test_pad_midplane_matches_the_block_midplane(self) -> None:
        # Representative grasp tilt: the hand is inverted (fingers down), its
        # +z axis = (0.150, 0, -0.989) - an 8.6 deg lean off straight down, the
        # measured tilt (17.5 mm of the 116.67 mm hand-to-pad arm).
        half = math.radians(171.4) / 2.0
        quat = (0.0, math.sin(half), 0.0, math.cos(half))
        for shape, actor in _ACTOR.items():
            bx, by, bz = _BLOCKS[actor][0]
            pad_mid = (bx, by, bz + _GRASP_HEIGHT)
            hand = hand_for_pad_midpoint(pad_mid, quat)
            got = pad_midpoint(hand, quat)
            for g, w in zip(got, pad_mid, strict=True):
                self.assertAlmostEqual(
                    g,
                    w,
                    delta=1e-9,
                    msg=f"{shape}: pad mid-plane is off the block mid-plane",
                )
            # The pre-fix target added the pad offset to world z only.
            old_mid = pad_midpoint(
                (bx, by, bz + _GRASP_HEIGHT + HAND_TO_GEL_FACE), quat
            )
            err = math.dist(old_mid, pad_mid)
            self.assertAlmostEqual(
                err,
                0.0177,
                delta=5e-4,
                msg=f"{shape}: the old z-sum formula's offset changed",
            )
            self.assertGreater(err, 0.010)

    def test_pre_grasp_keeps_the_pads_above_the_block(self) -> None:
        half = math.radians(171.4) / 2.0
        quat = (0.0, math.sin(half), 0.0, math.cos(half))
        pad_mid = (0.40, -0.10, 0.004 + _GRASP_HEIGHT)
        hand = hand_for_pad_midpoint(pad_mid, quat, clearance=0.09)
        self.assertGreater(
            pad_midpoint(hand, quat)[2], 0.004 + _GRASP_HEIGHT + 0.08
        )
