"""Body-name lookup order tests for the tactile attachment points."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from openvitac_sim.scene import Scene
from openvitac_tactile.config import (
    PadCfg,
    PadFrameCfg,
    PadGeometryCfg,
    PadMaterialCfg,
)


class _TensorView:
    """Minimal ``body_pose_w.torch[0, ids].cpu().numpy()`` stand-in."""

    def __init__(self, rows: np.ndarray[Any, np.dtype[np.float64]]) -> None:
        self._rows = rows

    @property
    def torch(self) -> _TensorView:
        return self

    def __getitem__(self, key: tuple[int, list[int]]) -> _TensorView:
        _, ids = key
        return _TensorView(self._rows[np.asarray(ids, dtype=np.int64)])

    def cpu(self) -> _TensorView:
        return self

    def numpy(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        return self._rows


class _FakeRobot:
    def __init__(self, body_order: tuple[str, ...], rows: np.ndarray[Any, Any]) -> None:
        self._body_order = list(body_order)
        self._rows = np.asarray(rows, dtype=np.float64)
        self.calls: list[tuple[list[str], bool]] = []
        self.data = SimpleNamespace(body_pose_w=_TensorView(self._rows))

    def find_bodies(
        self, name_keys: str | list[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        requested = [name_keys] if isinstance(name_keys, str) else list(name_keys)
        self.calls.append((requested, preserve_order))
        if preserve_order:
            matched = [name for name in requested if name in self._body_order]
        else:
            matched = [name for name in self._body_order if name in requested]
        return [self._body_order.index(name) for name in matched], matched


class _MismatchedRobot(_FakeRobot):
    def find_bodies(
        self, name_keys: str | list[str], preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        return [0, 1], ["right", "left"]


def _pad(side: str, body_name: str) -> PadCfg:
    return PadCfg(
        side=cast(Any, side),
        name=f"pad_{side}",
        gelpad_prim_path=f"/World/Robot/gelpad_{side}",
        frame=PadFrameCfg(
            body_name=body_name,
            body_prim_path=f"/World/Robot/{body_name}",
            pos=(0.0, 0.0, 0.0),
            rot_xyzw=(0.0, 0.0, 0.0, 1.0),
        ),
        geometry=PadGeometryCfg(),
        material=PadMaterialCfg(),
    )


def _scene(robot: _FakeRobot) -> Scene:
    scene = Scene.__new__(Scene)
    scene._tactile_bridge = cast(
        Any,
        SimpleNamespace(
            cfg=SimpleNamespace(
                pads=(_pad("left", "left"), _pad("right", "right")),
            )
        ),
    )
    scene._arm = cast(
        Any,
        SimpleNamespace(
            robot=robot,
            embodiment=SimpleNamespace(
                gripper=SimpleNamespace(contact_bodies=("left", "right"))
            ),
        ),
    )
    scene._steps_taken = cast(Any, 0)
    scene._physics_steps_total = cast(Any, 0)
    scene._body_index_cache = cast(Any, {})
    return scene


class BodyOrderTests(unittest.TestCase):
    def test_names_are_resolved_in_request_order(self) -> None:
        rows = np.array(
            [
                [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # right in articulation
                [0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],  # left in articulation
            ],
            dtype=np.float64,
        )
        robot = _FakeRobot(("right", "left"), rows)
        scene = _scene(robot)

        ids = scene._body_ids(["left", "right"])
        self.assertEqual(ids, [1, 0])
        self.assertEqual(robot.calls, [(["left", "right"], True)])
        self.assertEqual(scene._body_ids(["left", "right"]), [1, 0])
        self.assertEqual(len(robot.calls), 1)

        frames = scene._tactile_attachment_frames()
        np.testing.assert_allclose(frames["pad_left"][:3, 3], [0.02, 0.0, 0.0])
        np.testing.assert_allclose(frames["pad_right"][:3, 3], [0.01, 0.0, 0.0])

        sample = scene._make_finger_sample()
        np.testing.assert_allclose(sample.left_T[:3, 3], [0.02, 0.0, 0.0])
        np.testing.assert_allclose(sample.right_T[:3, 3], [0.01, 0.0, 0.0])

    def test_missing_or_reordered_names_fail_loud(self) -> None:
        rows = np.zeros((2, 7), dtype=np.float64)
        rows[:, 6] = 1.0
        with self.assertRaises(ValueError):
            _scene(_FakeRobot(("left", "right"), rows))._body_ids(["left", "left"])
        with self.assertRaises(RuntimeError):
            _scene(_MismatchedRobot(("left", "right"), rows))._body_ids(
                ["left", "right"]
            )


if __name__ == "__main__":
    unittest.main()
