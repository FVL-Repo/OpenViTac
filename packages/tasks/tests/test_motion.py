"""Shared motion helpers: pad alignment, the tool-center round trip, close law."""

from __future__ import annotations

import unittest

from _tool import gsmini_tool

from openvitac_tasks.motion import (
    align_pad_to_face,
    ee_for_tool_center,
    pad_close_q,
    quat_rotate,
    tool_center,
)

_TOOL = gsmini_tool()
_PAD_NORMAL = _TOOL.pad_normal_in_ee


class AlignPadTests(unittest.TestCase):
    def test_parallel_input_is_unchanged(self) -> None:
        quat = (0.0, 0.0, 0.0, 1.0)
        self.assertEqual(align_pad_to_face(_TOOL, quat, _PAD_NORMAL), quat)

    def test_antiparallel_input_turns_half_around(self) -> None:
        # 180 deg about x maps the pad normal -y to +y, so the face normal -y
        # is exactly antiparallel and the helper must produce a 180 deg turn.
        quat = (1.0, 0.0, 0.0, 0.0)
        flipped = quat_rotate(quat, _PAD_NORMAL)
        self.assertAlmostEqual(flipped[1], 1.0)
        face = (0.0, -1.0, 0.0)
        out = align_pad_to_face(_TOOL, quat, face)
        got = quat_rotate(out, _PAD_NORMAL)
        # 1e-6 rad: the pad normal is measured off the joint frame, so it is
        # (3.4e-8, -1, 0) rather than a clean axis.
        for a, b in zip(got, face, strict=True):
            self.assertAlmostEqual(a, b, places=6)
        self.assertAlmostEqual(sum(c * c for c in out), 1.0, places=12)


class ToolCenterTests(unittest.TestCase):
    def test_ee_target_round_trips_through_the_tool_center(self) -> None:
        point = (0.38, 0.0, 0.04)
        for quat in (
            (0.0, 0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.7071067811865476, 0.7071067811865476),
        ):
            ee = ee_for_tool_center(point, quat, _TOOL)
            got = tool_center(ee, quat, _TOOL)
            for a, b in zip(got, point, strict=True):
                self.assertAlmostEqual(a, b, places=12)

    def test_offset_rotates_with_the_hand(self) -> None:
        # A half turn about x puts the hand's own axis back on +z, so the pad
        # midpoint is above the EE origin instead of below it.
        quat = (0.0, 0.0, 0.0, 1.0)
        down = tool_center((0.0, 0.0, 0.0), quat, _TOOL)[2]
        up = tool_center((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), _TOOL)[2]
        self.assertAlmostEqual(down, _TOOL.center_in_ee[2], places=12)
        self.assertAlmostEqual(up, -_TOOL.center_in_ee[2], places=12)


class CloseLawTests(unittest.TestCase):
    def test_close_law_matches_the_measured_pad_gap(self) -> None:
        # Pad-face gap = 7.1276 mm + 2 mm per mm of q; a 15.88 mm block with
        # 1 mm overtravel per side commands (15.88 - 2 - 7.1276) / 2.
        self.assertAlmostEqual(pad_close_q(_TOOL, 0.01588), 0.0033762, places=7)
        self.assertAlmostEqual(pad_close_q(_TOOL, 0.04000), 0.0154362, places=7)

    def test_law_is_the_assets_own_gap(self) -> None:
        # The command is the inverse of the asset's declared close law, not a
        # code constant: gap(q) at the commanded q is the requested gap.
        q = pad_close_q(_TOOL, 0.01500)
        self.assertAlmostEqual(_TOOL.pad_gap(q), 0.01500 - 0.002, places=12)


if __name__ == "__main__":
    unittest.main()
