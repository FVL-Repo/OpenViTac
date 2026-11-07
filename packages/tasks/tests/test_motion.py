"""Shared motion helpers: pad alignment and the close-law pin."""

from __future__ import annotations

import unittest

from openvitac_tasks.motion import (
    PAD_NORMAL_IN_HAND,
    align_pad_to_face,
    pad_close_q,
    quat_rotate,
)


class AlignPadTests(unittest.TestCase):
    def test_parallel_input_is_unchanged(self) -> None:
        quat = (0.0, 0.0, 0.0, 1.0)
        self.assertEqual(align_pad_to_face(quat, PAD_NORMAL_IN_HAND), quat)

    def test_antiparallel_input_turns_half_around(self) -> None:
        # 180 deg about x maps the pad normal -y to +y, so the face normal -y
        # is exactly antiparallel and the helper must produce a 180 deg turn.
        quat = (1.0, 0.0, 0.0, 0.0)
        flipped = quat_rotate(quat, PAD_NORMAL_IN_HAND)
        self.assertAlmostEqual(flipped[1], 1.0)
        out = align_pad_to_face(quat, (0.0, -1.0, 0.0))
        got = quat_rotate(out, PAD_NORMAL_IN_HAND)
        self.assertAlmostEqual(got[0], 0.0, places=12)
        self.assertAlmostEqual(got[1], -1.0, places=12)
        self.assertAlmostEqual(got[2], 0.0, places=12)
        self.assertAlmostEqual(sum(c * c for c in out), 1.0, places=12)

    def test_close_law_matches_the_measured_pad_gap(self) -> None:
        # Pad-face gap = 7.1276 mm + 2 mm per mm of q; a 15.88 mm block with
        # 1 mm overtravel per side commands (15.88 - 2 - 7.1276) / 2.
        self.assertAlmostEqual(pad_close_q(0.01588), 0.0033762, places=7)
        self.assertAlmostEqual(pad_close_q(0.04000), 0.0154362, places=7)


if __name__ == "__main__":
    unittest.main()
