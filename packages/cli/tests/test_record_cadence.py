"""Default 60 Hz recording: physics stays at 120 Hz, frames land every N steps."""

from __future__ import annotations

import unittest

import numpy as np

from openvitac_cli.commands.collect import (
    _DEFAULT_RECORD_EVERY,
    _SIM_DT_US,
    collect,
)
from openvitac_sim.scene import SceneObservation, recorded_step_indices


class RecordCadenceTests(unittest.TestCase):
    def test_default_records_three_frames_for_five_steps(self) -> None:
        self.assertEqual(_DEFAULT_RECORD_EVERY, 2)
        # 0-based physics steps 0/2/4, i.e. the 1st/3rd/5th stepped frames.
        self.assertEqual(recorded_step_indices(5, 2), (0, 2, 4))

    def test_every_one_records_every_physics_step(self) -> None:
        self.assertEqual(recorded_step_indices(5, 1), (0, 1, 2, 3, 4))

    def test_every_four_records_two_frames_for_five_steps(self) -> None:
        self.assertEqual(recorded_step_indices(5, 4), (0, 4))

    def test_timestamps_follow_physics_step_not_frame_number(self) -> None:
        steps = recorded_step_indices(5, _DEFAULT_RECORD_EVERY)
        timestamps = [step * _SIM_DT_US for step in steps]
        self.assertEqual(_SIM_DT_US, 8333)
        self.assertEqual(timestamps, [0, 16666, 33332])

    def test_non_positive_cadence_is_rejected(self) -> None:
        for every in (0, -1):
            with self.assertRaises(ValueError):
                recorded_step_indices(5, every)
        with self.assertRaises(ValueError):
            collect("insert_usb", record_every=0)

    def test_observation_carries_physics_step(self) -> None:
        obs = SceneObservation(
            rgb={},
            joint_pos=np.zeros(1),
            joint_vel=np.zeros(1),
            action=np.zeros(1),
        )
        self.assertEqual(obs.physics_step, 0)


if __name__ == "__main__":
    unittest.main()
