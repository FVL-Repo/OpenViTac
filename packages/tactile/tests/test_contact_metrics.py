"""Unit tests for the pure pad/object contact metrics."""

from __future__ import annotations

import unittest

import numpy as np

from openvitac_tactile.contact_metrics import (
    PadObjectMetrics,
    pad_object_metrics,
    select_contact_index,
)


class ContactObjectSelectionTests(unittest.TestCase):
    def test_flagged_usb_wins_when_slot_is_first(self) -> None:
        objects = [
            ("slot_static", True, False),
            ("start_slot_static", True, False),
            ("usb", False, True),
        ]
        self.assertEqual(select_contact_index(objects), 2)

    def test_unique_non_fixed_object_is_selected_without_flag(self) -> None:
        objects = [("slot_static", True, False), ("usb", False, False)]
        self.assertEqual(select_contact_index(objects), 1)

    def test_multiple_non_fixed_without_flag_is_ambiguous(self) -> None:
        with self.assertRaises(ValueError):
            select_contact_index([("a", False, False), ("b", False, False)])

    def test_multiple_flags_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            select_contact_index([("a", False, True), ("b", False, True)])

    def test_fixed_flagged_object_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            select_contact_index([("slot_static", True, True)])

    def test_no_objects_returns_none(self) -> None:
        self.assertIsNone(select_contact_index([]))


class PadObjectMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.object_T = np.eye(4, dtype=np.float64)
        self.local_min = np.array([-0.01, -0.004, -0.03])
        self.local_max = np.array([0.01, 0.004, 0.03])
        self.d_hat = 5.0e-4

    def test_idle_vertices_are_far_and_not_in_contact(self) -> None:
        points = np.array(
            [[0.0, 0.02, 0.0], [0.0, 0.021, 0.0]], dtype=np.float64
        )
        metrics = pad_object_metrics(
            points,
            self.object_T,
            self.local_min,
            self.local_max,
            self.d_hat,
        )
        self.assertIsInstance(metrics, PadObjectMetrics)
        self.assertAlmostEqual(metrics.min_sd_mm, 16.0, places=6)
        self.assertEqual(metrics.contact_vertices, 0)

    def test_close_vertices_are_counted(self) -> None:
        points = np.array(
            [
                [0.0, 0.0041, 0.0],  # 0.1 mm outside
                [0.0, 0.0039, 0.0],  # 0.1 mm inside
                [0.0, 0.0046, 0.0],  # 0.6 mm outside, beyond d_hat
            ],
            dtype=np.float64,
        )
        metrics = pad_object_metrics(
            points,
            self.object_T,
            self.local_min,
            self.local_max,
            self.d_hat,
        )
        self.assertLess(metrics.min_sd_mm, 0.0)
        self.assertEqual(metrics.contact_vertices, 2)

    def test_object_transform_is_applied(self) -> None:
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = (0.5, 0.0, 0.1)
        points = np.array([[0.5, 0.0041, 0.1]], dtype=np.float64)
        metrics = pad_object_metrics(
            points,
            T,
            self.local_min,
            self.local_max,
            self.d_hat,
        )
        self.assertAlmostEqual(metrics.min_sd_mm, 0.1, places=6)
        self.assertEqual(metrics.contact_vertices, 1)

    def test_empty_vertices_are_not_a_contact(self) -> None:
        metrics = pad_object_metrics(
            np.zeros((0, 3)),
            self.object_T,
            self.local_min,
            self.local_max,
            self.d_hat,
        )
        self.assertTrue(np.isnan(metrics.min_sd_mm))
        self.assertEqual(metrics.contact_vertices, 0)


if __name__ == "__main__":
    unittest.main()
