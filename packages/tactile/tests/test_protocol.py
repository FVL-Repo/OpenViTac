from __future__ import annotations

import unittest

import numpy as np

from openvitac_tactile.protocol import (
    FINGER_SHM_SIZE,
    MAX_OBJECTS,
    MAX_PAD_VERTS,
    MAX_PADS,
    STATE_SHM_SIZE,
    FingerSample,
    TactileState,
    header,
    read_fingers,
    read_state,
    write_fingers,
    write_state,
)


def _state(seq: int = 1) -> TactileState:
    t = np.eye(4, dtype=np.float64)
    return TactileState(
        seq=seq,
        finger_seq=seq,
        physx_step=seq,
        timestamp_us=seq * 8333,
        object_T=np.stack([t] * MAX_OBJECTS),
        object_pose=np.zeros((MAX_OBJECTS, 7), dtype=np.float64),
        object_valid=np.ones((MAX_OBJECTS,), dtype=bool),
        n_objects=1,
        pad_vertices=np.zeros((MAX_PADS, MAX_PAD_VERTS, 3), dtype=np.float32),
        pad_n_vertices=np.array([3, 2, 0, 0], dtype=np.int32),
        pad_contact_gradient=np.zeros((MAX_PADS, 4), dtype=np.float64),
        pad_force=np.zeros((MAX_PADS, 3), dtype=np.float64),
        pad_force_kind=np.array([1, 1, 0, 0], dtype=np.uint8),
        pad_tracking_error_mm=np.zeros((MAX_PADS,), dtype=np.float64),
        pad_stale=np.zeros((MAX_PADS,), dtype=bool),
        pad_valid=np.array([True, True, False, False]),
    )


class ProtocolTests(unittest.TestCase):
    def test_sizes(self) -> None:
        self.assertEqual(
            header(bytearray(STATE_SHM_SIZE))["seq"].dtype, np.dtype("<u8")
        )
        self.assertGreater(STATE_SHM_SIZE, FINGER_SHM_SIZE)

    def test_finger_roundtrip_and_torn_rejection(self) -> None:
        buf = bytearray(FINGER_SHM_SIZE)
        sample = FingerSample(3, 7, np.eye(4), np.eye(4) * 2.0, timestamp_ns=123)
        hseq, _ = write_fingers(buf, sample, write_slot=0)
        self.assertEqual(hseq, 8)
        got = read_fingers(buf, 0)
        self.assertIsNotNone(got)
        assert got is not None
        self.assertEqual(got.seq, 8)
        self.assertEqual(got.physx_step, 7)
        self.assertIsNone(read_fingers(buf, hseq))
        # Tear the checksum: reader must reject, not return a half sample.
        h = header(buf)
        h["checksum"] = h["checksum"] + 1.0
        self.assertIsNone(read_fingers(buf, 0))
        # Odd sequence means writer active.
        h["seq"] = 9
        self.assertIsNone(read_fingers(buf, 0))

    def test_state_roundtrip(self) -> None:
        buf = bytearray(STATE_SHM_SIZE)
        st = _state(2)
        hseq, _ = write_state(buf, st, write_slot=1)
        self.assertEqual(hseq, 6)
        got = read_state(buf, 0)
        self.assertIsNotNone(got)
        assert got is not None
        self.assertEqual(got.object_T.shape, (MAX_OBJECTS, 4, 4))
        self.assertEqual(got.pad_vertices.shape, (MAX_PADS, MAX_PAD_VERTS, 3))
        self.assertEqual(got.pad_n_vertices.tolist(), [3, 2, 0, 0])
        self.assertEqual(got.pad_force_kind.tolist(), [1, 1, 0, 0])
        self.assertEqual(got.physx_step, 2)
        self.assertIsNone(read_state(buf, hseq))

    def test_state_torn_checksum(self) -> None:
        buf = bytearray(STATE_SHM_SIZE)
        write_state(buf, _state(1), write_slot=0)
        h = header(buf)
        h["checksum"] = h["checksum"] + 1.0
        self.assertIsNone(read_state(buf, 0))


if __name__ == "__main__":
    unittest.main()
