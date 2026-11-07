"""Shared-memory protocol: 64-byte seqlock headers + fixed slots.

The layout intentionally mirrors the verified ``iso/bridge.py`` prototype:
odd header sequence = writer active, even = stable; readers retry and validate
a checksum.  No UIPC or Isaac imports are allowed here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

type Side = Literal["left", "right"]
type ShmBuffer = bytearray | memoryview[Any] | NDArray[np.uint8]

MAX_PADS = 4
MAX_OBJECTS = 256
MAX_PAD_VERTS = 1024
HEADER_SIZE = 64
SLOTS = 2

HEADER_DTYPE = np.dtype(
    [
        ("seq", "<u8"),
        ("slot", "<u4"),
        ("done", "<u4"),
        ("frame", "<i4"),
        ("n_pad0", "<i4"),
        ("n_pad1", "<i4"),
        ("ts_ns", "<i8"),
        ("checksum", "<f8"),
        ("reserved", "u1", (20,)),
    ]
)
if HEADER_DTYPE.itemsize != HEADER_SIZE:
    raise RuntimeError(
        f"SHM header dtype is {HEADER_DTYPE.itemsize} bytes, expected {HEADER_SIZE}"
    )

FINGER_SLOT_DTYPE = np.dtype(
    [
        ("left_T", "<f8", (4, 4)),
        ("right_T", "<f8", (4, 4)),
        ("frame", "<i4"),
        ("physx_step", "<i8"),
        ("timestamp_ns", "<i8"),
        ("reserved", "u1", (8,)),
    ]
)
FINGER_SLOT_SIZE = FINGER_SLOT_DTYPE.itemsize
FINGER_SHM_SIZE = HEADER_SIZE + SLOTS * FINGER_SLOT_SIZE

FORCE_KIND_NONE = 0
FORCE_KIND_CONTACT_GRADIENT = 1
FORCE_KIND_NET_FORCE = 2

STATE_SLOT_DTYPE = np.dtype(
    [
        ("object_T", "<f8", (MAX_OBJECTS, 4, 4)),
        ("object_pose", "<f8", (MAX_OBJECTS, 7)),
        ("object_valid", "u1", (MAX_OBJECTS,)),
        ("n_objects", "<i4"),
        ("pad_vertices", "<f4", (MAX_PADS, MAX_PAD_VERTS, 3)),
        ("pad_n_vertices", "<i4", (MAX_PADS,)),
        ("pad_contact_gradient", "<f8", (MAX_PADS, 4)),
        ("pad_force", "<f8", (MAX_PADS, 3)),
        ("pad_force_kind", "u1", (MAX_PADS,)),
        ("pad_tracking_error_mm", "<f8", (MAX_PADS,)),
        ("pad_stale", "u1", (MAX_PADS,)),
        ("pad_valid", "u1", (MAX_PADS,)),
        ("pad_usb_min_sd_mm", "<f8", (MAX_PADS,)),
        ("pad_usb_contact_vertices", "<i4", (MAX_PADS,)),
        ("finger_seq", "<i8"),
        ("physx_step", "<i8"),
        ("timestamp_us", "<i8"),
    ]
)
STATE_SLOT_SIZE = STATE_SLOT_DTYPE.itemsize
STATE_SHM_SIZE = HEADER_SIZE + SLOTS * STATE_SLOT_SIZE


@dataclass(frozen=True)
class FingerSample:
    seq: int
    physx_step: int
    left_T: NDArray[np.float64]
    right_T: NDArray[np.float64]
    timestamp_ns: int = 0

    def __post_init__(self) -> None:
        if self.left_T.shape != (4, 4) or self.right_T.shape != (4, 4):
            raise ValueError("FingerSample transforms must be (4,4)")
        object.__setattr__(self, "left_T", np.asarray(self.left_T, dtype=np.float64))
        object.__setattr__(self, "right_T", np.asarray(self.right_T, dtype=np.float64))


@dataclass(frozen=True)
class TactileState:
    seq: int
    finger_seq: int
    physx_step: int
    timestamp_us: int
    object_T: NDArray[np.float64]
    object_pose: NDArray[np.float64]
    object_valid: NDArray[np.bool_]
    n_objects: int
    pad_vertices: NDArray[np.float32]
    pad_n_vertices: NDArray[np.int32]
    pad_contact_gradient: NDArray[np.float64]
    pad_force: NDArray[np.float64]
    pad_force_kind: NDArray[np.uint8]
    pad_tracking_error_mm: NDArray[np.float64]
    pad_stale: NDArray[np.bool_]
    pad_valid: NDArray[np.bool_]
    pad_usb_min_sd_mm: NDArray[np.float64] = field(
        default_factory=lambda: np.full((MAX_PADS,), np.nan, dtype=np.float64)
    )
    pad_usb_contact_vertices: NDArray[np.int32] = field(
        default_factory=lambda: np.zeros((MAX_PADS,), dtype=np.int32)
    )


def header(shm_buf: ShmBuffer) -> NDArray[Any]:
    return np.ndarray(1, dtype=HEADER_DTYPE, buffer=shm_buf, offset=0)


def finger_slot(shm_buf: ShmBuffer, index: int) -> NDArray[Any]:
    return np.ndarray(
        1,
        dtype=FINGER_SLOT_DTYPE,
        buffer=shm_buf,
        offset=HEADER_SIZE + index * FINGER_SLOT_SIZE,
    )


def state_slot(shm_buf: ShmBuffer, index: int) -> NDArray[Any]:
    return np.ndarray(
        1,
        dtype=STATE_SLOT_DTYPE,
        buffer=shm_buf,
        offset=HEADER_SIZE + index * STATE_SLOT_SIZE,
    )


def _finger_checksum(rec: NDArray[Any]) -> float:
    return float(
        np.sum(rec["left_T"], dtype=np.float64)
        + np.sum(rec["right_T"], dtype=np.float64)
    )


def _state_checksum(rec: NDArray[Any]) -> float:
    return float(
        np.nansum(rec["object_T"], dtype=np.float64)
        + np.nansum(rec["object_pose"], dtype=np.float64)
        + np.nansum(rec["pad_vertices"], dtype=np.float64)
        + np.nansum(rec["pad_contact_gradient"], dtype=np.float64)
        + np.nansum(rec["pad_force"], dtype=np.float64)
        + np.nansum(rec["pad_tracking_error_mm"], dtype=np.float64)
        + np.nansum(rec["pad_usb_min_sd_mm"], dtype=np.float64)
        + np.nansum(rec["pad_usb_contact_vertices"], dtype=np.float64)
    )


def write_fingers(
    shm_buf: ShmBuffer,
    sample: FingerSample,
    *,
    write_slot: int = 0,
) -> tuple[int, int]:
    """Publish a finger sample; return ``(header_seq, next_slot)``."""
    hdr = header(shm_buf)
    rec = finger_slot(shm_buf, write_slot)
    hdr["seq"] = 2 * sample.seq + 1
    rec["left_T"][0] = sample.left_T
    rec["right_T"][0] = sample.right_T
    rec["frame"][0] = sample.physx_step
    rec["physx_step"][0] = sample.physx_step
    rec["timestamp_ns"][0] = sample.timestamp_ns or time.perf_counter_ns()
    hdr["slot"] = write_slot
    hdr["frame"] = sample.physx_step
    hdr["checksum"] = _finger_checksum(rec[0])
    hdr["ts_ns"] = int(rec["timestamp_ns"][0])
    hdr["seq"] = 2 * sample.seq + 2
    return int(hdr["seq"][0]), 1 - write_slot


def read_fingers(shm_buf: ShmBuffer, last_seq: int = 0) -> FingerSample | None:
    hdr = header(shm_buf)
    for _ in range(8):
        seq1 = int(hdr["seq"][0])
        if seq1 == 0:
            return None
        if seq1 & 1:
            time.sleep(0.00005)
            continue
        slot_i = int(hdr["slot"][0])
        hcs = float(hdr["checksum"][0])
        rec = np.array(finger_slot(shm_buf, slot_i))
        if int(hdr["seq"][0]) != seq1:
            continue
        if abs(_finger_checksum(rec[0]) - hcs) > 1e-9 * max(1.0, abs(hcs)):
            continue
        if seq1 == last_seq:
            return None
        return FingerSample(
            seq=seq1,
            physx_step=int(rec["physx_step"][0]),
            left_T=np.array(rec["left_T"][0], dtype=np.float64),
            right_T=np.array(rec["right_T"][0], dtype=np.float64),
            timestamp_ns=int(rec["timestamp_ns"][0]),
        )
    return None


def write_state(
    shm_buf: ShmBuffer,
    state: TactileState,
    *,
    write_slot: int = 0,
) -> tuple[int, int]:
    """Publish a state sample; return ``(header_seq, next_slot)``."""
    hdr = header(shm_buf)
    rec = state_slot(shm_buf, write_slot)
    hdr["seq"] = 2 * state.seq + 1
    rec["object_T"][0] = state.object_T
    rec["object_pose"][0] = state.object_pose
    rec["object_valid"][0] = state.object_valid.astype(np.uint8)
    rec["n_objects"][0] = state.n_objects
    rec["pad_vertices"][0] = state.pad_vertices
    rec["pad_n_vertices"][0] = state.pad_n_vertices
    rec["pad_contact_gradient"][0] = state.pad_contact_gradient
    rec["pad_force"][0] = state.pad_force
    rec["pad_force_kind"][0] = state.pad_force_kind
    rec["pad_tracking_error_mm"][0] = state.pad_tracking_error_mm
    rec["pad_stale"][0] = state.pad_stale.astype(np.uint8)
    rec["pad_valid"][0] = state.pad_valid.astype(np.uint8)
    rec["pad_usb_min_sd_mm"][0] = state.pad_usb_min_sd_mm
    rec["pad_usb_contact_vertices"][0] = state.pad_usb_contact_vertices
    rec["finger_seq"][0] = state.finger_seq
    rec["physx_step"][0] = state.physx_step
    rec["timestamp_us"][0] = state.timestamp_us
    hdr["slot"] = write_slot
    hdr["checksum"] = _state_checksum(rec[0])
    hdr["ts_ns"] = time.perf_counter_ns()
    hdr["seq"] = 2 * state.seq + 2
    return int(hdr["seq"][0]), 1 - write_slot


def read_state(shm_buf: ShmBuffer, last_seq: int = 0) -> TactileState | None:
    hdr = header(shm_buf)
    for _ in range(8):
        seq1 = int(hdr["seq"][0])
        if seq1 == 0:
            return None
        if seq1 & 1:
            time.sleep(0.00005)
            continue
        slot_i = int(hdr["slot"][0])
        hcs = float(hdr["checksum"][0])
        rec = np.array(state_slot(shm_buf, slot_i))
        if int(hdr["seq"][0]) != seq1:
            continue
        if abs(_state_checksum(rec[0]) - hcs) > 1e-9 * max(1.0, abs(hcs)):
            continue
        if seq1 == last_seq:
            return None
        return TactileState(
            seq=seq1,
            finger_seq=int(rec["finger_seq"][0]),
            physx_step=int(rec["physx_step"][0]),
            timestamp_us=int(rec["timestamp_us"][0]),
            object_T=np.array(rec["object_T"][0], dtype=np.float64),
            object_pose=np.array(rec["object_pose"][0], dtype=np.float64),
            object_valid=np.array(rec["object_valid"][0], dtype=bool),
            n_objects=int(rec["n_objects"][0]),
            pad_vertices=np.array(rec["pad_vertices"][0], dtype=np.float32),
            pad_n_vertices=np.array(rec["pad_n_vertices"][0], dtype=np.int32),
            pad_contact_gradient=np.array(
                rec["pad_contact_gradient"][0], dtype=np.float64
            ),
            pad_force=np.array(rec["pad_force"][0], dtype=np.float64),
            pad_force_kind=np.array(rec["pad_force_kind"][0], dtype=np.uint8),
            pad_tracking_error_mm=np.array(
                rec["pad_tracking_error_mm"][0], dtype=np.float64
            ),
            pad_stale=np.array(rec["pad_stale"][0], dtype=bool),
            pad_valid=np.array(rec["pad_valid"][0], dtype=bool),
            pad_usb_min_sd_mm=np.array(
                rec["pad_usb_min_sd_mm"][0], dtype=np.float64
            ),
            pad_usb_contact_vertices=np.array(
                rec["pad_usb_contact_vertices"][0], dtype=np.int32
            ),
        )
    return None


def force_kind_name(code: int) -> Literal["net_force", "contact_gradient", "none"]:
    if code == FORCE_KIND_NET_FORCE:
        return "net_force"
    if code == FORCE_KIND_CONTACT_GRADIENT:
        return "contact_gradient"
    return "none"


def force_kind_code(name: str) -> int:
    if name == "net_force":
        return FORCE_KIND_NET_FORCE
    if name == "contact_gradient":
        return FORCE_KIND_CONTACT_GRADIENT
    if name == "none":
        return FORCE_KIND_NONE
    raise ValueError(f"unknown force_kind {name!r}")
