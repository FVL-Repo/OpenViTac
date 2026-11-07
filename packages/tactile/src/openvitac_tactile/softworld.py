"""Minimal SoftWorld manager boundary shared by worker and simulation.

The types here are pure data/interface definitions: no UIPC, SHM or Isaac
imports.  ``UipcScene`` implements the protocol in the worker; the parent-side
``TactileBridge`` exposes the same handle vocabulary over the existing SHM
``TactileState`` adapter.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .config import ObjectCfg, PadCfg
from .observation import Side
from .protocol import FingerSample


@dataclass(frozen=True, slots=True)
class PadHandle:
    """Stable handle for one registered pad."""

    uid: int
    name: str
    side: Side
    index: int


@dataclass(frozen=True, slots=True)
class ObjectHandle:
    """Stable handle for one registered object."""

    uid: int
    name: str
    index: int


@dataclass(frozen=True)
class PadState:
    """Read-only snapshot of one pad's latest state."""

    handle: PadHandle
    seq: int
    finger_seq: int
    physx_step: int
    timestamp_us: int
    n_vertices: int
    vertices: NDArray[np.float32]
    contact_gradient: NDArray[np.float64]
    force: NDArray[np.float64]
    force_kind: int
    tracking_error_mm: float
    stale: bool
    valid: bool
    usb_min_sd_mm: float
    usb_contact_vertices: int


@dataclass(frozen=True)
class ObjectState:
    """Read-only snapshot of one object's latest pose."""

    handle: ObjectHandle
    seq: int
    finger_seq: int
    physx_step: int
    timestamp_us: int
    T: NDArray[np.float64]
    pose_xyzw: NDArray[np.float64]
    valid: bool


@dataclass(frozen=True)
class PadSurface:
    """Rest surface returned by the worker for one registered pad."""

    handle: PadHandle
    rest_vertices: NDArray[np.float64]
    triangles: NDArray[np.int32]

    @property
    def name(self) -> str:
        return self.handle.name

    @property
    def side(self) -> Side:
        return self.handle.side


@runtime_checkable
class SoftWorld(Protocol):
    """Minimal manager API for the soft-pad world."""

    def register_pad(self, cfg: PadCfg) -> PadHandle: ...

    def register_object(self, cfg: ObjectCfg) -> ObjectHandle: ...

    def step(self, sample: FingerSample) -> None: ...

    def read_pad(self, handle: PadHandle) -> PadState: ...

    def read_object(self, handle: ObjectHandle) -> ObjectState: ...

    def pad_surfaces(self) -> Mapping[PadHandle, PadSurface]: ...

    def close(self) -> None: ...
