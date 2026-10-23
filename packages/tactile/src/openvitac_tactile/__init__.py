"""OpenViTac tactile: config, protocol, observations, and process client."""

from __future__ import annotations

from .client import (
    TactileHealth,
    TactileRuntime,
    TactileSequenceGap,
    TactileTimeout,
    TactileUnavailable,
)
from .config import (
    FallbackCfg,
    ObjectCfg,
    PadCfg,
    PadFrameCfg,
    PadGeometryCfg,
    PadMaterialCfg,
    RecordCfg,
    ResetCfg,
    SyncCfg,
    TactileCfg,
    WorkerCfg,
)
from .observation import (
    ObjectObservation,
    PadKey,
    PadObservation,
    TactileObservation,
)
from .protocol import (
    MAX_OBJECTS,
    MAX_PAD_VERTS,
    MAX_PADS,
    FingerSample,
    TactileState,
)

__all__ = [
    "MAX_OBJECTS",
    "MAX_PADS",
    "MAX_PAD_VERTS",
    "FallbackCfg",
    "FingerSample",
    "ObjectCfg",
    "ObjectObservation",
    "PadCfg",
    "PadFrameCfg",
    "PadGeometryCfg",
    "PadKey",
    "PadMaterialCfg",
    "PadObservation",
    "RecordCfg",
    "ResetCfg",
    "SyncCfg",
    "TactileCfg",
    "TactileHealth",
    "TactileObservation",
    "TactileRuntime",
    "TactileSequenceGap",
    "TactileState",
    "TactileTimeout",
    "TactileUnavailable",
    "WorkerCfg",
]
