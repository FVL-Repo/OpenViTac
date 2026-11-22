"""The tool contract the fake worlds grip with, read from the shipped asset.

``read_tool_geometry`` imports ``pxr`` lazily, so importing this module costs
nothing; the callers run in the sim venv, which has ``pxr``.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

from openvitac_spec import ToolGeometry, read_tool_geometry

ASSET = (
    Path(__file__).resolve().parents[3]
    / "assets/embodiment/franka_gsmini.usda"
)


@cache
def gsmini_tool() -> ToolGeometry:
    """Panda parallel fingers + GsMini gel pads: the asset's own numbers."""
    return read_tool_geometry(ASSET, ee_link="panda_hand")
