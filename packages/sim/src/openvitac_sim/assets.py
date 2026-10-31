"""Locate the repo ``assets/`` root by content, not by cwd or install layout."""

from __future__ import annotations

import os
from pathlib import Path

_ENV_VAR = "OPENVITAC_ASSETS"
# plate.usda sits at the assets root; its presence proves a candidate dir is the root.
_SENTINEL = "plate.usda"


def assets_root() -> Path:
    """Absolute path of the repo ``assets/`` directory.

    ``OPENVITAC_ASSETS`` wins when set; otherwise walk up from this file, which
    resolves correctly from a source checkout regardless of the process cwd.
    """
    override = os.environ.get(_ENV_VAR)
    if override is not None:
        root = Path(override).expanduser()
        if not (root / _SENTINEL).is_file():
            raise FileNotFoundError(f"{_ENV_VAR}={override!r} lacks {_SENTINEL}")
        return root
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "assets"
        if (candidate / _SENTINEL).is_file():
            return candidate
    raise FileNotFoundError(f"no assets/{_SENTINEL} above {__file__}; set {_ENV_VAR}")


def asset(relative: str) -> str:
    """Absolute path of the asset at ``relative`` under the assets root."""
    return str(assets_root() / relative)
