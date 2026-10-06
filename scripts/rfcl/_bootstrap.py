"""Repository path helpers shared by RFCL command-line tools."""

from __future__ import annotations

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def add_repository_root() -> Path:
    value = str(REPOSITORY_ROOT)
    if value not in sys.path:
        sys.path.insert(0, value)
    return REPOSITORY_ROOT
