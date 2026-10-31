"""PEP 562 lazy attribute access for isaaclab-dependent ``openvitac_sim`` modules.

``openvitac_sim/__init__.py`` re-exports ``__getattr__`` from here so the package
can keep a re-export-only namespace while still resolving ``spawn``/``robot``/
``scene`` names only after the app is up.
"""

from __future__ import annotations

import importlib
import sys

# submodule -> names it exports, imported on first access (post-launch).
_LAZY: dict[str, tuple[str, ...]] = {
    "spawn": (
        "create_joint",
        "spawn_articulation",
        "spawn_camera",
        "spawn_light",
        "spawn_rigid",
        "spawn_static",
    ),
    "robot": ("RobotArm", "assemble_robot"),
    "scene": ("Scene", "SceneObservation", "StepLimitExceeded"),
}

_PACKAGE = __name__.rpartition(".")[0]


def __getattr__(name: str) -> object:
    for mod_name, names in _LAZY.items():
        if name in names:
            module = importlib.import_module(f".{mod_name}", _PACKAGE)
            value = getattr(module, name)
            # Cache on the package so __getattr__ fires once per name.
            setattr(sys.modules[_PACKAGE], name, value)
            return value
    raise AttributeError(f"module {_PACKAGE!r} has no attribute {name!r}")
