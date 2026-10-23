"""Task-package import boundary and preset wiring tests (no Isaac app)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from openvitac_tasks import TASKS

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TASKS_SRC = _REPO_ROOT / "packages" / "tasks" / "src"

_BOUNDARY_PROBE = """
import json
import sys

import openvitac_tasks  # noqa: F401

banned = (
    "isaaclab",
    "pxr",
    "pyuipc",
    "openvitac_sim",
    "openvitac_tactile",
)
print(json.dumps([name for name in sys.modules if name.startswith(banned)]))
"""


class TaskImportBoundaryTests(unittest.TestCase):
    def test_registry_imports_without_sim_or_runtime(self) -> None:
        # Probe a fresh interpreter so the result cannot depend on modules
        # imported earlier by other tests in the same pytest process.
        env = dict(os.environ)
        pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(_TASKS_SRC)
            if not pythonpath
            else os.pathsep.join((str(_TASKS_SRC), pythonpath))
        )
        result = subprocess.run(
            [sys.executable, "-c", _BOUNDARY_PROBE],
            cwd=_REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        loaded = json.loads(result.stdout)
        self.assertEqual(loaded, [])

    def test_sensor_bindings_are_explicit_and_independent_of_robot(self) -> None:
        for task in TASKS.values():
            self.assertTrue(task.sensors)
            self.assertFalse(hasattr(task, "sensor"))
            for binding in task.sensors:
                self.assertFalse(hasattr(binding.sensor, "gripper_kind"))
                self.assertTrue(binding.pad_key)

    def test_robotiq_task_declares_adaptive_capability(self) -> None:
        task = TASKS["turn_gear_pair"]
        self.assertIn("adaptive_close", task.required_capabilities)
        self.assertIn(
            "adaptive_close", task.embodiment.gripper.capabilities
        )


if __name__ == "__main__":
    unittest.main()
