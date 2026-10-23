"""sample-seeds: pure task-randomization manifests, no simulator."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast
from unittest import mock

from openvitac_cli.commands.sample_seeds import sample_seeds
from openvitac_tasks import TASKS, EpisodeRandomization

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PACKAGES = ("cli", "data", "sim", "spec", "tactile", "tasks")
_SRC_DIRS = [str(_REPO_ROOT / "packages" / package / "src") for package in _PACKAGES]

_BOUNDARY_PROBE = """
import json
import sys

import openvitac_cli.commands.sample_seeds  # noqa: F401

banned = (
    "isaaclab",
    "isaacsim",
    "omni",
    "pxr",
    "pyuipc",
    "openvitac_sim",
    "openvitac_tactile",
)
print(json.dumps(sorted({name for name in sys.modules if name.startswith(banned)})))
"""


def _probe_env() -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        [*_SRC_DIRS, existing] if existing else _SRC_DIRS
    )
    return env


def _expected(task: str, seed: int) -> tuple[str, dict[str, float | str]]:
    """Direct ``Task.setup`` result minus expert-only parameters."""
    randomization = cast(Any, TASKS)[task].setup(seed)
    expert_keys = {f"expert.{name}" for name in randomization.expert_params}
    metadata = {
        key: value
        for key, value in randomization.to_metadata().items()
        if key not in expert_keys
    }
    return str(randomization.instruction), cast(dict[str, float | str], metadata)


class SampleSeedsTests(unittest.TestCase):
    def test_rows_are_contiguous_parseable_scalars(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "seeds.jsonl"
            sample_seeds("roughness_classify", out, start=17, count=5)
            lines = out.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(lines), 5)
        rows = [json.loads(line) for line in lines]
        self.assertEqual([row["seed"] for row in rows], [17, 18, 19, 20, 21])
        for row in rows:
            self.assertEqual(set(row), {"seed", "task", "instruction", "randomization"})
            self.assertEqual(row["task"], "roughness_classify")
            self.assertIsInstance(row["instruction"], str)
            self.assertIsInstance(row["randomization"], dict)
            for value in row["randomization"].values():
                self.assertIsInstance(value, (str, int, float))

    def test_label_and_metadata_match_direct_setup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "seeds.jsonl"
            sample_seeds("roughness_classify", out, start=0, count=8)
            rows = [
                json.loads(line)
                for line in out.read_text(encoding="utf-8").splitlines()
            ]

        for row in rows:
            instruction, metadata = _expected("roughness_classify", row["seed"])
            self.assertEqual(row["instruction"], instruction)
            self.assertEqual(row["randomization"], metadata)
            self.assertIn(row["randomization"]["roughness_label"], {"smooth", "rough"})

    def test_expert_params_are_not_written(self) -> None:
        class _Spec:
            def setup(self, seed: int) -> EpisodeRandomization:
                return EpisodeRandomization(
                    seed=seed,
                    instruction="pick",
                    metadata={"label": "a"},
                    expert_params={"gain": 0.25},
                )

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "seeds.jsonl"
            with mock.patch.dict(TASKS, {"insert_usb": _Spec()}):
                sample_seeds("insert_usb", out, count=1)
            row = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(row["randomization"], {"label": "a"})

    def test_unknown_task_raises_and_writes_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "seeds.jsonl"
            with self.assertRaises(KeyError):
                sample_seeds("not_a_task", out)
            self.assertFalse(out.exists())

    def test_negative_start_and_non_positive_count_raise(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "seeds.jsonl"
            with self.assertRaises(ValueError):
                sample_seeds("insert_usb", out, start=-1, count=1)
            with self.assertRaises(ValueError):
                sample_seeds("insert_usb", out, count=0)
            with self.assertRaises(ValueError):
                sample_seeds("insert_usb", out, count=-3)
            self.assertFalse(out.exists())

    def test_failure_leaves_no_partial_file(self) -> None:
        class _Flaky:
            calls = 0

            def setup(self, seed: int) -> EpisodeRandomization:
                type(self).calls += 1
                if type(self).calls == 3:
                    raise RuntimeError("sampler boom")
                return EpisodeRandomization(seed=seed, instruction="pick")

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "seeds.jsonl"
            out.write_text("previous\n", encoding="utf-8")
            with (
                mock.patch.dict(TASKS, {"insert_usb": _Flaky()}),
                self.assertRaises(RuntimeError),
            ):
                sample_seeds("insert_usb", out, count=4)
            self.assertEqual(out.read_text(encoding="utf-8"), "previous\n")
            self.assertEqual(
                [path.name for path in Path(directory).iterdir()], ["seeds.jsonl"]
            )

    def test_cli_subcommand_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "seeds.jsonl"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "openvitac_cli",
                    "sample-seeds",
                    "--task",
                    "insert_usb",
                    "--start",
                    "4",
                    "--count",
                    "3",
                    "--out",
                    str(out),
                ],
                cwd=_REPO_ROOT,
                env=_probe_env(),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = [
                json.loads(line)
                for line in out.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual([row["seed"] for row in rows], [4, 5, 6])
        self.assertTrue(all(row["task"] == "insert_usb" for row in rows))

    def test_command_import_loads_no_sim_or_isaaclab(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", _BOUNDARY_PROBE],
            cwd=_REPO_ROOT,
            env=_probe_env(),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [])


if __name__ == "__main__":
    unittest.main()
