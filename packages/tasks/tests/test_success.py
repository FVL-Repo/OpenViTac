"""Success generators: every registered task exposes one per-tick verdict."""

from __future__ import annotations

import inspect
import unittest
from typing import Any, cast

from openvitac_tasks import TASKS


class SuccessGeneratorTests(unittest.TestCase):
    def test_all_task_success_checks_are_generators(self) -> None:
        for name, task in TASKS.items():
            with self.subTest(task=name):
                self.assertTrue(inspect.isgeneratorfunction(task.success))

    def test_unimplemented_success_stubs_fail_on_first_read(self) -> None:
        for name in ("hardness_classify", "turn_gear_pair"):
            with (
                self.subTest(task=name),
                self.assertRaises(NotImplementedError),
            ):
                next(TASKS[name].success(cast(Any, object())))


if __name__ == "__main__":
    unittest.main()
