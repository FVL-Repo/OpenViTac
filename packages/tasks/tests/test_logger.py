"""Task logger API: setup injection, pure metadata, and protocol exposure."""

from __future__ import annotations

import inspect
import logging
import unittest

from openvitac_tasks import (
    TASKS,
    EpisodeRandomization,
    ExpertWorld,
    Task,
)


class TaskLoggerTests(unittest.TestCase):
    def test_setup_accepts_logger_and_logs_summary(self) -> None:
        task = TASKS["roughness_classify"]
        logger = logging.LoggerAdapter(
            logging.getLogger("openvitac.task.test"), {"task": task.name, "seed": 7}
        )
        with self.assertLogs("openvitac.task.test", level="INFO") as records:
            randomization = task.setup(7, logger=logger)
        self.assertEqual(randomization.seed, 7)
        text = "\n".join(records.output)
        self.assertIn("task setup start", text)
        self.assertIn("task setup sampled", text)
        self.assertIn("metadata", text)

    def test_setup_without_logger_still_works(self) -> None:
        task = TASKS["roughness_classify"]
        self.assertEqual(task.setup(7).seed, 7)

    def test_logger_is_not_stored_in_pure_data(self) -> None:
        self.assertNotIn("logger", Task.__dataclass_fields__)
        self.assertNotIn("logger", EpisodeRandomization.__dataclass_fields__)
        self.assertEqual(
            list(inspect.signature(EpisodeRandomization.to_metadata).parameters),
            ["self"],
        )

    def test_expert_world_exposes_logger(self) -> None:
        self.assertTrue(hasattr(ExpertWorld, "logger"))
        self.assertIsInstance(ExpertWorld.__dict__["logger"], property)


if __name__ == "__main__":
    unittest.main()
