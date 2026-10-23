from __future__ import annotations

import sys
import unittest
from unittest import mock

from openvitac_tactile.client import TactileRuntime, TactileUnavailable
from openvitac_tactile.config import (
    PadCfg,
    PadFrameCfg,
    PadGeometryCfg,
    PadMaterialCfg,
    TactileCfg,
)


def _pad() -> PadCfg:
    return PadCfg(
        side="left",
        name="left",
        gelpad_prim_path="/World/Robot/gelpad_left",
        frame=PadFrameCfg(
            body_name="panda_leftfinger",
            body_prim_path="/World/Robot/panda_leftfinger",
            pos=(0.0, 0.0, 0.0),
            rot_xyzw=(0.0, 0.0, 0.0, 1.0),
        ),
        geometry=PadGeometryCfg(max_surface_vertices=1024),
        material=PadMaterialCfg(),
    )


class ClientTests(unittest.TestCase):
    def test_import_does_not_import_pyuipc(self) -> None:
        import openvitac_tactile as tactile

        self.assertIsNotNone(tactile)
        self.assertNotIn("uipc", sys.modules)

    def test_skeleton_worker_lifecycle(self) -> None:
        cfg = TactileCfg(pads=(_pad(),), dt_s=1.0 / 120.0)
        rt = TactileRuntime(cfg, run_id="test_lifecycle", check_uipc=False)
        rt.start()
        try:
            self.assertTrue(rt.health.alive)
            ack = rt.reset({"dt_s": cfg.dt_s})
            self.assertEqual(ack["status"], "ready")
            self.assertEqual(rt.health.last_seq, 0)
            self.assertIsNone(rt.latest())
        finally:
            rt.close()
        self.assertFalse(rt.health.alive)
        rt.close()

    def test_invalid_serialized_config_fails_loud(self) -> None:
        cfg = TactileCfg(pads=(_pad(),), dt_s=1.0 / 120.0)
        rt = TactileRuntime(cfg, run_id="test_bad_config", check_uipc=False)
        rt.start()
        try:
            with self.assertRaises(TactileUnavailable):
                rt.reset({"config": {"enabled": True, "pads": [{"name": "bad"}]}})
        finally:
            rt.close()

    def test_missing_pyuipc_fails_loud(self) -> None:
        cfg = TactileCfg(pads=(_pad(),))
        rt = TactileRuntime(cfg, run_id="test_missing_uipc", check_uipc=True)
        with (
            mock.patch(
                "openvitac_tactile.client.importlib.util.find_spec", return_value=None
            ),
            self.assertRaises(TactileUnavailable),
        ):
            rt.start()

    def test_failed_start_cleans_up_and_allows_retry(self) -> None:
        cfg = TactileCfg(pads=(_pad(),), dt_s=1.0 / 120.0)
        rt = TactileRuntime(
            cfg,
            run_id="test_start_retry",
            check_uipc=False,
            worker_module="openvitac_tactile.missing_worker",
        )
        with self.assertRaises(TactileUnavailable):
            rt.start()
        self.assertIsNone(rt._proc)
        self.assertIsNone(rt._state_shm)
        self.assertIsNone(rt._finger_shm)
        # The same names must be creatable again; otherwise the failed start
        # leaked a named SHM segment.
        rt._worker_module = "openvitac_tactile.worker"
        rt.start()
        try:
            self.assertTrue(rt.health.alive)
        finally:
            rt.close()

    def test_reset_requires_start(self) -> None:
        cfg = TactileCfg(pads=(_pad(),))
        rt = TactileRuntime(cfg, run_id="test_not_started", check_uipc=False)
        with self.assertRaises(TactileUnavailable):
            rt.reset({})


if __name__ == "__main__":
    unittest.main()
