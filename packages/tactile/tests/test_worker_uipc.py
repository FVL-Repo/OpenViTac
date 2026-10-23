from __future__ import annotations

import os
import time
import unittest
from typing import Any, cast

import numpy as np

from openvitac_tactile.client import TactileRuntime
from openvitac_tactile.config import (
    ObjectCfg,
    PadCfg,
    PadFrameCfg,
    PadGeometryCfg,
    PadMaterialCfg,
    ResetCfg,
    SyncCfg,
    TactileCfg,
    WorkerCfg,
)
from openvitac_tactile.protocol import FingerSample

# Real UIPC needs Isaac/Kit's runtime libraries on the loader path; the bare
# default-dependency pyuipc wheel alone cannot resolve its native symbols. The
# unit suite therefore stays Isaac-free and this test is an explicit integration
# run inside the Kit environment.
RUN_UIPC_INTEGRATION = os.environ.get("OPENVITAC_RUN_UIPC_INTEGRATION") == "1"
UIPC_INTEGRATION_SKIP = (
    "UIPC integration test: requires the Isaac/Kit runtime plus pyuipc; "
    "set OPENVITAC_RUN_UIPC_INTEGRATION=1 to run it there"
)


def _pad(side: str, y: float) -> PadCfg:
    return PadCfg(
        side=cast(Any, side),
        name=f"pad_{side}",
        gelpad_prim_path=f"/World/Robot/gelpad_{side}",
        frame=PadFrameCfg(
            body_name=f"panda_{side}finger",
            body_prim_path=f"/World/Robot/panda_{side}finger",
            pos=(0.0, y, 0.05),
            rot_xyzw=(0.0, 0.0, 0.0, 1.0),
        ),
        geometry=PadGeometryCfg(
            kind="box",
            size_m=(0.03, 0.004, 0.02),
            tet_resolution=(4, 1, 4),
            max_surface_vertices=1024,
            surface_decimation=1,
        ),
        material=PadMaterialCfg(
            youngs_mpa=0.1,
            poisson=0.49,
            density_kg_m3=1.0e4,
            constraint_strength=1.0e4,
            contact_d_hat_m=5.0e-4,
            friction_rate=2.5,
            resistance_pa=1.0e9,
        ),
    )


@unittest.skipUnless(RUN_UIPC_INTEGRATION, UIPC_INTEGRATION_SKIP)
class UipcWorkerSmokeTests(unittest.TestCase):
    def test_softworld_handles(self) -> None:
        import tempfile

        from openvitac_tactile.softworld import ObjectHandle, PadHandle
        from openvitac_tactile.uipc_scene import build_scene

        cfg = TactileCfg(
            pads=(_pad("left", -0.012), _pad("right", 0.012)),
            objects=(
                ObjectCfg(
                    name="box",
                    prim_path="/World/box",
                    authority="uipc",
                    constitution="abd",
                    kappa_mpa=100.0,
                    density_kg_m3=1.0e3,
                ),
            ),
            dt_s=1.0 / 120.0,
            sync=SyncCfg(
                mode="lockstep", state_timeout_ms=5000.0, max_pad_vertices=1024
            ),
            worker=WorkerCfg(workspace="/tmp/openvitac_uipc_test"),
            reset=ResetCfg(),
        )
        with tempfile.TemporaryDirectory() as workspace:
            scene = build_scene(cfg, workspace=workspace)
            try:
                scene.initial_state()
                pad_handles = scene.pad_handles
                object_handles = scene.object_handles
                self.assertEqual(len(pad_handles), 2)
                self.assertEqual(len(object_handles), 1)
                left = scene.read_pad(pad_handles[0])
                self.assertGreater(left.n_vertices, 0)
                self.assertTrue(left.valid)
                self.assertTrue(np.all(np.isfinite(left.vertices)))
                obj = scene.read_object(object_handles[0])
                self.assertTrue(obj.valid)
                self.assertTrue(np.all(np.isfinite(obj.T)))
                identity = np.eye(4, dtype=np.float64)
                scene.step(
                    FingerSample(
                        seq=1,
                        physx_step=1,
                        left_T=identity,
                        right_T=identity,
                        timestamp_ns=time.time_ns(),
                    )
                )
                after = scene.read_pad(pad_handles[0])
                legacy = scene.legacy_state()
                self.assertEqual(after.n_vertices, int(legacy.pad_n_vertices[0]))
                np.testing.assert_allclose(
                    after.vertices, legacy.pad_vertices[0, : after.n_vertices]
                )
                after_obj = scene.read_object(object_handles[0])
                self.assertTrue(after_obj.valid)
                with self.assertRaises(ValueError):
                    scene.read_pad(
                        PadHandle(uid=99, name="missing", side="left", index=0)
                    )
                with self.assertRaises(ValueError):
                    scene.read_object(ObjectHandle(uid=99, name="missing", index=0))
            finally:
                scene.close()

    def test_dual_pad_60_steps(self) -> None:
        cfg = TactileCfg(
            pads=(_pad("left", -0.012), _pad("right", 0.012)),
            objects=(
                ObjectCfg(
                    name="box",
                    prim_path="/World/box",
                    authority="uipc",
                    constitution="abd",
                    kappa_mpa=100.0,
                    density_kg_m3=1.0e3,
                ),
            ),
            dt_s=1.0 / 120.0,
            sync=SyncCfg(
                mode="lockstep", state_timeout_ms=5000.0, max_pad_vertices=1024
            ),
            worker=WorkerCfg(workspace="/tmp/openvitac_uipc_test"),
            reset=ResetCfg(),
        )
        rt = TactileRuntime(cfg, run_id=f"uipc_smoke_{time.time_ns()}", check_uipc=True)
        rt.start()
        try:
            ack = rt.reset({})
            self.assertEqual(ack["status"], "ready")
            self.assertEqual(ack["n_pads"], 2)
            self.assertEqual(ack["n_objects"], 1)
            surfaces = {entry["side"]: entry for entry in ack["pads"]}
            self.assertEqual(set(surfaces), {"left", "right"})
            for entry in surfaces.values():
                rest = np.asarray(entry["rest_vertices"], dtype=np.float64)
                tris = np.asarray(entry["triangles"], dtype=np.int32)
                self.assertEqual(rest.ndim, 2)
                self.assertEqual(rest.shape[1], 3)
                self.assertGreater(len(rest), 0)
                self.assertEqual(tris.ndim, 2)
                self.assertEqual(tris.shape[1], 3)
                self.assertGreaterEqual(int(tris.min()), 0)
                self.assertLess(int(tris.max()), len(rest))
                self.assertTrue(np.all(np.isfinite(rest)))
            identity = np.eye(4, dtype=np.float64)
            for step in range(1, 61):
                state = rt.exchange(
                    FingerSample(
                        seq=step,
                        physx_step=step,
                        left_T=identity,
                        right_T=identity,
                        timestamp_ns=time.time_ns(),
                    ),
                    wait=True,
                )
                self.assertIsNotNone(state)
                assert state is not None
                self.assertGreater(int(state.pad_n_vertices[0]), 0)
                self.assertGreater(int(state.pad_n_vertices[1]), 0)
                self.assertTrue(np.all(np.isfinite(state.pad_vertices[:2])))
                self.assertTrue(np.all(np.isfinite(state.pad_contact_gradient[:2])))
                self.assertTrue(np.all(np.isfinite(state.pad_usb_min_sd_mm[:2])))
                self.assertTrue(np.all(state.pad_usb_contact_vertices[:2] >= 0))
                self.assertTrue(bool(state.pad_valid[0]))
                self.assertTrue(bool(state.pad_valid[1]))
                self.assertEqual(
                    int(state.pad_n_vertices[0]),
                    len(surfaces["left"]["rest_vertices"]),
                )
                self.assertEqual(
                    int(state.pad_n_vertices[1]),
                    len(surfaces["right"]["rest_vertices"]),
                )
                self.assertTrue(bool(state.object_valid[0]))
                self.assertTrue(np.all(np.isfinite(state.object_T[0])))
        finally:
            rt.close()


if __name__ == "__main__":
    unittest.main()
