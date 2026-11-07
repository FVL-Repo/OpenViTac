from __future__ import annotations

import unittest
from dataclasses import asdict, replace
from typing import Any, cast

from openvitac_tactile.config import (
    FallbackCfg,
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


def pad(side: str = "left") -> PadCfg:
    return PadCfg(
        side=cast(Any, side),
        name=f"pad_{side}",
        gelpad_prim_path=f"/World/Robot/gelpad_{side}",
        frame=PadFrameCfg(
            body_name=f"panda_{side}finger",
            body_prim_path=f"/World/Robot/panda_{side}finger",
            pos=(0.0, 0.0, 0.0),
            rot_xyzw=(0.0, 0.0, 0.0, 1.0),
        ),
        geometry=PadGeometryCfg(),
        material=PadMaterialCfg(),
    )


class ConfigTests(unittest.TestCase):
    def test_defaults(self) -> None:
        cfg = TactileCfg()
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.dt_s, 1.0 / 120.0)
        self.assertEqual(cfg.sync.mode, "bounded_lag")
        self.assertEqual(cfg.fallback.on_worker_abort, "fail")
        self.assertEqual(cfg.reset.warmup_steps, 0)
        self.assertEqual(cfg.worker.python, WorkerCfg().python)
        self.assertEqual(ObjectCfg("usb", "/World/usb").authority, "uipc")
        self.assertEqual(FallbackCfg().max_seq_gap, 1)
        self.assertEqual(ResetCfg().settle_max_steps, 0)

    def test_object_usd_path_is_separate_from_mesh_path(self) -> None:
        obj = ObjectCfg(
            "usb",
            "/World/usb",
            mesh_path="usb_tet.npz",
            usd_path="usb.usd",
        )
        self.assertEqual(obj.mesh_path, "usb_tet.npz")
        self.assertEqual(obj.usd_path, "usb.usd")
        with self.assertRaises(ValueError):
            ObjectCfg("usb", "/World/usb", usd_path="")

    def test_worker_round_trip_preserves_object_usd_path(self) -> None:
        from openvitac_tactile.worker import _cfg_from_request

        cfg = TactileCfg(
            pads=(pad("left"), pad("right")),
            objects=(
                ObjectCfg(
                    "usb",
                    "/World/usb",
                    mesh_path="usb_tet.npz",
                    usd_path="usb.usd",
                ),
            ),
        )
        parsed = _cfg_from_request({"config": asdict(cfg)})
        assert parsed is not None
        self.assertEqual(parsed.objects[0].usd_path, "usb.usd")
        self.assertEqual(parsed.objects[0].mesh_path, "usb_tet.npz")

    def test_runtime_validation(self) -> None:
        cfg = TactileCfg(pads=(pad("left"), pad("right")))
        cfg.validate()
        self.assertRaises(ValueError, TactileCfg(enabled=True).validate)
        self.assertRaises(
            ValueError,
            TactileCfg,
            dt_s=0.0,
        )
        self.assertRaises(
            ValueError,
            TactileCfg,
            substeps=2,
        )
        with self.assertRaises(ValueError):
            TactileCfg(pads=(pad("left"), pad("left"))).validate()

    def test_worker_pad_frame_round_trip_preserves_back_is_min(self) -> None:
        from openvitac_tactile.worker import _cfg_from_request

        left = replace(pad("left"), frame=replace(pad("left").frame, back_is_min=True))
        right = replace(
            pad("right"), frame=replace(pad("right").frame, back_is_min=False)
        )
        cfg = TactileCfg(pads=(left, right))
        parsed = _cfg_from_request({"config": asdict(cfg)})
        assert parsed is not None
        self.assertEqual(
            [item.frame.back_is_min for item in parsed.pads],
            [True, False],
        )
        self.assertEqual(
            [asdict(item) for item in parsed.pads],
            [asdict(item) for item in cfg.pads],
        )

        raw = asdict(cfg)
        raw["pads"][0]["frame"].pop("back_is_min")
        raw["pads"][1]["frame"].pop("back_is_min")
        parsed_missing = _cfg_from_request({"config": raw})
        assert parsed_missing is not None
        self.assertEqual(
            [item.frame.back_is_min for item in parsed_missing.pads],
            [None, None],
        )

    def test_object_fem_material_validation(self) -> None:
        fem = ObjectCfg(
            "soft",
            "/World/soft",
            constitution="fem",
            youngs_mpa=2.0,
            poisson=0.45,
        )
        self.assertEqual(fem.constitution, "fem")
        self.assertEqual(fem.youngs_mpa, 2.0)
        self.assertEqual(fem.poisson, 0.45)

        with self.assertRaisesRegex(ValueError, "requires youngs_mpa"):
            ObjectCfg("soft", "/World/soft", constitution="fem", poisson=0.45)
        with self.assertRaisesRegex(ValueError, "requires poisson"):
            ObjectCfg("soft", "/World/soft", constitution="fem", youngs_mpa=2.0)
        with self.assertRaises(ValueError):
            ObjectCfg(
                "soft",
                "/World/soft",
                constitution="fem",
                youngs_mpa=0.0,
                poisson=0.45,
            )
        with self.assertRaises(ValueError):
            ObjectCfg(
                "soft",
                "/World/soft",
                constitution="fem",
                youngs_mpa=2.0,
                poisson=0.5,
            )

    def test_object_abd_rejects_fem_material(self) -> None:
        with self.assertRaisesRegex(ValueError, "only apply"):
            ObjectCfg(
                "box",
                "/World/box",
                constitution="abd",
                youngs_mpa=2.0,
                poisson=0.45,
            )
        with self.assertRaisesRegex(ValueError, "only apply"):
            ObjectCfg("box", "/World/box", constitution="abd", youngs_mpa=2.0)

    def test_object_unknown_constitution_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be 'abd' or 'fem'"):
            ObjectCfg("box", "/World/box", constitution=cast(Any, "spring"))

    def test_worker_object_fem_round_trip(self) -> None:
        from openvitac_tactile.worker import _cfg_from_request

        obj = ObjectCfg(
            "soft",
            "/World/soft",
            constitution="fem",
            youngs_mpa=2.0,
            poisson=0.45,
            density_kg_m3=900.0,
            mesh_path="/tmp/soft.npz",
        )
        cfg = TactileCfg(objects=(obj,))
        parsed = _cfg_from_request({"config": asdict(cfg)})
        assert parsed is not None
        self.assertEqual(asdict(parsed.objects[0]), asdict(obj))

        raw = asdict(cfg)
        raw["objects"][0]["constitution"] = "spring"
        with self.assertRaisesRegex(ValueError, "must be 'abd' or 'fem'"):
            _cfg_from_request({"config": raw})

    def test_sync_cap_validation(self) -> None:
        small = PadCfg(
            side="left",
            name="small",
            gelpad_prim_path="/World/gel",
            frame=PadFrameCfg("body", "/World/body", (0, 0, 0), (0, 0, 0, 1)),
            geometry=PadGeometryCfg(max_surface_vertices=64),
            material=PadMaterialCfg(),
        )
        self.assertRaises(
            ValueError,
            TactileCfg,
            pads=(small,),
            sync=SyncCfg(max_pad_vertices=128),
        )


if __name__ == "__main__":
    unittest.main()
