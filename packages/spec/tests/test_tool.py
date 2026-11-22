"""The tool contract reader: the shipped asset, and assets that must fail loud."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

from openvitac_spec import ToolGeometry, read_tool_geometry

_GSMINI = Path(__file__).resolve().parents[3] / "assets/embodiment/franka_gsmini.usda"


class ShippedAssetTests(unittest.TestCase):
    """The gsmini asset's declared numbers, pinned to the measurement they came from."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tool = read_tool_geometry(_GSMINI, ee_link="panda_hand")

    def test_pad_midpoint_is_measured_in_the_hand_frame(self) -> None:
        # Midpoint of the two gel-pad contact faces, both fingers at q=0.
        for got, want in zip(
            self.tool.center_in_ee, (-0.0000816, -0.0002294, 0.1181816), strict=True
        ):
            self.assertAlmostEqual(got, want, places=6)

    def test_close_law_is_the_asset_gap_and_travel(self) -> None:
        self.assertAlmostEqual(self.tool.pad_gap_closed, 0.007127577, places=9)
        # The joint limit is a float32 in the USD, so the span carries ~1e-9.
        self.assertAlmostEqual(self.tool.open_travel, 0.04, places=8)
        self.assertEqual(self.tool.finger_count, 2)
        self.assertEqual(self.tool.finger_q, (0.0, 0.03999999910593033))
        self.assertFalse(self.tool.closed_at_upper)
        self.assertEqual(self.tool.q_closed, 0.0)
        self.assertAlmostEqual(
            self.tool.pad_gap(self.tool.q_closed), 0.007127577, places=9
        )
        self.assertAlmostEqual(
            self.tool.pad_gap(self.tool.q_open), 0.007127577 + 0.08, places=8
        )

    def test_pad_normal_is_the_closing_axis(self) -> None:
        # The second finger's joint X axis in the hand frame: the jaws move on
        # +/-Y, so the pads face each other along Y.
        for got, want in zip(
            self.tool.pad_normal_in_ee, (0.0, -1.0, 0.0), strict=True
        ):
            self.assertAlmostEqual(got, want, places=6)

    def test_pad_extent_is_the_gel_pad_body(self) -> None:
        for got, want in zip(
            self.tool.pad_extent, (0.02075, 0.004, 0.02525), strict=True
        ):
            self.assertAlmostEqual(got, want, places=6)

    def test_command_for_a_gap_inverts_the_law(self) -> None:
        for gap in (0.007127577, 0.02, 0.087):
            q = self.tool.finger_command_for_gap(gap)
            self.assertAlmostEqual(self.tool.pad_gap(q), gap, places=8)

    def test_command_is_clamped_to_the_joint_limits(self) -> None:
        self.assertEqual(self.tool.finger_command_for_gap(-1.0), self.tool.q_closed)
        self.assertEqual(self.tool.finger_command_for_gap(1.0), self.tool.q_open)


class GeometryValidationTests(unittest.TestCase):
    def _tool(self, **overrides: object) -> ToolGeometry:
        fields: dict[str, object] = {
            "ee_link": "panda_hand",
            "center_in_ee": (0.0, 0.0, 0.1),
            "pad_normal_in_ee": (0.0, -1.0, 0.0),
            "pad_extent": (0.02, 0.004, 0.02),
            "pad_gap_closed": 0.0,
            "open_travel": 0.04,
            "finger_count": 2,
            "finger_q": (0.0, 0.04),
        }
        fields.update(overrides)
        return ToolGeometry(**fields)  # type: ignore[arg-type]

    def test_upper_limit_can_be_the_closed_end(self) -> None:
        tool = self._tool(closed_at_upper=True)
        self.assertEqual(tool.q_closed, 0.04)
        self.assertEqual(tool.q_open, 0.0)
        self.assertAlmostEqual(tool.pad_gap(0.04), 0.0, places=12)
        self.assertAlmostEqual(tool.pad_gap(0.0), 0.08, places=12)

    def test_rejects_inconsistent_fields(self) -> None:
        for bad in (
            {"ee_link": ""},
            {"pad_extent": (0.02, 0.004)},
            {"pad_gap_closed": -0.001},
            {"open_travel": -0.001},
            {"finger_count": 0},
            {"finger_q": (0.04, 0.04)},
            {"pad_normal_in_ee": (0.0, -1.0, 0.0) and (0.0, -2.0, 0.0)},
        ):
            with self.assertRaises(ValueError, msg=str(bad)):
                self._tool(**bad)


_JOINT = """
    def {klass} "{name}" (
        prepend apiSchemas = ["PhysicsDriveAPI:angular"]
    )
    {{
        rel physics:body0 = <{body0}>
        uniform token physics:axis = "{axis}"
        quatf physics:localRot0 = {rot0}
        float physics:lowerLimit = {lower}
        float physics:upperLimit = {upper}
    }}
"""


def _write_asset(
    path: Path,
    *,
    joints: tuple[tuple[str, float, float], ...] = (
        ("finger_left", 0.0, 0.04),
        ("finger_right", 0.0, 0.04),
    ),
    revolute: bool = False,
    travel: float = 0.04,
    close_gap: float = 0.0,
    close_gap_at: str | None = None,
    contract: bool = True,
    tool_translate: tuple[float, float, float] = (0.0, 0.0, 0.1),
) -> None:
    """Write a minimal robot asset: a hand, two finger joints, two pads, a tool."""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/robot")
    stage.SetDefaultPrim(root.GetPrim())
    hand = UsdGeom.Xform.Define(stage, "/robot/hand")
    tool = UsdGeom.Xform.Define(stage, "/robot/hand/tool_center")
    UsdGeom.Xformable(tool).AddTranslateOp().Set(tool_translate)
    for side, y in (("left", 0.005), ("right", -0.005)):
        pad = UsdGeom.Xform.Define(stage, f"/robot/pad_{side}")
        UsdGeom.Xformable(pad).AddTranslateOp().Set((0.0, y, 0.1))
    for name, lower, upper in joints:
        klass = "RevoluteJoint" if revolute else "PrismaticJoint"
        joint = getattr(UsdPhysics, klass).Define(stage, f"/robot/hand/{name}")
        joint.CreateBody0Rel().SetTargets([hand.GetPrim().GetPath()])
        joint.CreateAxisAttr().Set("X")
        # 90 deg about z maps the joint's X onto the hand's Y, as in the shipped
        # panda asset, so the mirrored jaw closes on -Y.
        from pxr import Gf

        joint.CreateLocalRot0Attr().Set(
            Gf.Quatf(0.70710678, Gf.Vec3f(0.0, 0.0, 0.70710678))
            if name.endswith("left")
            else Gf.Quatf(0.70710678, Gf.Vec3f(0.0, 0.0, -0.70710678))
        )
        joint.CreateLowerLimitAttr().Set(lower)
        joint.CreateUpperLimitAttr().Set(upper)
    if contract:
        from pxr import Vt

        data: dict[str, object] = {
            "contract_version": "1",
            "tool_center": "/robot/hand/tool_center",
            "pad_frames": Vt.StringArray(["/robot/pad_left", "/robot/pad_right"]),
            "pad_extent_m": Vt.FloatArray([0.02, 0.004, 0.02]),
            "finger_joints": Vt.StringArray([name for name, _lo, _hi in joints]),
            "open_travel_m": travel,
            "close_gap_m": close_gap,
        }
        if close_gap_at is not None:
            data["close_gap_at"] = close_gap_at
        stage.GetRootLayer().customLayerData = {"openvitac": data}
    stage.Save()


class SyntheticAssetTests(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "robot.usda"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_reads_a_sliding_jaw(self) -> None:
        _write_asset(
            self.path, travel=0.04, close_gap=0.008, close_gap_at="lower"
        )
        tool = read_tool_geometry(self.path, ee_link="hand")
        self.assertAlmostEqual(tool.center_in_ee[2], 0.1, places=9)
        self.assertAlmostEqual(tool.pad_gap_closed, 0.008, places=9)
        self.assertEqual(tool.finger_q[0], 0.0)
        self.assertAlmostEqual(tool.finger_q[1], 0.04, places=8)
        self.assertFalse(tool.closed_at_upper)

    def test_missing_asset_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            read_tool_geometry(self.path, ee_link="hand")

    def test_missing_contract_raises(self) -> None:
        _write_asset(self.path, contract=False)
        with self.assertRaises(ValueError):
            read_tool_geometry(self.path, ee_link="hand")

    def test_incomplete_contract_names_the_key(self) -> None:
        _write_asset(self.path)
        from pxr import Usd

        stage = Usd.Stage.Open(str(self.path))
        data = dict(stage.GetRootLayer().customLayerData["openvitac"])
        del data["close_gap_m"]
        stage.GetRootLayer().customLayerData = {"openvitac": data}
        stage.Save()
        with self.assertRaisesRegex(ValueError, "close_gap_m"):
            read_tool_geometry(self.path, ee_link="hand")

    def test_travel_must_equal_the_joint_span_for_a_sliding_jaw(self) -> None:
        _write_asset(self.path, travel=0.03)
        with self.assertRaisesRegex(ValueError, "open_travel_m"):
            read_tool_geometry(self.path, ee_link="hand")

    def test_finger_joints_must_share_one_span(self) -> None:
        _write_asset(
            self.path,
            joints=(("finger_left", 0.0, 0.04), ("finger_right", 0.0, 0.05)),
        )
        with self.assertRaisesRegex(ValueError, "different travel"):
            read_tool_geometry(self.path, ee_link="hand")

    def test_unknown_close_gap_at_raises(self) -> None:
        _write_asset(self.path, close_gap_at="middle")
        with self.assertRaisesRegex(ValueError, "close_gap_at"):
            read_tool_geometry(self.path, ee_link="hand")

    def test_rotating_jaw_takes_the_normal_between_the_pads(self) -> None:
        # USD authors a revolute joint's limits in degrees; the reader converts
        # to radians and takes the closing axis from the two pad frames, so the
        # declared travel may be the (shorter) arc length.
        _write_asset(
            self.path,
            joints=(("finger_left", 0.0, 45.0), ("finger_right", 0.0, 45.0)),
            revolute=True,
            travel=0.02,
        )
        tool = read_tool_geometry(self.path, ee_link="hand")
        self.assertAlmostEqual(tool.finger_q[1], math.radians(45.0), places=12)
        # The closing axis is the first pad frame minus the second.
        for got, want in zip(
            tool.pad_normal_in_ee, (0.0, 1.0, 0.0), strict=True
        ):
            self.assertAlmostEqual(got, want, places=9)


if __name__ == "__main__":
    unittest.main()
