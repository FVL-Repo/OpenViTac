from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np

from openvitac_data import (
    SCHEMA_VERSION,
    ActionSpec,
    CameraMeta,
    DepthCameraMeta,
    EpisodeExtra,
    EpisodeMeta,
    Frame,
    Hdf5EpisodeReader,
    Hdf5EpisodeWriter,
    Hdf5SchemaError,
    ObjectFrame,
    ObjectMeta,
    PadTactileMeta,
    TactileFrame,
)

_IMAGE = np.zeros((3, 4, 3), dtype=np.uint8)


def _meta(
    schema_version: int = SCHEMA_VERSION,
    *,
    tactile: tuple[PadTactileMeta, ...] = (),
    objects: tuple[ObjectMeta, ...] = (),
    episode_id: str = "",
    tactile_status: str = "ok",
) -> EpisodeMeta[EpisodeExtra]:
    return EpisodeMeta(
        schema_version=schema_version,
        task_id="test",
        sim_dt=8333,
        joint_names=("j0",),
        cameras=(
            CameraMeta(
                role="head",
                prim_path="/World/Camera",
                width=4,
                height=3,
                pos=(0.0, 0.0, 0.0),
                rot=(0.0, 0.0, 0.0, 1.0),
            ),
        ),
        action=ActionSpec("abs_joint_pos", ("j0",)),
        extra=EpisodeExtra(seed=0, instruction="", outcome="success"),
        episode_id=episode_id,
        tactile_status=cast(Any, tactile_status),
        tactile=tactile,
        objects=objects,
    )


def _frame(
    timestamp_us: int | None,
    *,
    objects: dict[str, ObjectFrame] | None = None,
    tactile: dict[str, TactileFrame] | None = None,
) -> Frame:
    return Frame(
        images={"head": _IMAGE},
        joint_pos=np.zeros(1, dtype=np.float32),
        joint_vel=np.zeros(1, dtype=np.float32),
        action=np.zeros(1, dtype=np.float32),
        tactile=tactile,
        timestamp_us=timestamp_us,
        objects=objects,
    )


def _pad(
    side: str,
    max_vertices: int = 4,
    depth_camera: DepthCameraMeta | None = None,
    image_streams: tuple[str, ...] = (),
) -> PadTactileMeta:
    return PadTactileMeta(
        side=side,
        rest_vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        surface_triangles=np.array([[0, 1, 2]], dtype=np.int32),
        pad_frame=np.eye(4, dtype=np.float64),
        youngs_mpa=0.1,
        strength=1e4,
        prim_path=f"/World/Robot/gelpad_{side}",
        attachment_body=f"gelsight_mini_case_{side}",
        max_vertices=max_vertices,
        width=320,
        height=240,
        marker_rows=6,
        marker_cols=7,
        depth_camera=depth_camera,
        image_streams=cast(Any, image_streams),
    )


def _tactile(
    side: str,
    n_vertices: int,
    seq: int,
    depth: np.ndarray | None = None,
    rgb: np.ndarray | None = None,
    marker_rgb: np.ndarray | None = None,
) -> TactileFrame:
    return TactileFrame(
        side=side,
        vertices=np.arange(n_vertices * 3, dtype=np.float32).reshape(n_vertices, 3),
        n_vertices=n_vertices,
        contact_gradient=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        force=None,
        force_kind="contact_gradient",
        contact_count=2,
        min_sd_proxy_mm=1.0,
        min_sd_full_mm=None,
        tracking_error_mm=0.5,
        seq=seq,
        age_ms=2.0,
        stale=False,
        valid=True,
        finger_seq=seq,
        physx_step=seq,
        finger_age_ms=1.5,
        finger_stale=False,
        pad_usb_min_sd_mm=0.25,
        pad_usb_contact_vertices=3,
        depth=depth,
        rgb=rgb,
        marker_rgb=marker_rgb,
    )


def _object(seq: int) -> ObjectFrame:
    return ObjectFrame(
        pose_xyzw=np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
        T=np.eye(4, dtype=np.float64),
        seq=seq,
        valid=True,
    )


def _object_meta() -> ObjectMeta:
    return ObjectMeta(
        name="usb",
        prim_path="/World/usb",
        authority="physx",
        constitution="rigid",
    )


class EpisodeRoundTripTests(unittest.TestCase):
    def test_write_read_round_trip_without_streams(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.h5"
            with Hdf5EpisodeWriter(path, _meta()) as writer:
                writer.append(_frame(0))
                writer.append(_frame(8333))
            with cast(Any, h5py.File(path, "r")) as file:
                self.assertEqual(int(file.attrs["schema_version"]), SCHEMA_VERSION)
                self.assertEqual(file["frames/timestamp_us"].shape, (2,))
            with Hdf5EpisodeReader(path) as reader:
                self.assertEqual(reader.meta.schema_version, SCHEMA_VERSION)
                frames = list(reader)
            self.assertEqual(len(frames), 2)
            self.assertIsNone(frames[0].tactile)
            self.assertIsNone(frames[0].objects)
            self.assertEqual(frames[1].timestamp_us, 8333)

    def test_policy_error_round_trips_with_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy-error.h5"
            with Hdf5EpisodeWriter(path, _meta()) as writer:
                writer.append(_frame(0))
                writer.set_outcome("policy_error")
            with Hdf5EpisodeReader(path) as reader:
                self.assertEqual(reader.meta.extra.outcome, "policy_error")
                frames = list(reader)
            self.assertEqual(len(frames), 1)

    def test_records_throttled_timestamps(self) -> None:
        # Default 120 Hz physics recording every 2nd step: 60 Hz frames.
        timestamps = [0, 16666, 33332]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "throttled.h5"
            with Hdf5EpisodeWriter(path, _meta()) as writer:
                for timestamp in timestamps:
                    writer.append(_frame(timestamp))
            with cast(Any, h5py.File(path, "r")) as file:
                self.assertEqual(file["frames/timestamp_us"][:].tolist(), timestamps)
            with Hdf5EpisodeReader(path) as reader:
                recorded = reader.timestamps_us
                self.assertIsNotNone(recorded)
                assert recorded is not None
                self.assertEqual(recorded.tolist(), timestamps)
                sim_dt = reader.meta.sim_dt
                frames = list(reader)
        self.assertEqual([frame.timestamp_us for frame in frames], timestamps)
        # The recorded interval is two physics steps, not ``sim_dt``.
        self.assertEqual(timestamps[1] - timestamps[0], 2 * sim_dt)

    def test_writer_requires_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "no_ts.h5"
            with (
                Hdf5EpisodeWriter(path, _meta()) as writer,
                self.assertRaises(ValueError),
            ):
                writer.append(_frame(None))


class V3RoundTripTests(unittest.TestCase):
    def test_v3_write_read_round_trip(self) -> None:
        pads = (_pad("left"), _pad("right"))
        objects = (
            ObjectMeta(
                name="usb",
                prim_path="/World/usb",
                authority="physx",
                constitution="rigid",
                mass_kg=0.1,
                inertia=np.array([1, 2, 3], dtype=np.float32),
                initial_pose_xyzw=np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
                usd="usb.usd",
                mesh_checksum="abc123",
            ),
        )
        meta = _meta(
            SCHEMA_VERSION,
            tactile=pads,
            objects=objects,
            episode_id="episode_1",
            tactile_status="ok",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v3.h5"
            with Hdf5EpisodeWriter(path, meta) as writer:
                for index, timestamp in enumerate((0, 8333)):
                    writer.append(
                        _frame(
                            timestamp,
                            objects={"usb": _object(index)},
                            tactile={
                                "left": _tactile("left", 3 - index, index),
                                "right": _tactile("right", 2, index),
                            },
                        )
                    )

            with cast(Any, h5py.File(path, "r")) as file:
                self.assertEqual(int(file.attrs["schema_version"]), 3)
                self.assertEqual(file.attrs["episode_id"], "episode_1")
                self.assertEqual(file.attrs["tactile_status"], "ok")
                self.assertEqual(file["frames/timestamp_us"].shape, (2,))
                self.assertEqual(file["frames/tactile/left/vertices"].shape, (2, 4, 3))
                self.assertEqual(
                    file["frames/tactile/left/n_vertices"][:].tolist(), [3, 2]
                )
                self.assertEqual(
                    file["frames/tactile/left/n_vertices"].dtype, np.dtype("int32")
                )
                self.assertEqual(
                    file["frames/tactile/left"].attrs["force_kind"],
                    "contact_gradient",
                )
                self.assertEqual(file["frames/objects/usb/T"].shape, (2, 4, 4))
                self.assertEqual(
                    file["meta/objects/usb/inertia_diag_kg_m2"].shape, (3,)
                )
                self.assertEqual(file["meta/objects/usb/initial_pose"].shape, (4, 4))
                self.assertEqual(file["meta/objects/usb"].attrs["authority"], "physx")
                self.assertEqual(
                    file["meta/tactile/left"].attrs["attachment_body"],
                    "gelsight_mini_case_left",
                )
                self.assertEqual(file["meta/tactile/left"].attrs["E_MPa"], 0.1)

            with Hdf5EpisodeReader(path) as reader:
                self.assertEqual(reader.meta.schema_version, 3)
                self.assertEqual(reader.meta.episode_id, "episode_1")
                self.assertEqual(reader.meta.tactile_status, "ok")
                self.assertEqual(
                    [pad.side for pad in reader.meta.tactile], ["left", "right"]
                )
                self.assertEqual(reader.meta.objects[0].mass_kg, 0.1)
                frames = list(reader)

            self.assertEqual(len(frames), 2)
            self.assertEqual([frame.timestamp_us for frame in frames], [0, 8333])
            for index, frame in enumerate(frames):
                objects = frame.objects
                tactile = frame.tactile
                assert objects is not None
                assert tactile is not None
                self.assertEqual(set(objects), {"usb"})
                self.assertEqual(objects["usb"].seq, index)
                np.testing.assert_allclose(
                    objects["usb"].pose_xyzw,
                    [0, 0, 0, 0, 0, 0, 1],
                )
                self.assertEqual(set(tactile), {"left", "right"})
                left = tactile["left"]
                self.assertEqual(left.side, "left")
                n_vertices = int(left.n_vertices)
                self.assertEqual(n_vertices, 3 - index)
                assert left.vertices is not None
                self.assertEqual(left.vertices.shape, (4, 3))
                self.assertTrue(np.all(left.vertices[n_vertices:] == 0))
                self.assertEqual(left.force_kind, "contact_gradient")
                self.assertIsNotNone(left.min_sd_full_mm)
                assert left.min_sd_full_mm is not None
                self.assertTrue(np.isnan(left.min_sd_full_mm))
                assert left.force is not None
                self.assertTrue(np.all(np.isnan(left.force)))
                assert left.pad_usb_min_sd_mm is not None
                self.assertAlmostEqual(float(left.pad_usb_min_sd_mm), 0.25)
                self.assertEqual(left.pad_usb_contact_vertices, 3)

    def test_v3_depth_round_trip(self) -> None:
        depth_camera = DepthCameraMeta(
            width=4,
            height=3,
            focal_length=20.0,
            horizontal_aperture=20.0,
            vertical_aperture=18.0,
            clipping_range=(0.02, 0.029),
            surface="contact_far",
            outside_aperture="zero",
        )
        depth = np.full((3, 4), 0.024, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "depth.h5"
            with Hdf5EpisodeWriter(
                path,
                _meta(
                    SCHEMA_VERSION,
                    tactile=(
                        _pad(
                            "left",
                            depth_camera=depth_camera,
                            image_streams=("depth",),
                        ),
                    ),
                ),
            ) as writer:
                writer.append(
                    _frame(
                        0,
                        tactile={"left": _tactile("left", 1, 0, depth=depth)},
                    )
                )
            with Hdf5EpisodeReader(path) as reader:
                meta = reader.meta.tactile[0].depth_camera
                self.assertEqual(meta, depth_camera)
                frame = next(iter(reader))
                assert frame.tactile is not None
                stored = frame.tactile["left"].depth
                assert stored is not None
                np.testing.assert_array_equal(stored, depth)

    def test_v3_rgb_round_trip(self) -> None:
        rgb = np.zeros((240, 320, 3), dtype=np.uint8)
        rgb[10:20, 30:40] = (10, 200, 40)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rgb.h5"
            with Hdf5EpisodeWriter(
                path,
                _meta(
                    SCHEMA_VERSION,
                    tactile=(_pad("left", image_streams=("rgb",)),),
                ),
            ) as writer:
                writer.append(
                    _frame(
                        0,
                        tactile={"left": _tactile("left", 1, 0, rgb=rgb)},
                    )
                )
            with Hdf5EpisodeReader(path) as reader:
                frame = next(iter(reader))
                assert frame.tactile is not None
                stored = frame.tactile["left"].rgb
                assert stored is not None
                self.assertEqual(stored.dtype, np.uint8)
                np.testing.assert_array_equal(stored, rgb)

    def test_v3_marker_rgb_round_trip(self) -> None:
        marker = np.zeros((240, 320, 3), dtype=np.uint8)
        marker[10:20, 30:40] = (250, 40, 10)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "marker.h5"
            with Hdf5EpisodeWriter(
                path,
                _meta(
                    SCHEMA_VERSION,
                    tactile=(_pad("left", image_streams=("marker_rgb",)),),
                ),
            ) as writer:
                writer.append(
                    _frame(
                        0,
                        tactile={"left": _tactile("left", 1, 0, marker_rgb=marker)},
                    )
                )
            with Hdf5EpisodeReader(path) as reader:
                frame = next(iter(reader))
                assert frame.tactile is not None
                stored = frame.tactile["left"].marker_rgb
                assert stored is not None
                self.assertEqual(stored.dtype, np.uint8)
                np.testing.assert_array_equal(stored, marker)

    def test_v3_requires_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v3_no_ts.h5"
            with (
                Hdf5EpisodeWriter(
                    path,
                    _meta(SCHEMA_VERSION, tactile=(_pad("left"),)),
                ) as writer,
                self.assertRaises(ValueError),
            ):
                writer.append(
                    _frame(
                        None,
                        objects={},
                        tactile={"left": _tactile("left", 1, 0)},
                    )
                )

    def test_v3_randomization_attrs_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v3_dr.h5"
            meta = _meta(SCHEMA_VERSION, tactile=(_pad("left"),))
            meta = replace(
                meta,
                extra=replace(
                    meta.extra,
                    randomization={
                        "dr.version": "insert_usb.scene_pose.v1",
                        "dr.slot_xy_half_m": 0.030,
                        "actor.start_slot.pos_x": 0.41,
                    },
                ),
            )
            with Hdf5EpisodeWriter(path, meta) as writer:
                writer.append(
                    _frame(
                        0,
                        objects={},
                        tactile={"left": _tactile("left", 1, 0)},
                    )
                )
            with Hdf5EpisodeReader(path) as reader:
                stored = reader.meta.extra.randomization
                self.assertEqual(stored["dr.version"], "insert_usb.scene_pose.v1")
                self.assertAlmostEqual(float(stored["dr.slot_xy_half_m"]), 0.030)
                self.assertAlmostEqual(float(stored["actor.start_slot.pos_x"]), 0.41)


class V3ImageStreamDeclarationTests(unittest.TestCase):
    """Dataset presence follows meta.image_streams, never the first frame."""

    @staticmethod
    def _rgb() -> np.ndarray:
        return np.zeros((240, 320, 3), dtype=np.uint8)

    def test_declared_stream_missing_on_first_frame_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "first_missing.h5"
            with (
                Hdf5EpisodeWriter(
                    path,
                    _meta(
                        SCHEMA_VERSION,
                        tactile=(_pad("left", image_streams=("rgb",)),),
                    ),
                ) as writer,
                self.assertRaisesRegex(ValueError, "rgb is required"),
            ):
                writer.append(
                    _frame(
                        0,
                        objects={},
                        tactile={"left": _tactile("left", 1, 0)},
                    )
                )

    def test_declared_stream_missing_on_later_frame_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "later_missing.h5"
            with Hdf5EpisodeWriter(
                path,
                _meta(
                    SCHEMA_VERSION,
                    tactile=(_pad("left", image_streams=("rgb",)),),
                ),
            ) as writer:
                writer.append(
                    _frame(
                        0,
                        objects={},
                        tactile={"left": _tactile("left", 1, 0, rgb=self._rgb())},
                    )
                )
                with self.assertRaisesRegex(ValueError, "rgb is required"):
                    writer.append(
                        _frame(
                            8333,
                            objects={},
                            tactile={"left": _tactile("left", 1, 1)},
                        )
                    )

    def test_undeclared_stream_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "undeclared.h5"
            with (
                Hdf5EpisodeWriter(
                    path,
                    _meta(SCHEMA_VERSION, tactile=(_pad("left"),)),
                ) as writer,
                self.assertRaisesRegex(ValueError, "does not declare"),
            ):
                writer.append(
                    _frame(
                        0,
                        objects={},
                        tactile={"left": _tactile("left", 1, 0, rgb=self._rgb())},
                    )
                )

    def test_declared_streams_determine_created_datasets(self) -> None:
        marker = self._rgb()
        marker[0, 0] = (1, 2, 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "declared.h5"
            with Hdf5EpisodeWriter(
                path,
                _meta(
                    SCHEMA_VERSION,
                    tactile=(_pad("left", image_streams=("marker_rgb",)),),
                ),
            ) as writer:
                for index in (0, 1):
                    writer.append(
                        _frame(
                            index * 8333,
                            objects={},
                            tactile={
                                "left": _tactile("left", 1, index, marker_rgb=marker)
                            },
                        )
                    )
            with cast(Any, h5py.File(path, "r")) as file:
                self.assertEqual(
                    file["meta/tactile/left"].attrs["image_streams"],
                    "marker_rgb",
                )
                names = set(file["frames/tactile/left"])
                self.assertIn("marker_rgb", names)
                self.assertNotIn("rgb", names)
                self.assertNotIn("depth", names)
            with Hdf5EpisodeReader(path) as reader:
                self.assertEqual(reader.meta.tactile[0].image_streams, ("marker_rgb",))
                frames = list(reader)
            self.assertEqual(len(frames), 2)

    def test_truncated_image_stream_is_rejected(self) -> None:
        truncated = np.zeros((0, 240, 320, 3), dtype=np.uint8)
        frame = Frame(
            images={"head": _IMAGE},
            joint_pos=np.zeros(1, dtype=np.float32),
            joint_vel=np.zeros(1, dtype=np.float32),
            action=np.zeros(1, dtype=np.float32),
            tactile={"left": _tactile("left", 1, 0, rgb=truncated)},
            timestamp_us=0,
            objects={},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truncated.h5"
            with (
                Hdf5EpisodeWriter(
                    path,
                    _meta(
                        SCHEMA_VERSION,
                        tactile=(_pad("left", image_streams=("rgb",)),),
                    ),
                ) as writer,
                self.assertRaisesRegex(ValueError, "expected 1"),
            ):
                writer.append(frame)


class ReaderSchemaValidationTests(unittest.TestCase):
    def _write_v3(self, path: Path, *, image_streams: tuple[str, ...] = ()) -> None:
        with Hdf5EpisodeWriter(
            path,
            _meta(
                SCHEMA_VERSION,
                tactile=(_pad("left", image_streams=image_streams),),
            ),
        ) as writer:
            for index in (0, 1):
                writer.append(
                    _frame(
                        index * 8333,
                        objects={},
                        tactile={"left": _tactile("left", 1, index)},
                    )
                )

    def test_mismatched_frame_lengths_raise_schema_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mismatch.h5"
            self._write_v3(path)
            with cast(Any, h5py.File(path, "r+")) as file:
                file["frames/tactile/left/vertices"].resize(1, axis=0)
            with self.assertRaisesRegex(Hdf5SchemaError, "disagree on length"):
                Hdf5EpisodeReader(path)

    def test_missing_v3_camera_pose_raises_schema_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "no_pose.h5"
            self._write_v3(path)
            with cast(Any, h5py.File(path, "r+")) as file:
                del file["meta/cameras/head/pos"]
            with self.assertRaises(Hdf5SchemaError):
                Hdf5EpisodeReader(path)

    def test_v3_ok_outcome_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v3_ok.h5"
            self._write_v3(path)
            with cast(Any, h5py.File(path, "r+")) as file:
                file.attrs["outcome"] = "ok"
            with self.assertRaisesRegex(ValueError, "invalid outcome 'ok'"):
                Hdf5EpisodeReader(path)


if __name__ == "__main__":
    unittest.main()
