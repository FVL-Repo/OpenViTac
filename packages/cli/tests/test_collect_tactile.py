from __future__ import annotations

import inspect
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np

from openvitac_cli.commands.collect import (
    _tactile_meta,
    _tactile_payload,
    collect,
)
from openvitac_data import (
    ActionSpec,
    CameraMeta,
    EpisodeExtra,
    EpisodeMeta,
    Frame,
    Hdf5EpisodeReader,
    Hdf5EpisodeWriter,
)
from openvitac_sim import AttachmentBody, GsMiniCfg
from openvitac_sim.tactile_bridge import PadSurface, tactile_cfg_from_sensor
from openvitac_tactile import (
    ObjectCfg,
    ObjectObservation,
    PadObservation,
    TactileObservation,
)


def _cfg():
    return tactile_cfg_from_sensor(
        GsMiniCfg(
            name="tactile",
            prim_path="/World/Robot/gelsight_mini_case_left",
            gelpad_prim_path="/World/Robot/gelpad_left",
            attachment_body=AttachmentBody(
                "gelsight_mini_case_left",
                "/World/Robot/gelsight_mini_case_left",
            ),
            gripper_kind="franka_panda",
        ),
        objects=(
            ObjectCfg(
                "usb",
                "/World/actors/usb",
                mesh_path="usb_tet.npz",
                usd_path="usb.usd",
            ),
        ),
    )


def _pad(side: str, offset: tuple[float, float, float]) -> PadObservation:
    vertices = np.array([[0, 0, 0], [0.001, 0, 0], [0, 0.001, 0]], dtype=np.float32)
    return PadObservation(
        side=side,
        vertices=vertices + np.asarray(offset, dtype=np.float32),
        n_vertices=3,
        contact_gradient=np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        force=None,
        force_kind="contact_gradient",
        contact_count=2,
        min_sd_proxy_mm=1.0,
        min_sd_full_mm=None,
        tracking_error_mm=0.5,
        seq=7,
        age_ms=2.0,
        stale=False,
    )


def _observation(
    *,
    objects: dict[str, ObjectObservation] | None = None,
    pads: dict[str, PadObservation] | None = None,
) -> TactileObservation:
    resolved_pads = (
        {
            "left": _pad("left", (0.10, -0.02, 0.30)),
            "right": _pad("right", (0.10, 0.02, 0.30)),
        }
        if pads is None
        else pads
    )
    return TactileObservation(
        seq=7,
        finger_seq=7,
        physx_step=7,
        objects=(
            {
                "usb": ObjectObservation(
                    name="usb",
                    pose_xyzw=np.array([0, 0, 0.035, 0, 0, 0, 1], dtype=np.float32),
                )
            }
            if objects is None
            else objects
        ),
        pads=resolved_pads,
        tactile_rgb={
            side: np.zeros((240, 320, 3), dtype=np.uint8) for side in resolved_pads
        },
        age_ms=2.0,
        finger_age_ms=1.0,
    )


def _invalid_object(name: str = "usb") -> ObjectObservation:
    return ObjectObservation(
        name=name,
        pose_xyzw=np.array([np.nan, 0, 0, 0, 0, 0, 1], dtype=np.float32),
    )


class TactileCollectTests(unittest.TestCase):
    def test_collect_has_no_public_tactile_switch(self) -> None:
        self.assertNotIn("tactile", inspect.signature(collect).parameters)

    def test_v3_round_trip_from_synthetic_observation(self) -> None:
        cfg = _cfg()
        left_world = np.eye(4)
        left_world[:3, 3] = (0.10, -0.02, 0.30)
        right_world = np.eye(4)
        right_world[:3, 3] = (0.10, 0.02, 0.30)
        left_rest = np.array(
            [[0, 0, 0], [0.001, 0, 0], [0, 0.001, 0]], dtype=np.float64
        )
        right_rest = left_rest.copy()
        left_surface = PadSurface(
            name="tactile_left",
            side="left",
            rest_vertices=left_rest + left_world[:3, 3],
            triangles=np.array([[0, 1, 2]], dtype=np.int32),
        )
        right_surface = PadSurface(
            name="tactile_right",
            side="right",
            rest_vertices=right_rest + right_world[:3, 3],
            triangles=np.array([[0, 1, 2]], dtype=np.int32),
        )
        pads, objects = _tactile_meta(
            cfg,
            pad_world_frames={
                "tactile_left": left_world,
                "tactile_right": right_world,
            },
            pad_surfaces={
                "tactile_left": left_surface,
                "tactile_right": right_surface,
            },
        )
        self.assertEqual(len(pads), 2)
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0].usd, "usb.usd")
        self.assertNotEqual(objects[0].usd, "usb_tet.npz")
        np.testing.assert_allclose(pads[0].pad_frame, left_world)
        np.testing.assert_allclose(pads[1].pad_frame, right_world)
        self.assertFalse(np.allclose(pads[0].pad_frame, np.eye(4)))
        self.assertFalse(np.allclose(pads[1].pad_frame, np.eye(4)))
        self.assertFalse(np.allclose(pads[0].pad_frame, pads[1].pad_frame))
        camera = CameraMeta(
            role="head",
            prim_path="/World/Camera",
            width=3,
            height=2,
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        )
        meta = EpisodeMeta(
            schema_version=3,
            task_id="test",
            sim_dt=8333,
            joint_names=("j0",),
            cameras=(camera,),
            action=ActionSpec("abs_joint_pos", ("j0",)),
            extra=EpisodeExtra(seed=0, instruction="", outcome="success"),
            episode_id="episode",
            tactile_status="ok",
            tactile=pads,
            objects=objects,
        )
        image = np.zeros((2, 3, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.h5"
            with Hdf5EpisodeWriter(path, meta) as writer:
                for index in range(2):
                    timestamp = index * 8333
                    observation = _observation()
                    if index == 1:
                        # The second frame records a worker-invalid pad; the
                        # writer must persist that bit as-is.
                        observation = _observation(
                            pads={
                                **observation.pads,
                                "left": replace(observation.pads["left"], valid=False),
                            }
                        )
                    tactile, object_frames = _tactile_payload(
                        observation, timestamp_us=timestamp
                    )
                    writer.append(
                        Frame(
                            images={"head": image},
                            joint_pos=np.zeros(1),
                            joint_vel=np.zeros(1),
                            action=np.zeros(1),
                            tactile=tactile,
                            objects=object_frames,
                            timestamp_us=timestamp,
                        )
                    )
            with Hdf5EpisodeReader(path) as reader:
                self.assertEqual(reader.meta.schema_version, 3)
                self.assertEqual(reader.meta.episode_id, "episode")
                self.assertEqual(reader.meta.tactile_status, "ok")
                self.assertEqual(len(reader.meta.tactile), 2)
                frames = list(reader)
            self.assertEqual(len(frames), 2)
            self.assertEqual([frame.timestamp_us for frame in frames], [0, 8333])
            for index, frame in enumerate(frames):
                assert frame.tactile is not None
                assert frame.objects is not None
                self.assertEqual(set(frame.tactile), {"left", "right"})
                self.assertEqual(set(frame.objects), {"usb"})
                self.assertEqual(frame.tactile["left"].timestamp_us, index * 8333)
                self.assertEqual(frame.tactile["left"].valid, index == 0)
                self.assertFalse(frame.tactile["left"].stale)
                self.assertTrue(frame.tactile["right"].valid)
                self.assertEqual(frame.tactile["left"].n_vertices, 3)
                assert frame.tactile["left"].vertices is not None
                self.assertEqual(frame.tactile["left"].vertices.shape, (1024, 3))
                self.assertTrue(np.all(frame.tactile["left"].vertices[3:] == 0))
                self.assertEqual(frame.objects["usb"].pose_xyzw.shape, (7,))
                for side, pad_frame in (
                    ("left", left_world),
                    ("right", right_world),
                ):
                    meta_pad = next(p for p in reader.meta.tactile if p.side == side)
                    frame_pad = cast(dict[str, Any], frame.tactile)[side]
                    n = frame_pad.n_vertices
                    world_rest = (
                        meta_pad.pad_frame[:3, :3] @ meta_pad.rest_vertices.T
                    ).T + meta_pad.pad_frame[:3, 3]
                    np.testing.assert_allclose(
                        world_rest,
                        frame_pad.vertices[:n],
                        atol=1e-6,
                    )
                    np.testing.assert_array_equal(
                        meta_pad.surface_triangles,
                        np.array([[0, 1, 2]], dtype=np.int32),
                    )
                    np.testing.assert_allclose(meta_pad.pad_frame, pad_frame)

    def test_payload_uses_real_pad_valid_not_stale(self) -> None:
        pads = {
            # No worker vertices: invalid, yet not stale.
            "left": replace(
                _pad("left", (0.10, -0.02, 0.30)),
                vertices=None,
                n_vertices=0,
                stale=False,
                valid=False,
            ),
            # Stale transport state does not make the geometry invalid.
            "right": replace(_pad("right", (0.10, 0.02, 0.30)), stale=True, valid=True),
        }
        tactile, _ = _tactile_payload(_observation(pads=pads), timestamp_us=0)
        self.assertFalse(tactile["left"].valid)
        self.assertFalse(tactile["left"].stale)
        self.assertEqual(tactile["left"].n_vertices, 0)
        self.assertTrue(tactile["right"].valid)
        self.assertTrue(tactile["right"].stale)

    def test_invalid_physx_pose_uses_scene_fallback_and_revalidates(self) -> None:
        queried: list[str] = []

        def reader(
            prim_path: str,
        ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
            queried.append(prim_path)
            return (0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0)

        _, objects = _tactile_payload(
            _observation(objects={"usb": _invalid_object()}),
            timestamp_us=0,
            object_prim_paths={"usb": "/World/usb"},
            object_authorities={"usb": "physx"},
            pose_reader=reader,
        )
        self.assertEqual(queried, ["/World/usb"])
        self.assertTrue(objects["usb"].valid)
        np.testing.assert_allclose(
            objects["usb"].pose_xyzw,
            [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
        )

    def test_physx_fallback_stays_invalid_when_scene_pose_is_bad(self) -> None:
        _, objects = _tactile_payload(
            _observation(objects={"usb": _invalid_object()}),
            timestamp_us=0,
            object_prim_paths={"usb": "/World/usb"},
            object_authorities={"usb": "physx"},
            pose_reader=lambda _: ((np.nan, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        )
        self.assertFalse(objects["usb"].valid)

    def test_uipc_invalid_pose_never_uses_kinematic_proxy(self) -> None:
        def reader(
            prim_path: str,
        ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
            raise AssertionError(f"UIPC object {prim_path!r} must not be read back")

        _, objects = _tactile_payload(
            _observation(objects={"usb": _invalid_object()}),
            timestamp_us=0,
            object_prim_paths={"usb": "/World/usb"},
            object_authorities={"usb": "uipc"},
            pose_reader=reader,
        )
        self.assertFalse(objects["usb"].valid)
        self.assertTrue(np.isnan(objects["usb"].pose_xyzw[0]))

    def test_single_pad_payload_uses_observed_pads(self) -> None:
        observation = _observation(pads={"left": _pad("left", (0.10, -0.02, 0.30))})
        tactile, _ = _tactile_payload(observation, timestamp_us=0)
        self.assertEqual(set(tactile), {"left"})
        self.assertEqual(tactile["left"].side, "left")

    def test_single_pad_episode_round_trip(self) -> None:
        pair_cfg = _cfg()
        cfg = replace(pair_cfg, pads=(pair_cfg.pads[0],))
        left_world = np.eye(4)
        left_world[:3, 3] = (0.10, -0.02, 0.30)
        rest = np.array([[0, 0, 0], [0.001, 0, 0], [0, 0.001, 0]], dtype=np.float64)
        surface = PadSurface(
            name="tactile_left",
            side="left",
            rest_vertices=rest + left_world[:3, 3],
            triangles=np.array([[0, 1, 2]], dtype=np.int32),
        )
        pads, objects = _tactile_meta(
            cfg,
            pad_world_frames={"tactile_left": left_world},
            pad_surfaces={"tactile_left": surface},
        )
        self.assertEqual([pad.side for pad in pads], ["left"])
        meta = EpisodeMeta(
            schema_version=3,
            task_id="test",
            sim_dt=8333,
            joint_names=("j0",),
            cameras=(
                CameraMeta(
                    role="head",
                    prim_path="/World/Camera",
                    width=3,
                    height=2,
                    pos=(0.0, 0.0, 0.0),
                    rot=(0.0, 0.0, 0.0, 1.0),
                ),
            ),
            action=ActionSpec("abs_joint_pos", ("j0",)),
            extra=EpisodeExtra(seed=0, instruction="", outcome="success"),
            episode_id="episode",
            tactile_status="ok",
            tactile=pads,
            objects=objects,
        )
        observation = _observation(pads={"left": _pad("left", (0.10, -0.02, 0.30))})
        image = np.zeros((2, 3, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.h5"
            with Hdf5EpisodeWriter(path, meta) as writer:
                for index in range(2):
                    timestamp = index * 16666
                    tactile, object_frames = _tactile_payload(
                        observation, timestamp_us=timestamp
                    )
                    writer.append(
                        Frame(
                            images={"head": image},
                            joint_pos=np.zeros(1),
                            joint_vel=np.zeros(1),
                            action=np.zeros(1),
                            tactile=tactile,
                            objects=object_frames,
                            timestamp_us=timestamp,
                        )
                    )
            with Hdf5EpisodeReader(path) as reader:
                self.assertEqual([pad.side for pad in reader.meta.tactile], ["left"])
                frames = list(reader)
            self.assertEqual(len(frames), 2)
            self.assertEqual(
                set(frames[0].tactile or {}),
                {"left"},
            )


if __name__ == "__main__":
    unittest.main()
