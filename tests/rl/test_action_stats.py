from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest

from policy.RL.action_stats import (
    compute_action_statistics,
    split_episode_paths,
)


def _encoded_frames(frame_count: int) -> np.ndarray:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    assert success

    payload = encoded.tobytes()
    return np.asarray(
        [payload] * frame_count,
        dtype=f"S{len(payload)}",
    )


def _write_episode(path: Path, deltas: np.ndarray) -> None:
    deltas = np.asarray(deltas, dtype=np.float32)
    frame_count = len(deltas) + 1

    joints = np.zeros((frame_count, 9), dtype=np.float32)
    joints[1:, :7] = np.cumsum(deltas, axis=0)

    tags = np.asarray(
        ["delay"] + ["insert_usb_into_slot"] * len(deltas),
        dtype="S24",
    )
    encoded_frames = _encoded_frames(frame_count)

    with h5py.File(path, "w") as hdf5_file:
        embodiment = hdf5_file.create_group("embodiment")
        embodiment.create_dataset("joint", data=joints)

        atom = hdf5_file.create_group("atom")
        atom.create_dataset("tag", data=tags)

        phase = hdf5_file.create_group("phase")
        phase.create_dataset(
            "id",
            data=np.ones(frame_count, dtype=np.int64),
        )
        phase.attrs["policy_id"] = 1

        observation = hdf5_file.create_group("observation")
        observation.create_group("head").create_dataset(
            "rgb",
            data=encoded_frames,
        )
        observation.create_group("wrist").create_dataset(
            "rgb",
            data=encoded_frames,
        )

        tactile = hdf5_file.create_group("tactile")
        tactile.create_group("left_tactile").create_dataset(
            "rgb_marker",
            data=encoded_frames,
        )
        tactile.create_group("right_tactile").create_dataset(
            "rgb_marker",
            data=encoded_frames,
        )


def test_episode_split_is_reproducible_and_disjoint(tmp_path):
    paths = [tmp_path / f"{index}.hdf5" for index in range(10)]

    first_train, first_validation = split_episode_paths(
        paths,
        validation_fraction=0.2,
        seed=7,
    )
    second_train, second_validation = split_episode_paths(
        reversed(paths),
        validation_fraction=0.2,
        seed=7,
    )

    assert first_train == second_train
    assert first_validation == second_validation
    assert len(first_train) == 8
    assert len(first_validation) == 2
    assert set(first_train).isdisjoint(first_validation)
    assert set(first_train + first_validation) == set(paths)


def test_action_statistics_match_selected_joint_deltas(tmp_path):
    deltas = np.asarray(
        [
            [0.10, -0.20, 0.00, 0.40, -0.50, 0.60, -0.70],
            [-0.30, 0.10, 0.00, -0.20, 0.40, -0.10, 0.20],
        ],
        dtype=np.float32,
    )
    episode_path = tmp_path / "episode.hdf5"
    _write_episode(episode_path, deltas)

    statistics = compute_action_statistics(
        [episode_path],
        scale_margin=1.10,
        minimum_scale=0.05,
    )

    absolute_delta = np.abs(deltas)
    expected_abs_max = absolute_delta.max(axis=0)
    expected_scale = np.maximum(expected_abs_max * 1.10, 0.05)

    assert statistics.transition_count == 2
    np.testing.assert_allclose(statistics.delta_mean, deltas.mean(axis=0))
    np.testing.assert_allclose(statistics.delta_std, deltas.std(axis=0))
    np.testing.assert_allclose(
        statistics.delta_abs_p95,
        np.percentile(absolute_delta, 95, axis=0),
    )
    np.testing.assert_allclose(
        statistics.delta_abs_p99,
        np.percentile(absolute_delta, 99, axis=0),
    )
    np.testing.assert_allclose(statistics.delta_abs_max, expected_abs_max)
    np.testing.assert_allclose(statistics.action_scale, expected_scale)

    normalized_action = deltas / statistics.action_scale
    assert np.all(np.abs(normalized_action) < 1.0)


def test_action_statistics_use_the_requested_horizon(tmp_path):
    deltas = np.asarray(
        [
            [0.1] * 7,
            [0.2] * 7,
            [-0.1] * 7,
        ],
        dtype=np.float32,
    )
    episode_path = tmp_path / "episode.hdf5"
    _write_episode(episode_path, deltas)

    statistics = compute_action_statistics(
        [episode_path],
        action_horizon=2,
        scale_margin=1.05,
    )

    expected_deltas = np.asarray(
        [deltas[0] + deltas[1], deltas[1] + deltas[2]]
    )
    assert statistics.transition_count == 2
    np.testing.assert_allclose(
        statistics.delta_mean,
        expected_deltas.mean(axis=0),
    )
    np.testing.assert_allclose(
        statistics.delta_abs_max,
        np.abs(expected_deltas).max(axis=0),
    )


def test_transition_valid_excludes_source_boundary(tmp_path):
    deltas = np.asarray(
        [
            [0.1] * 7,
            [3.0] * 7,
            [0.2] * 7,
        ],
        dtype=np.float32,
    )
    episode_path = tmp_path / "episode.hdf5"
    _write_episode(episode_path, deltas)
    with h5py.File(episode_path, "a") as hdf5_file:
        provenance = hdf5_file.create_group("provenance")
        provenance.create_dataset(
            "transition_valid",
            data=np.asarray([0, 1, 0, 1], dtype=np.int8),
        )

    statistics = compute_action_statistics([episode_path])

    assert statistics.transition_count == 2
    np.testing.assert_allclose(
        statistics.delta_abs_max,
        np.asarray([0.2] * 7),
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("scale_margin", "minimum_scale"),
    [
        (0.99, 1e-6),
        (1.05, 0.0),
        (1.05, -1e-6),
    ],
)
def test_action_statistics_reject_invalid_scale_settings(
    scale_margin,
    minimum_scale,
):
    with pytest.raises(ValueError):
        compute_action_statistics(
            [],
            scale_margin=scale_margin,
            minimum_scale=minimum_scale,
        )
