from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from policy.RL.action import target_qpos_to_action
from policy.RL.dataset import InsertUSBBCDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_EPISODE = (
    REPO_ROOT
    / "data"
    / "insert_usb"
    / "gelsight"
    / "hdf5"
    / "4.hdf5"
)

# This scale is only for smoke testing. It is not the final training scale.
TEST_ACTION_SCALE = np.full(7, 0.01, dtype=np.float32)


@pytest.fixture
def dataset():
    if not REAL_EPISODE.is_file():
        pytest.skip(f"Real episode is unavailable: {REAL_EPISODE}")

    return InsertUSBBCDataset(
        hdf5_paths=[REAL_EPISODE],
        action_scale=TEST_ACTION_SCALE,
        image_size=64,
    )


def test_real_episode_sample_contract(dataset):
    # Episode 4 currently contains 31 transitions entering the insertion tag.
    assert len(dataset) == 31

    observation, action = dataset[0]

    assert set(observation) == {
        "qpos",
        "cam_high",
        "cam_wrist",
        "tac_left",
        "tac_right",
    }

    assert observation["qpos"].shape == (7,)
    assert observation["qpos"].dtype == torch.float32

    for image_key in (
        "cam_high",
        "cam_wrist",
        "tac_left",
        "tac_right",
    ):
        assert observation[image_key].shape == (3, 64, 64)
        assert observation[image_key].dtype == torch.uint8

    assert action.shape == (7,)
    assert action.dtype == torch.float32
    assert torch.all(action >= -1.0)
    assert torch.all(action <= 1.0)


def test_action_matches_recorded_joint_pair(dataset):
    episode_index, frame_index = dataset.records[0]
    layout = dataset.episodes[episode_index]

    with h5py.File(layout.path, "r") as hdf5_file:
        current_qpos = hdf5_file[
            "embodiment/joint"
        ][frame_index, :7].astype(np.float32)

        target_qpos = hdf5_file[
            "embodiment/joint"
        ][frame_index + 1, :7].astype(np.float32)

        next_tag = (
            hdf5_file["atom/tag"][frame_index + 1]
            .decode("utf-8")
            .lower()
        )

    expected_action = target_qpos_to_action(
        current_qpos=current_qpos,
        target_qpos=target_qpos,
        action_scale=TEST_ACTION_SCALE,
    )
    _, actual_action = dataset[0]

    assert next_tag == "insert_usb_into_slot"
    np.testing.assert_allclose(
        actual_action.numpy(),
        expected_action,
        atol=1e-6,
    )


def test_dataloader_batches_samples(dataset):
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    observation, action = next(iter(dataloader))

    assert observation["qpos"].shape == (4, 7)
    assert observation["cam_high"].shape == (4, 3, 64, 64)
    assert observation["cam_wrist"].shape == (4, 3, 64, 64)
    assert observation["tac_left"].shape == (4, 3, 64, 64)
    assert observation["tac_right"].shape == (4, 3, 64, 64)
    assert action.shape == (4, 7)


def test_multiple_tags_include_the_full_policy_suffix():
    if not REAL_EPISODE.is_file():
        pytest.skip(f"Real episode is unavailable: {REAL_EPISODE}")

    suffix_dataset = InsertUSBBCDataset(
        hdf5_paths=[REAL_EPISODE],
        action_scale=TEST_ACTION_SCALE,
        image_size=64,
        insertion_tag=(
            "move_usb_to_play_pre_insert,insert_usb_into_slot"
        ),
    )

    assert len(suffix_dataset) == 62
    with h5py.File(REAL_EPISODE, "r") as hdf5_file:
        destination_tags = {
            hdf5_file["atom/tag"][frame_index + 1]
            .decode("utf-8")
            .lower()
            for _, frame_index in suffix_dataset.records
        }

    assert destination_tags == {
        "move_usb_to_play_pre_insert",
        "insert_usb_into_slot",
    }


def test_action_horizon_uses_a_future_saved_joint_state():
    if not REAL_EPISODE.is_file():
        pytest.skip(f"Real episode is unavailable: {REAL_EPISODE}")

    horizon = 4
    horizon_dataset = InsertUSBBCDataset(
        hdf5_paths=[REAL_EPISODE],
        action_scale=TEST_ACTION_SCALE,
        image_size=64,
        insertion_tag=(
            "move_usb_to_play_pre_insert,insert_usb_into_slot"
        ),
        action_horizon=horizon,
    )
    episode_index, frame_index = horizon_dataset.records[0]
    target_index = horizon_dataset.target_indices[0]
    layout = horizon_dataset.episodes[episode_index]

    assert target_index == frame_index + horizon
    with h5py.File(layout.path, "r") as hdf5_file:
        current_qpos = hdf5_file[
            "embodiment/joint"
        ][frame_index, :7].astype(np.float32)
        target_qpos = hdf5_file[
            "embodiment/joint"
        ][target_index, :7].astype(np.float32)
    expected_action = target_qpos_to_action(
        current_qpos=current_qpos,
        target_qpos=target_qpos,
        action_scale=TEST_ACTION_SCALE,
    )
    _, actual_action = horizon_dataset[0]
    np.testing.assert_allclose(actual_action.numpy(), expected_action)


@pytest.mark.parametrize("action_horizon", [0, -1, True])
def test_action_horizon_must_be_a_positive_integer(action_horizon):
    if not REAL_EPISODE.is_file():
        pytest.skip(f"Real episode is unavailable: {REAL_EPISODE}")
    with pytest.raises(ValueError, match="action_horizon"):
        InsertUSBBCDataset(
            hdf5_paths=[REAL_EPISODE],
            action_scale=TEST_ACTION_SCALE,
            image_size=64,
            action_horizon=action_horizon,
        )
