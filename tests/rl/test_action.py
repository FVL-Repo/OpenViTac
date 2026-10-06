import numpy as np
import pytest

from policy.RL.action import (
    action_to_target_qpos,
    clip_action,
    target_qpos_to_action,
)


def test_unbatched_action_round_trip():
    current_qpos = np.array(
        [0.10, -0.20, 0.30, -0.40, 0.50, -0.60, 0.70],
        dtype=np.float32,
    )
    action_scale = np.array(
        [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
        dtype=np.float32,
    )
    expected_action = np.array(
        [-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0],
        dtype=np.float32,
    )
    target_qpos = current_qpos + action_scale * expected_action

    encoded_action = target_qpos_to_action(
        current_qpos,
        target_qpos,
        action_scale,
    )
    decoded_target = action_to_target_qpos(
        current_qpos,
        encoded_action,
        action_scale,
    )

    np.testing.assert_allclose(encoded_action, expected_action, atol=1e-6)
    np.testing.assert_allclose(decoded_target, target_qpos, atol=1e-6)
    assert encoded_action.dtype == np.float32
    assert decoded_target.dtype == np.float32


def test_batched_action_round_trip():
    current_qpos = np.array(
        [
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        ],
        dtype=np.float32,
    )
    expected_action = np.array(
        [
            [-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0],
            [1.0, 0.75, 0.5, 0.25, 0.0, -0.5, -1.0],
        ],
        dtype=np.float32,
    )
    action_scale = np.full(7, 0.05, dtype=np.float32)
    target_qpos = current_qpos + action_scale * expected_action

    encoded_action = target_qpos_to_action(
        current_qpos,
        target_qpos,
        action_scale,
    )
    decoded_target = action_to_target_qpos(
        current_qpos,
        encoded_action,
        action_scale,
    )

    assert encoded_action.shape == (2, 7)
    assert decoded_target.shape == (2, 7)
    np.testing.assert_allclose(encoded_action, expected_action, atol=1e-6)
    np.testing.assert_allclose(decoded_target, target_qpos, atol=1e-6)


def test_actions_are_clipped():
    current_qpos = np.zeros(7, dtype=np.float32)
    action_scale = np.full(7, 0.1, dtype=np.float32)
    raw_action = np.array(
        [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
        dtype=np.float32,
    )
    expected_action = np.array(
        [-1.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.0],
        dtype=np.float32,
    )

    clipped_action = clip_action(raw_action)
    decoded_target = action_to_target_qpos(
        current_qpos,
        raw_action,
        action_scale,
    )
    encoded_action = target_qpos_to_action(
        current_qpos,
        action_scale * raw_action,
        action_scale,
    )

    np.testing.assert_array_equal(clipped_action, expected_action)
    np.testing.assert_allclose(
        decoded_target,
        action_scale * expected_action,
        atol=1e-6,
    )
    np.testing.assert_array_equal(encoded_action, expected_action)


def test_rejects_wrong_action_dimension():
    with pytest.raises(ValueError, match="trailing dimension 7"):
        clip_action(np.zeros(6, dtype=np.float32))


def test_rejects_mismatched_qpos_shapes():
    current_qpos = np.zeros(7, dtype=np.float32)
    target_qpos = np.zeros((2, 7), dtype=np.float32)
    action_scale = np.ones(7, dtype=np.float32)

    with pytest.raises(ValueError, match="must have the same shape"):
        target_qpos_to_action(
            current_qpos,
            target_qpos,
            action_scale,
        )


@pytest.mark.parametrize(
    "invalid_scale",
    [
        np.zeros(7, dtype=np.float32),
        -np.ones(7, dtype=np.float32),
        np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.nan]),
    ],
)
def test_rejects_invalid_action_scale(invalid_scale):
    with pytest.raises(ValueError):
        action_to_target_qpos(
            current_qpos=np.zeros(7, dtype=np.float32),
            action=np.zeros(7, dtype=np.float32),
            action_scale=invalid_scale,
        )