import numpy as np

from scripts.rfcl.concat_full_trajectories import downsample_transition_valid


def test_downsample_transition_valid_preserves_skipped_invalid_transition():
    transition_valid = np.asarray([0, 1, 0, 1, 1], dtype=bool)
    selected_indices = np.asarray([0, 1, 3, 4], dtype=np.int64)

    selected_valid = downsample_transition_valid(
        transition_valid,
        selected_indices,
        first_valid=False,
    )

    np.testing.assert_array_equal(selected_valid, [0, 1, 0, 1])


def test_downsample_transition_valid_can_join_from_an_external_source():
    transition_valid = np.asarray([0, 1, 1], dtype=bool)
    selected_indices = np.asarray([0, 2], dtype=np.int64)

    selected_valid = downsample_transition_valid(
        transition_valid,
        selected_indices,
        first_valid=True,
    )

    np.testing.assert_array_equal(selected_valid, [1, 1])
