from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import h5py
import numpy as np
from numpy.typing import NDArray
from .action import ARM_ACTION_DIM
from .dataset import InsertUSBBCDataset

@dataclass(frozen=True)
class ActionStatistics:
    transition_count: int
    delta_mean: NDArray[np.float32]
    delta_std: NDArray[np.float32]
    delta_abs_p95: NDArray[np.float32]
    delta_abs_p99: NDArray[np.float32]
    delta_abs_max: NDArray[np.float32]
    action_scale: NDArray[np.float32]


def split_episode_paths(
    hdf5_paths: Sequence[str | Path],
    validation_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[list[Path], list[Path]]:
    """Split complete episodes into deterministic train/validation sets."""
    paths = sorted(
        (Path(path) for path in hdf5_paths),
        key=lambda path: str(path),
    )

    if len(paths) < 2:
        raise ValueError("At least two episodes are required for splitting")

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "validation_fraction must be strictly between 0 and 1"
        )

    generator = np.random.default_rng(seed)
    shuffled_indices = generator.permutation(len(paths))

    validation_count = max(
        1,
        round(len(paths) * validation_fraction),
    )
    validation_count = min(validation_count, len(paths) - 1)

    validation_indices = set(
        shuffled_indices[:validation_count].tolist()
    )

    train_paths = [
        path
        for index, path in enumerate(paths)
        if index not in validation_indices
    ]
    validation_paths = [
        path
        for index, path in enumerate(paths)
        if index in validation_indices
    ]

    return train_paths, validation_paths

def compute_action_statistics(
    hdf5_paths: Sequence[str | Path],
    insertion_tag: str = "insert_usb_into_slot",
    require_policy_phase: bool = True,
    action_horizon: int = 1,
    scale_margin: float = 1.05,
    minimum_scale: float = 1e-6,
) -> ActionStatistics:
    """Compute arm-delta statistics from selected insertion transitions."""
    if scale_margin < 1.0:
        raise ValueError("scale_margin must be at least 1.0")

    if minimum_scale <= 0.0:
        raise ValueError("minimum_scale must be positive")

    # Build exactly the same phase/tag transition index as the BC Dataset.
    selector = InsertUSBBCDataset(
        hdf5_paths=hdf5_paths,
        action_scale=np.ones(ARM_ACTION_DIM, dtype=np.float32),
        image_size=1,
        insertion_tag=insertion_tag,
        require_policy_phase=require_policy_phase,
        action_horizon=action_horizon,
    )

    records_by_episode: dict[int, list[tuple[int, int]]] = {}
    for record_index, (episode_index, frame_index) in enumerate(
        selector.records
    ):
        records_by_episode.setdefault(
            episode_index,
            [],
        ).append(
            (frame_index, selector.target_indices[record_index])
        )

    episode_deltas = []

    for episode_index, frame_pairs in records_by_episode.items():
        episode = selector.episodes[episode_index]
        indices = np.asarray(frame_pairs, dtype=np.int64)

        with h5py.File(episode.path, "r") as hdf5_file:
            joints = hdf5_file["embodiment/joint"]

            current_qpos = np.asarray(
                joints[indices[:, 0], :ARM_ACTION_DIM],
                dtype=np.float64,
            )
            target_qpos = np.asarray(
                joints[indices[:, 1], :ARM_ACTION_DIM],
                dtype=np.float64,
            )

        episode_deltas.append(target_qpos - current_qpos)

    delta = np.concatenate(episode_deltas, axis=0)

    if not np.all(np.isfinite(delta)):
        raise ValueError("Joint deltas contain NaN or infinite values")

    absolute_delta = np.abs(delta)
    delta_abs_max = absolute_delta.max(axis=0)
    action_scale = np.maximum(
        delta_abs_max * scale_margin,
        minimum_scale,
    )

    return ActionStatistics(
        transition_count=len(delta),
        delta_mean=delta.mean(axis=0).astype(np.float32),
        delta_std=delta.std(axis=0).astype(np.float32),
        delta_abs_p95=np.percentile(
            absolute_delta,
            95,
            axis=0,
        ).astype(np.float32),
        delta_abs_p99=np.percentile(
            absolute_delta,
            99,
            axis=0,
        ).astype(np.float32),
        delta_abs_max=delta_abs_max.astype(np.float32),
        action_scale=action_scale.astype(np.float32),
    )
