from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))

from policy.RL.checkpoint import load_bc_checkpoint, restore_actor_from_bc_checkpoint
from scripts.train_single_step_bc import datasets_from_checkpoint


NOMINAL_TARGET_SLOT_XY = np.asarray([0.52, 0.0], dtype=np.float64)
USB_INSERT_Z = 0.008


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare deterministic BC actions with held-out expert actions, "
            "including coarse-like insertion states."
        )
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _validation_metadata(dataset) -> dict[str, np.ndarray]:
    sample_count = len(dataset)
    tags = np.empty(sample_count, dtype=object)
    xy_error_mm = np.empty(sample_count, dtype=np.float64)
    z_error_mm = np.empty(sample_count, dtype=np.float64)
    slot_radius_mm = np.empty(sample_count, dtype=np.float64)

    indices_by_episode: dict[int, list[tuple[int, int, int]]] = {}
    for sample_index, ((episode_index, frame_index), target_index) in enumerate(
        zip(dataset.records, dataset.target_indices, strict=True)
    ):
        indices_by_episode.setdefault(episode_index, []).append(
            (sample_index, frame_index, target_index)
        )

    for episode_index, entries in indices_by_episode.items():
        path = dataset.episodes[episode_index].path
        with h5py.File(path, "r") as hdf5_file:
            prism_positions = np.asarray(
                hdf5_file["actor/prism"][:, :3], dtype=np.float64
            )
            slot_positions = np.asarray(
                hdf5_file["actor/slot"][:, :3], dtype=np.float64
            )
            episode_tags = np.asarray(hdf5_file["atom/tag"][()]).astype("U")

            for sample_index, frame_index, target_index in entries:
                relative_position = (
                    prism_positions[frame_index] - slot_positions[frame_index]
                )
                relative_position[2] -= USB_INSERT_Z
                tags[sample_index] = episode_tags[target_index].lower()
                xy_error_mm[sample_index] = (
                    np.linalg.norm(relative_position[:2]) * 1000.0
                )
                z_error_mm[sample_index] = relative_position[2] * 1000.0
                slot_radius_mm[sample_index] = (
                    np.linalg.norm(
                        slot_positions[frame_index, :2]
                        - NOMINAL_TARGET_SLOT_XY
                    )
                    * 1000.0
                )

    return {
        "tag": tags,
        "xy_error_mm": xy_error_mm,
        "z_error_mm": z_error_mm,
        "slot_radius_mm": slot_radius_mm,
    }


def _group_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, object]:
    predicted = predicted[mask]
    target = target[mask]
    if len(predicted) == 0:
        return {"sample_count": 0}

    predicted_norm = np.linalg.norm(predicted, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    valid_direction = target_norm > 1e-6
    denominator = predicted_norm * target_norm
    valid_cosine = valid_direction & (denominator > 1e-8)
    cosine = np.full(len(predicted), np.nan, dtype=np.float64)
    cosine[valid_cosine] = (
        np.sum(predicted[valid_cosine] * target[valid_cosine], axis=1)
        / denominator[valid_cosine]
    )
    magnitude_ratio = np.divide(
        predicted_norm,
        target_norm,
        out=np.full_like(predicted_norm, np.nan),
        where=valid_direction,
    )
    significant_joint = np.abs(target) >= 0.05
    sign_matches = (np.sign(predicted) == np.sign(target)) & significant_joint
    significant_count = int(significant_joint.sum())

    return {
        "sample_count": int(len(predicted)),
        "action_mae": float(np.abs(predicted - target).mean()),
        "mean_cosine": float(np.nanmean(cosine)),
        "median_cosine": float(np.nanmedian(cosine)),
        "negative_cosine_fraction": float(np.nanmean(cosine < 0.0)),
        "mean_predicted_norm": float(predicted_norm.mean()),
        "mean_expert_norm": float(target_norm.mean()),
        "ratio_of_mean_norms": float(
            predicted_norm.mean() / max(target_norm.mean(), 1e-8)
        ),
        "median_per_sample_norm_ratio": float(
            np.nanmedian(magnitude_ratio)
        ),
        "under_half_expert_norm_fraction": float(
            np.nanmean(magnitude_ratio < 0.5)
        ),
        "sign_agreement_for_abs_target_ge_0p05": (
            float(sign_matches.sum() / significant_count)
            if significant_count
            else None
        ),
        "mean_predicted_action": predicted.mean(axis=0).tolist(),
        "mean_expert_action": target.mean(axis=0).tolist(),
        "per_joint_mae": np.abs(predicted - target).mean(axis=0).tolist(),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")

    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = load_bc_checkpoint(checkpoint_path, map_location="cpu")
    actor = restore_actor_from_bc_checkpoint(checkpoint, device=args.device)
    actor.eval()
    datasets, _, _, _ = datasets_from_checkpoint(
        checkpoint=checkpoint,
        action_dim=actor.action_dim,
    )
    dataset = datasets.validation_dataset
    metadata = _validation_metadata(dataset)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=str(args.device).startswith("cuda"),
        persistent_workers=args.num_workers > 0,
    )

    predicted_batches = []
    target_batches = []
    with torch.inference_mode():
        for batch_index, (observation, target_action) in enumerate(loader, start=1):
            observation = {
                key: value.to(args.device, non_blocking=True)
                for key, value in observation.items()
            }
            predicted_batches.append(
                actor.deterministic_action(observation).cpu().numpy()
            )
            target_batches.append(target_action.numpy())
            print(
                f"audit batch={batch_index}/{len(loader)} "
                f"samples={sum(len(batch) for batch in target_batches)}",
                flush=True,
            )

    predicted = np.concatenate(predicted_batches, axis=0)
    target = np.concatenate(target_batches, axis=0)
    if len(predicted) != len(dataset):
        raise RuntimeError(
            f"Prediction count mismatch: {len(predicted)} != {len(dataset)}"
        )

    tag = metadata["tag"]
    xy = metadata["xy_error_mm"]
    z = metadata["z_error_mm"]
    slot_radius = metadata["slot_radius_mm"]
    move_tag = tag == "move_usb_to_play_pre_insert"
    insert_tag = tag == "insert_usb_into_slot"
    groups = {
        "all_validation": np.ones(len(dataset), dtype=bool),
        "move_stage": move_tag,
        "insert_stage": insert_tag,
        "coarse_like_z23_27_xy0p5_3p5": (
            move_tag & (z >= 23.0) & (z <= 27.0) & (xy >= 0.5) & (xy <= 3.5)
        ),
        "legacy_like_z10_14_xy_le2": (
            insert_tag & (z >= 10.0) & (z <= 14.0) & (xy <= 2.0)
        ),
        "move_xy_le0p5": move_tag & (xy <= 0.5),
        "move_xy0p5_1": move_tag & (xy > 0.5) & (xy <= 1.0),
        "move_xy1_2": move_tag & (xy > 1.0) & (xy <= 2.0),
        "move_xy_gt2": move_tag & (xy > 2.0),
        "slot_radius_le15": slot_radius <= 15.0,
        "slot_radius15_30": (slot_radius > 15.0) & (slot_radius <= 30.0),
        "slot_radius_gt30": slot_radius > 30.0,
    }
    report = {
        "checkpoint": str(checkpoint_path),
        "split": "validation",
        "sample_count": len(dataset),
        "definitions": {
            "z_error_mm": (
                "prism_z - slot_z - 8 mm completed-insertion offset"
            ),
            "slot_radius_mm": (
                "target slot XY distance from nominal (0.52 m, 0.0 m)"
            ),
            "action": "normalized relative joint delta",
        },
        "state_summary": {
            "xy_error_mm_mean": float(xy.mean()),
            "z_error_mm_mean": float(z.mean()),
            "slot_radius_mm_mean": float(slot_radius.mean()),
        },
        "groups": {
            name: _group_metrics(predicted, target, mask)
            for name, mask in groups.items()
        },
    }
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(report, output_file, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
