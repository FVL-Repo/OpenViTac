#!/usr/bin/env python3
"""Analyze and select diverse successful RFCL trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class TrajectorySummary:
    path: str
    source_schema: str
    episode: int
    policy_version: int
    demo_index: int
    seed: int
    state_index: int
    raw_state_index: int
    terminal_state_index: int
    normalized_start: float
    length: int
    initial_rel_x_mm: float
    initial_rel_y_mm: float
    initial_rel_z_mm: float
    final_rel_x_mm: float
    final_rel_y_mm: float
    final_rel_z_mm: float
    relative_path_length_mm: float
    force_write_fraction: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize RFCL success diversity and select a balanced subset."
    )
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-demo", type=int, default=32)
    parser.add_argument("--minimum-long-steps", type=int, default=20)
    parser.add_argument("--long-fraction", type=float, default=0.75)
    return parser.parse_args()


def resample(sequence: np.ndarray, count: int) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float64)
    if sequence.ndim != 2 or sequence.shape[0] == 0:
        raise ValueError(f"Expected a non-empty 2D sequence, got {sequence.shape}")
    if sequence.shape[0] == 1:
        return np.repeat(sequence, count, axis=0)
    source = np.linspace(0.0, 1.0, sequence.shape[0])
    target = np.linspace(0.0, 1.0, count)
    return np.stack(
        [np.interp(target, source, sequence[:, column]) for column in range(sequence.shape[1])],
        axis=1,
    )


def load_manifest(path: Path) -> tuple[dict[int, int], dict[int, int]]:
    manifest = json.loads(path.read_text())
    seeds = {}
    terminals = {}
    for demo_index, demo in enumerate(manifest["demos"]):
        seeds[demo_index] = int(demo["seed"])
        policy_start = int(demo.get("policy_start_state_index", 0))
        terminals[demo_index] = int(demo["state_count"]) - policy_start - 1
    return seeds, terminals


def load_trajectory(
    path: Path,
    *,
    seeds: dict[int, int],
    terminals: dict[int, int],
) -> tuple[TrajectorySummary, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        source_schema = str(np.asarray(data["schema"]).item())
        supported_schemas = {
            "rfcl_distributed_training_trajectory_v1",
            "rfcl_privileged_trajectory_v2",
        }
        if source_schema not in supported_schemas:
            raise ValueError(f"Unsupported trajectory schema {source_schema!r}")
        states = np.asarray(data["states"], dtype=np.float64)
        next_states = np.asarray(data["next_states"], dtype=np.float64)
        actions = np.asarray(data["actions"], dtype=np.float64)
        if states.ndim != 2 or states.shape[0] == 0 or states.shape[1] < 46:
            raise ValueError(f"Invalid states in {path}: {states.shape}")
        if next_states.shape != states.shape or actions.shape[0] != states.shape[0]:
            raise ValueError(f"Transition arrays disagree in {path}")
        demo_index = int(data["demo_index"])
        state_index = int(data["state_index"])
        raw_state_index = int(data["raw_state_index"])
        episode = int(data["episode"]) if "episode" in data.files else -1
        policy_version = (
            int(data["policy_version"]) if "policy_version" in data.files else -1
        )

    relative_path = np.concatenate((states[:1, 39:42], next_states[:, 39:42]), axis=0)
    relative_path_mm = relative_path * 1000.0
    path_length_mm = float(
        np.linalg.norm(np.diff(relative_path_mm, axis=0), axis=1).sum()
    )
    terminal_state = terminals[demo_index]
    force_fraction = float(np.mean(actions[:, -1] >= 0.0))
    summary = TrajectorySummary(
        path=str(path),
        source_schema=source_schema,
        episode=episode,
        policy_version=policy_version,
        demo_index=demo_index,
        seed=seeds[demo_index],
        state_index=state_index,
        raw_state_index=raw_state_index,
        terminal_state_index=terminal_state,
        normalized_start=float(state_index / terminal_state),
        length=int(states.shape[0]),
        initial_rel_x_mm=float(relative_path_mm[0, 0]),
        initial_rel_y_mm=float(relative_path_mm[0, 1]),
        initial_rel_z_mm=float(relative_path_mm[0, 2]),
        final_rel_x_mm=float(relative_path_mm[-1, 0]),
        final_rel_y_mm=float(relative_path_mm[-1, 1]),
        final_rel_z_mm=float(relative_path_mm[-1, 2]),
        relative_path_length_mm=path_length_mm,
        force_write_fraction=force_fraction,
    )

    scalar_features = np.asarray(
        [
            summary.normalized_start,
            np.log1p(summary.length),
            summary.relative_path_length_mm,
            summary.force_write_fraction,
        ],
        dtype=np.float64,
    )
    path_features = resample(relative_path_mm, 12).reshape(-1)
    action_features = np.concatenate(
        (
            actions.mean(axis=0),
            actions.std(axis=0),
            np.quantile(actions, 0.1, axis=0),
            np.quantile(actions, 0.9, axis=0),
        )
    )
    features = np.concatenate(
        (
            scalar_features / np.sqrt(scalar_features.size),
            path_features / np.sqrt(path_features.size),
            action_features / np.sqrt(action_features.size),
        )
    )
    return summary, features


def standardized(features: np.ndarray) -> np.ndarray:
    median = np.median(features, axis=0)
    scale = np.quantile(features, 0.75, axis=0) - np.quantile(features, 0.25, axis=0)
    standard_deviation = features.std(axis=0)
    scale = np.where(scale > 1e-8, scale, standard_deviation)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (features - median) / scale


def farthest_point_indices(
    summaries: list[TrajectorySummary], features: np.ndarray, count: int
) -> list[int]:
    if count >= len(summaries):
        return list(range(len(summaries)))
    normalized = standardized(features)
    first = min(
        range(len(summaries)),
        key=lambda index: (summaries[index].normalized_start, -summaries[index].length),
    )
    selected = [first]
    minimum_distance = np.linalg.norm(normalized - normalized[first], axis=1)
    minimum_distance[first] = -np.inf
    while len(selected) < count:
        next_index = int(np.argmax(minimum_distance))
        selected.append(next_index)
        distance = np.linalg.norm(normalized - normalized[next_index], axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[selected] = -np.inf
    return selected


def write_csv(
    path: Path,
    summaries: list[TrajectorySummary],
    selected_paths: set[str],
) -> None:
    rows = []
    for summary in summaries:
        row = asdict(summary)
        row["selected"] = summary.path in selected_paths
        rows.append(row)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_overview(
    output: Path,
    summaries: list[TrajectorySummary],
    selected_paths: set[str],
) -> None:
    seeds = sorted({summary.seed for summary in summaries})
    colors = dict(zip(seeds, plt.cm.tab10(np.linspace(0.0, 1.0, len(seeds)))))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    counts = [sum(summary.seed == seed for summary in summaries) for seed in seeds]
    axes[0, 0].bar([str(seed) for seed in seeds], counts, color=[colors[seed] for seed in seeds])
    axes[0, 0].set(title="Successful trajectories per seed", xlabel="Seed", ylabel="Count")

    for seed in seeds:
        values = [summary.normalized_start for summary in summaries if summary.seed == seed]
        axes[0, 1].hist(values, bins=np.linspace(0.0, 1.0, 21), alpha=0.42, label=str(seed))
    axes[0, 1].set(
        title="Successful start coverage",
        xlabel="Start index / terminal index (lower is earlier)",
        ylabel="Count",
    )
    axes[0, 1].legend(title="Seed", ncol=2, fontsize=8)

    lengths = [summary.length for summary in summaries]
    selected_lengths = [
        summary.length for summary in summaries if summary.path in selected_paths
    ]
    axes[1, 0].hist(lengths, bins=30, alpha=0.55, label="All")
    axes[1, 0].hist(selected_lengths, bins=30, alpha=0.65, label="Selected")
    axes[1, 0].set(title="Trajectory length", xlabel="RL steps", ylabel="Count")
    axes[1, 0].legend()

    for seed in seeds:
        rows = [
            summary
            for summary in summaries
            if summary.seed == seed and summary.path in selected_paths
        ]
        axes[1, 1].scatter(
            [row.initial_rel_x_mm for row in rows],
            [row.initial_rel_y_mm for row in rows],
            s=18,
            alpha=0.75,
            label=str(seed),
            color=colors[seed],
        )
    axes[1, 1].set(
        title="Selected initial USB-in-slot XY",
        xlabel="Relative X (mm)",
        ylabel="Relative Y (mm)",
    )
    axes[1, 1].legend(title="Seed", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "diversity_overview.png", dpi=180)
    plt.close(fig)


def plot_selected_paths(
    output: Path,
    summaries: list[TrajectorySummary],
    selected_paths: set[str],
) -> None:
    seeds = sorted({summary.seed for summary in summaries})
    colors = dict(zip(seeds, plt.cm.tab10(np.linspace(0.0, 1.0, len(seeds)))))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for summary in summaries:
        if summary.path not in selected_paths:
            continue
        with np.load(summary.path, allow_pickle=False) as data:
            states = np.asarray(data["states"], dtype=np.float64)
            next_states = np.asarray(data["next_states"], dtype=np.float64)
        path = np.concatenate((states[:1, 39:42], next_states[:, 39:42]), axis=0) * 1000.0
        axes[0].plot(path[:, 0], path[:, 2], color=colors[summary.seed], alpha=0.16)
        axes[1].plot(path[:, 1], path[:, 2], color=colors[summary.seed], alpha=0.16)
    axes[0].set(title="Selected successful paths: X-Z", xlabel="Relative X (mm)", ylabel="Relative Z (mm)")
    axes[1].set(title="Selected successful paths: Y-Z", xlabel="Relative Y (mm)", ylabel="Relative Z (mm)")
    fig.tight_layout()
    fig.savefig(output / "selected_success_paths.png", dpi=180)
    plt.close(fig)


def build_report(
    summaries: list[TrajectorySummary],
    selected_paths: set[str],
) -> dict[str, object]:
    lengths = np.asarray([summary.length for summary in summaries])
    normalized_starts = np.asarray([summary.normalized_start for summary in summaries])
    selected = [summary for summary in summaries if summary.path in selected_paths]
    selected_lengths = np.asarray([summary.length for summary in selected])
    pairs = {(summary.demo_index, summary.state_index) for summary in summaries}
    per_seed = []
    for seed in sorted({summary.seed for summary in summaries}):
        rows = [summary for summary in summaries if summary.seed == seed]
        per_seed.append(
            {
                "seed": seed,
                "count": len(rows),
                "selected": sum(row.path in selected_paths for row in rows),
                "earliest_state": min(row.state_index for row in rows),
                "latest_state": max(row.state_index for row in rows),
                "unique_start_states": len({row.state_index for row in rows}),
                "mean_length": float(np.mean([row.length for row in rows])),
                "max_length": max(row.length for row in rows),
            }
        )
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "trajectory_count": len(summaries),
        "selected_count": len(selected_paths),
        "unique_demo_state_pairs": len(pairs),
        "length_ge_20_fraction": float(np.mean(lengths >= 20)),
        "length_ge_30_fraction": float(np.mean(lengths >= 30)),
        "length_ge_50_fraction": float(np.mean(lengths >= 50)),
        "selected_length_ge_20_fraction": float(np.mean(selected_lengths >= 20)),
        "selected_length_ge_30_fraction": float(np.mean(selected_lengths >= 30)),
        "start_first_quarter_fraction": float(np.mean(normalized_starts <= 0.25)),
        "start_first_half_fraction": float(np.mean(normalized_starts <= 0.5)),
        "per_seed": per_seed,
    }


def write_markdown(path: Path, report: dict[str, object]) -> None:
    lines = [
        "# RFCL Diversity First-Pass Report",
        "",
        f"Generated: {report['generated_at']}",
        "",
        f"- Successful trajectories: {report['trajectory_count']}",
        f"- Diverse balanced candidates: {report['selected_count']}",
        f"- Unique `(demo, start state)` pairs: {report['unique_demo_state_pairs']}",
        f"- Trajectories with at least 30 RL steps: {100 * report['length_ge_30_fraction']:.1f}%",
        f"- Selected candidates with at least 20 RL steps: {100 * report['selected_length_ge_20_fraction']:.1f}%",
        f"- Starts in the first half of each suffix: {100 * report['start_first_half_fraction']:.1f}%",
        f"- Starts in the first quarter of each suffix: {100 * report['start_first_quarter_fraction']:.1f}%",
        "",
        "| Seed | Successes | Selected | Earliest state | Unique starts | Mean length | Max length |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["per_seed"]:
        lines.append(
            "| {seed} | {count} | {selected} | {earliest_state} | "
            "{unique_start_states} | {mean_length:.1f} | {max_length} |".format(**row)
        )
    lines.extend(
        (
            "",
            "## Interpretation",
            "",
            "This subset is suitable for a first behavior-diversity review and privileged-state replay.",
            "It is not yet a final multimodal BC dataset: the saved files contain no RGB or tactile frames,",
            "and the curriculum has not produced a successful state-0 rollout.",
            "",
            "See `diversity_overview.png`, `selected_success_paths.png`, and `trajectories.csv`.",
        )
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    if args.per_demo <= 0:
        raise ValueError("--per-demo must be positive")
    if args.minimum_long_steps <= 0:
        raise ValueError("--minimum-long-steps must be positive")
    if not 0.0 <= args.long_fraction <= 1.0:
        raise ValueError("--long-fraction must be in [0, 1]")
    args.output.mkdir(parents=True, exist_ok=True)
    seeds, terminals = load_manifest(args.manifest)
    summaries = []
    features = []
    skipped = []
    for path in sorted(args.trajectory_dir.glob("*.npz")):
        try:
            summary, feature = load_trajectory(path, seeds=seeds, terminals=terminals)
        except (OSError, ValueError, KeyError) as error:
            skipped.append({"path": str(path), "error": repr(error)})
            continue
        summaries.append(summary)
        features.append(feature)
    if not summaries:
        raise RuntimeError(f"No valid trajectories found in {args.trajectory_dir}")

    feature_array = np.stack(features)
    selected_indices = []
    for demo_index in sorted({summary.demo_index for summary in summaries}):
        group = [index for index, summary in enumerate(summaries) if summary.demo_index == demo_index]
        target = min(args.per_demo, len(group))
        long_target = min(round(target * args.long_fraction), target)
        long_group = [
            index for index in group if summaries[index].length >= args.minimum_long_steps
        ]
        short_group = [
            index for index in group if summaries[index].length < args.minimum_long_steps
        ]
        long_count = min(long_target, len(long_group))
        short_count = min(target - long_count, len(short_group))
        remaining = target - long_count - short_count
        if remaining > 0:
            extra_long = min(remaining, len(long_group) - long_count)
            long_count += extra_long
            remaining -= extra_long
        if remaining > 0:
            short_count += min(remaining, len(short_group) - short_count)
        for subgroup, count in ((long_group, long_count), (short_group, short_count)):
            if count <= 0:
                continue
            local = farthest_point_indices(
                [summaries[index] for index in subgroup],
                feature_array[subgroup],
                count,
            )
            selected_indices.extend(subgroup[index] for index in local)
    selected_paths = {summaries[index].path for index in selected_indices}

    write_csv(args.output / "trajectories.csv", summaries, selected_paths)
    (args.output / "selected_trajectories.txt").write_text(
        "\n".join(sorted(selected_paths)) + "\n"
    )
    plot_overview(args.output, summaries, selected_paths)
    plot_selected_paths(args.output, summaries, selected_paths)
    report = build_report(summaries, selected_paths)
    report["skipped"] = skipped
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(args.output / "REPORT.md", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
