from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRIC_FIELDS = (
    "frame",
    "reference_geometry_valid",
    "raw_3d_p50_mm",
    "raw_3d_p95_mm",
    "registered_3d_p50_mm",
    "registered_3d_p95_mm",
    "registered_dx_median_mm",
    "registered_dy_median_mm",
    "registered_dz_median_mm",
    "registered_dx_abs_p95_mm",
    "registered_dy_abs_p95_mm",
    "registered_dz_abs_p95_mm",
    "init_depth_median_mm",
    "registered_depth_median_mm",
    "raw_2d_p50_px",
    "raw_2d_p95_px",
    "registered_2d_p50_px",
    "registered_2d_p95_px",
    "constraint_3d_p50_mm",
    "constraint_3d_p95_mm",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize opt-in XSense FEM marker layer diagnostics."
    )
    parser.add_argument("layer_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--valid-reference-only",
        action="store_true",
        help="Exclude reset transients outside the physical XSense camera range.",
    )
    return parser.parse_args()


def percentile_metrics(
    displacement: np.ndarray,
    prefix: str,
    unit: str,
    scale: float,
) -> dict:
    norm = np.linalg.norm(displacement, axis=-1) * scale
    return {
        f"{prefix}_p50_{unit}": float(np.percentile(norm, 50)),
        f"{prefix}_p95_{unit}": float(np.percentile(norm, 95)),
    }


def sensor_label(path: Path, prim_path: str) -> str:
    text = f"{path.parent.name} {prim_path}".lower()
    if "left" in text:
        return "left"
    if "right" in text:
        return "right"
    return path.parent.name


def component_metrics(
    init_xyz: np.ndarray,
    current_xyz: np.ndarray,
) -> dict[str, float]:
    displacement_mm = (current_xyz - init_xyz) * 1000.0
    axis_names = ("x", "y", "z")
    metrics = {
        f"registered_d{axis}_median_mm": float(np.median(displacement_mm[:, index]))
        for index, axis in enumerate(axis_names)
    }
    metrics.update(
        {
            f"registered_d{axis}_abs_p95_mm": float(
                np.percentile(np.abs(displacement_mm[:, index]), 95)
            )
            for index, axis in enumerate(axis_names)
        }
    )
    metrics["init_depth_median_mm"] = float(np.median(init_xyz[:, 2]) * 1000.0)
    metrics["registered_depth_median_mm"] = float(
        np.median(current_xyz[:, 2]) * 1000.0
    )
    return metrics


def load_rows(layer_dir: Path) -> tuple[dict[str, list[dict]], dict[str, list[Path]]]:
    grouped_rows: dict[str, list[dict]] = {}
    grouped_paths: dict[str, list[Path]] = {}
    for path in sorted(layer_dir.rglob("frame_*.npz")):
        with np.load(path) as sample:
            prim_path = str(sample["prim_path"].item())
            label = sensor_label(path, prim_path)
            init_xyz = sample["init_marker_xyz"]
            raw_xyz = sample["raw_marker_xyz"]
            registered_xyz = sample["registered_marker_xyz"]
            init_uv = sample["init_marker_uv"]
            raw_uv = sample["raw_marker_uv"]
            registered_uv = sample["registered_marker_uv"]
            reference_constrain = sample["reference_constrain_xyz"]
            current_constrain = sample["current_constrain_xyz"]
            surface_depth = float(np.median(raw_xyz[:, 2]))
            constrain_depth = float(np.median(current_constrain[:, 2]))
            reference_geometry_valid = bool(
                np.isfinite(raw_xyz).all()
                and np.isfinite(current_constrain).all()
                and 0.020 <= surface_depth <= 0.035
                and 0.015 <= constrain_depth <= 0.030
                and float(np.abs(raw_xyz[:, :2]).max()) <= 0.050
                and float(np.abs(current_constrain[:, :2]).max()) <= 0.050
            )
            row = {
                "frame": int(sample["frame_index"]),
                "reference_geometry_valid": reference_geometry_valid,
                **percentile_metrics(raw_xyz - init_xyz, "raw_3d", "mm", 1000.0),
                **percentile_metrics(
                    registered_xyz - init_xyz,
                    "registered_3d",
                    "mm",
                    1000.0,
                ),
                **component_metrics(init_xyz, registered_xyz),
                **percentile_metrics(raw_uv - init_uv, "raw_2d", "px", 1.0),
                **percentile_metrics(
                    registered_uv - init_uv,
                    "registered_2d",
                    "px",
                    1.0,
                ),
                **percentile_metrics(
                    current_constrain - reference_constrain,
                    "constraint_3d",
                    "mm",
                    1000.0,
                ),
            }
        grouped_rows.setdefault(label, []).append(row)
        grouped_paths.setdefault(label, []).append(path)

    if not grouped_rows:
        raise FileNotFoundError(f"No frame_*.npz files under {layer_dir}")
    for label in grouped_rows:
        order = np.argsort([row["frame"] for row in grouped_rows[label]])
        grouped_rows[label] = [grouped_rows[label][index] for index in order]
        grouped_paths[label] = [grouped_paths[label][index] for index in order]
    return grouped_rows, grouped_paths


def write_metrics(output_dir: Path, task: str, grouped_rows: dict[str, list[dict]]) -> dict:
    fields = ("task", "side", *METRIC_FIELDS)
    with (output_dir / "fem_layer_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        for side, rows in grouped_rows.items():
            for row in rows:
                writer.writerow({"task": task, "side": side, **row})

    summary = {"task": task, "sides": {}}
    for side, rows in grouped_rows.items():
        peak_index = int(np.argmax([row["registered_2d_p95_px"] for row in rows]))
        peak = rows[peak_index]
        summary["sides"][side] = {
            "samples": len(rows),
            "peak_sample_index": peak_index,
            "peak_frame": peak["frame"],
            "peak": peak,
            "max_raw_3d_p95_mm": max(row["raw_3d_p95_mm"] for row in rows),
            "max_registered_3d_p95_mm": max(
                row["registered_3d_p95_mm"] for row in rows
            ),
            "max_raw_2d_p95_px": max(row["raw_2d_p95_px"] for row in rows),
            "max_registered_2d_p95_px": max(
                row["registered_2d_p95_px"] for row in rows
            ),
        }
    (output_dir / "fem_layer_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def plot_timeseries(
    output_dir: Path,
    task: str,
    grouped_rows: dict[str, list[dict]],
) -> None:
    sides = sorted(grouped_rows, key=lambda side: (side != "left", side != "right", side))
    figure, axes = plt.subplots(
        len(sides),
        2,
        figsize=(13, max(4.2, 3.6 * len(sides))),
        squeeze=False,
    )
    for row_index, side in enumerate(sides):
        rows = grouped_rows[side]
        frames = [row["frame"] for row in rows]
        axes[row_index, 0].plot(
            frames,
            [row["raw_3d_p95_mm"] for row in rows],
            label="raw FEM",
            color="#d95f02",
            linewidth=1.6,
        )
        axes[row_index, 0].plot(
            frames,
            [row["registered_3d_p95_mm"] for row in rows],
            label="registered FEM",
            color="#1b9e77",
            linewidth=1.6,
        )
        axes[row_index, 0].set_ylabel(f"{side}: displacement p95 (mm)")
        axes[row_index, 0].legend(loc="upper left")
        axes[row_index, 1].plot(
            frames,
            [row["raw_2d_p95_px"] for row in rows],
            label="raw projection",
            color="#7570b3",
            linewidth=1.6,
        )
        axes[row_index, 1].plot(
            frames,
            [row["registered_2d_p95_px"] for row in rows],
            label="FEM output",
            color="#e7298a",
            linewidth=1.6,
        )
        axes[row_index, 1].set_ylabel(f"{side}: displacement p95 (px)")
        axes[row_index, 1].legend(loc="upper left")
        for axis in axes[row_index]:
            axis.set_xlabel("simulation frame")
            axis.grid(alpha=0.22)
    figure.suptitle(f"{task}: XSense FEM marker layers")
    figure.tight_layout()
    figure.savefig(output_dir / "fem_layer_comparison.png", dpi=180)
    plt.close(figure)


def draw_quiver(axis, init_uv: np.ndarray, current_uv: np.ndarray, title: str) -> None:
    displacement = current_uv - init_uv
    magnitude = np.linalg.norm(displacement, axis=-1)
    axis.scatter(init_uv[:, 0], init_uv[:, 1], s=5, color="#404040", alpha=0.5)
    axis.quiver(
        init_uv[:, 0],
        init_uv[:, 1],
        displacement[:, 0],
        displacement[:, 1],
        magnitude,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.003,
        cmap="viridis",
    )
    axis.set_title(title)
    axis.set_aspect("equal", adjustable="box")
    axis.invert_yaxis()
    axis.grid(alpha=0.15)


def plot_peak_quivers(
    output_dir: Path,
    task: str,
    grouped_rows: dict[str, list[dict]],
    grouped_paths: dict[str, list[Path]],
) -> None:
    sides = sorted(grouped_rows, key=lambda side: (side != "left", side != "right", side))
    figure, axes = plt.subplots(
        len(sides),
        2,
        figsize=(11, max(4.5, 4.2 * len(sides))),
        squeeze=False,
    )
    for row_index, side in enumerate(sides):
        rows = grouped_rows[side]
        peak_index = int(np.argmax([row["registered_2d_p95_px"] for row in rows]))
        with np.load(grouped_paths[side][peak_index]) as sample:
            init_uv = sample["init_marker_uv"]
            raw_uv = sample["raw_marker_uv"]
            registered_uv = sample["registered_marker_uv"]
        frame = rows[peak_index]["frame"]
        draw_quiver(axes[row_index, 0], init_uv, raw_uv, f"{side} raw projection, frame {frame}")
        draw_quiver(
            axes[row_index, 1],
            init_uv,
            registered_uv,
            f"{side} FEM output, frame {frame}",
        )
    figure.suptitle(f"{task}: peak FEM marker displacement")
    figure.tight_layout()
    figure.savefig(output_dir / "fem_layer_peak_quiver.png", dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    grouped_rows, grouped_paths = load_rows(args.layer_dir)
    if args.valid_reference_only:
        for side in list(grouped_rows):
            retained = [
                (row, path)
                for row, path in zip(grouped_rows[side], grouped_paths[side])
                if row["reference_geometry_valid"]
            ]
            if not retained:
                raise RuntimeError(
                    f"No physically valid XSense reference frames for {side}"
                )
            grouped_rows[side] = [row for row, _ in retained]
            grouped_paths[side] = [path for _, path in retained]
    summary = write_metrics(args.output_dir, args.task, grouped_rows)
    plot_timeseries(args.output_dir, args.task, grouped_rows)
    plot_peak_quivers(args.output_dir, args.task, grouped_rows, grouped_paths)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
