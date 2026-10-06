#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


SIDES = ("left_tactile", "right_tactile")
SIDE_LABELS = {"left_tactile": "LEFT", "right_tactile": "RIGHT"}
BACKGROUND = (12, 21, 35)
PANEL = (18, 32, 51)
TEXT = (232, 238, 246)
MUTED = (160, 174, 192)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit XSense marker deformation across multiple task rollouts."
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "tasks",
        nargs="+",
        help="Task/HDF5 pairs in the form task_name=/path/to/episode.hdf5",
    )
    parser.add_argument("--tile-width", type=int, default=180)
    parser.add_argument("--tile-height", type=int, default=315)
    parser.add_argument("--numeric-only", action="store_true")
    parser.add_argument(
        "--min-step",
        type=int,
        default=None,
        help="Only audit saved frames at or after this simulation step.",
    )
    return parser.parse_args()


def decode_image(raw) -> np.ndarray:
    encoded = np.frombuffer(bytes(raw), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode rgb_marker JPEG")
    return image


def detect_rgb_marker_points(image: np.ndarray) -> np.ndarray:
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = (grayscale < 55).astype(np.uint8)
    component_count, _, statistics, centroids = cv2.connectedComponentsWithStats(
        binary,
        8,
    )
    points = []
    for component_index in range(1, component_count):
        _, _, width, height, area = statistics[component_index]
        if 2 <= area <= 90 and width <= 12 and height <= 12:
            points.append(centroids[component_index])
    points = np.asarray(points, dtype=np.float32)
    return points


def rgb_marker_points(image: np.ndarray) -> np.ndarray:
    points = detect_rgb_marker_points(image)
    if len(points) != 220:
        raise ValueError(f"Expected 220 rgb_marker dots, found {len(points)}")
    points_by_y = points[np.argsort(points[:, 1])].reshape(20, 11, 2)
    ordered_rows = [row[np.argsort(row[:, 0])] for row in points_by_y]
    return np.concatenate(ordered_rows, axis=0)


def rgb_numeric_alignment_metrics(
    image: np.ndarray,
    numeric_points: np.ndarray,
) -> dict[str, float | int]:
    observed_points = detect_rgb_marker_points(image)
    numeric_points = np.asarray(numeric_points, dtype=np.float32)
    in_bounds = (
        (numeric_points[:, 0] >= 0)
        & (numeric_points[:, 0] < 400)
        & (numeric_points[:, 1] >= 0)
        & (numeric_points[:, 1] < 700)
    )
    visible_numeric_points = numeric_points[in_bounds]
    distances = np.linalg.norm(
        observed_points[:, None, :] - visible_numeric_points[None, :, :],
        axis=-1,
    )
    return {
        "rgb_dot_count": int(len(observed_points)),
        "numeric_in_bounds_count": int(len(visible_numeric_points)),
        "rgb_to_numeric_p95_px": float(
            np.percentile(distances.min(axis=1), 95)
        ),
        "numeric_to_rgb_p95_px": float(
            np.percentile(distances.min(axis=0), 95)
        ),
    }


def rgb_displacement_metrics(
    baseline_points: np.ndarray,
    current_points: np.ndarray,
) -> dict[str, float]:
    displacement = current_points - baseline_points
    global_shift = np.median(displacement, axis=0)
    residual = displacement - global_shift
    displacement_norm = np.linalg.norm(displacement, axis=-1)
    residual_norm = np.linalg.norm(residual, axis=-1)
    return {
        "rgb_disp_p50_px": float(np.percentile(displacement_norm, 50)),
        "rgb_disp_p95_px": float(np.percentile(displacement_norm, 95)),
        "rgb_residual_p95_px": float(np.percentile(residual_norm, 95)),
        "rgb_global_shift_x_px": float(global_shift[0]),
        "rgb_global_shift_y_px": float(global_shift[1]),
    }


def rgb_flow_image(
    baseline_image: np.ndarray,
    baseline_points: np.ndarray,
    current_points: np.ndarray,
) -> np.ndarray:
    canvas = baseline_image.copy()
    displacement = current_points - baseline_points
    magnitude = np.linalg.norm(displacement, axis=-1)
    displacement_p95 = float(np.percentile(magnitude, 95))
    arrow_scale = 6.0 if displacement_p95 < 3 else 4.0 if displacement_p95 < 8 else 2.5
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (400, 700), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
    for position, vector, vector_magnitude in zip(
        baseline_points,
        displacement,
        magnitude,
    ):
        start = (int(round(position[0])), int(round(position[1])))
        end = (
            int(round(position[0] + vector[0] * arrow_scale)),
            int(round(position[1] + vector[1] * arrow_scale)),
        )
        if vector_magnitude >= displacement_p95:
            color = (35, 35, 220)
        elif vector_magnitude >= 0.75:
            color = (0, 130, 245)
        else:
            color = (55, 55, 55)
        cv2.circle(canvas, start, 2, (30, 30, 30), -1, cv2.LINE_AA)
        cv2.arrowedLine(
            canvas,
            start,
            end,
            color,
            1,
            cv2.LINE_AA,
            tipLength=0.22,
        )
    return canvas


def marker_displacement(marker: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference = np.asarray(marker[0], dtype=np.float32)
    current = np.asarray(marker[1], dtype=np.float32)
    displacement = current - reference
    global_shift = np.nanmedian(displacement, axis=0)
    residual = displacement - global_shift
    return reference, global_shift, residual


def marker_metrics(marker: np.ndarray) -> dict[str, float]:
    reference, global_shift, residual = marker_displacement(marker)
    residual_norm = np.linalg.norm(residual, axis=-1)
    finite_reference = np.all(np.isfinite(reference), axis=1)
    in_bounds = (
        finite_reference
        & (reference[:, 0] >= 0)
        & (reference[:, 0] < 400)
        & (reference[:, 1] >= 0)
        & (reference[:, 1] < 700)
    )
    metrics = {
        "residual_p50_px": float(np.nanpercentile(residual_norm, 50)),
        "residual_p95_px": float(np.nanpercentile(residual_norm, 95)),
        "residual_max_px": float(np.nanmax(residual_norm)),
        "global_shift_x_px": float(global_shift[0]),
        "global_shift_y_px": float(global_shift[1]),
        "reference_out_of_bounds_fraction": float(1.0 - np.mean(in_bounds)),
    }
    if residual.shape[0] == 220:
        grid = residual.reshape(11, 20, 2)
        horizontal_jumps = np.linalg.norm(
            grid[:, 1:, :] - grid[:, :-1, :], axis=-1
        ).ravel()
        vertical_jumps = np.linalg.norm(
            grid[1:, :, :] - grid[:-1, :, :], axis=-1
        ).ravel()
        metrics["neighbor_jump_p95_px"] = float(
            np.nanpercentile(
                np.concatenate((horizontal_jumps, vertical_jumps)),
                95,
            )
        )
    else:
        metrics["neighbor_jump_p95_px"] = float("nan")
    return metrics


def title_tile(
    image: np.ndarray,
    title: str,
    subtitle: str,
    size: tuple[int, int],
) -> np.ndarray:
    tile_width, tile_height = size
    resized = cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
    canvas = np.full((tile_height + 52, tile_width, 3), PANEL, dtype=np.uint8)
    canvas[52:] = resized
    cv2.putText(
        canvas,
        title,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        TEXT,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        subtitle,
        (8, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        MUTED,
        1,
        cv2.LINE_AA,
    )
    return canvas


def flow_image(marker: np.ndarray) -> np.ndarray:
    canvas = np.full((700, 400, 3), 245, dtype=np.uint8)
    reference, _, residual = marker_displacement(marker)
    magnitude = np.linalg.norm(residual, axis=-1)
    residual_p95 = float(np.nanpercentile(magnitude, 95))
    arrow_scale = 3.0 if residual_p95 < 10 else 2.0 if residual_p95 < 30 else 1.0
    for position, vector, vector_magnitude in zip(reference, residual, magnitude):
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(vector)):
            continue
        start = (int(round(position[0])), int(round(position[1])))
        if not 0 <= start[0] < 400 or not 0 <= start[1] < 700:
            continue
        end = (
            int(round(position[0] + vector[0] * arrow_scale)),
            int(round(position[1] + vector[1] * arrow_scale)),
        )
        if vector_magnitude >= residual_p95:
            color = (35, 35, 220)
        elif vector_magnitude >= 1.0:
            color = (0, 130, 245)
        else:
            color = (55, 55, 55)
        cv2.circle(canvas, start, 2, (30, 30, 30), -1, cv2.LINE_AA)
        cv2.arrowedLine(
            canvas,
            start,
            end,
            color,
            1,
            cv2.LINE_AA,
            tipLength=0.22,
        )
    return canvas


def reference_drift_px(
    marker_dataset: h5py.Dataset,
    frame_indices: np.ndarray,
) -> float:
    initial_reference = np.asarray(marker_dataset[frame_indices[0]][0], dtype=np.float32)
    max_drift = 0.0
    for frame_index in frame_indices:
        reference = np.asarray(marker_dataset[frame_index][0], dtype=np.float32)
        drift = np.linalg.norm(reference - initial_reference, axis=-1)
        max_drift = max(max_drift, float(np.nanmax(drift)))
    return max_drift


def temporal_jump_px(
    marker_dataset: h5py.Dataset,
    frame_indices: np.ndarray,
) -> float:
    previous_current = np.asarray(marker_dataset[frame_indices[0]][1], dtype=np.float32)
    max_jump = 0.0
    for frame_index in frame_indices[1:]:
        current = np.asarray(marker_dataset[frame_index][1], dtype=np.float32)
        jump = np.linalg.norm(current - previous_current, axis=-1)
        max_jump = max(max_jump, float(np.nanpercentile(jump, 95)))
        previous_current = current
    return max_jump


def audit_task(
    task_name: str,
    hdf5_path: Path,
    tile_size: tuple[int, int],
    numeric_only: bool,
    min_step: int | None,
) -> tuple[np.ndarray, list[dict]]:
    rows = []
    tiles = []
    with h5py.File(hdf5_path, "r") as hdf5_file:
        steps = np.asarray(hdf5_file["step"])
        frame_indices = np.arange(len(steps), dtype=np.int64)
        if min_step is not None:
            frame_indices = frame_indices[steps >= min_step]
        if not len(frame_indices):
            raise ValueError(
                f"No saved frames at or after step {min_step} in {hdf5_path}"
            )
        frame_count = len(frame_indices)
        for side in SIDES:
            prefix = f"tactile/{side}"
            marker_dataset = hdf5_file[f"{prefix}/marker"]
            rgb_marker_dataset = hdf5_file[f"{prefix}/rgb_marker"]
            frame_metrics = [
                marker_metrics(np.asarray(marker_dataset[frame_index]))
                for frame_index in frame_indices
            ]
            numeric_peak_offset = int(
                np.argmax([item["residual_p95_px"] for item in frame_metrics])
            )
            numeric_peak_frame = int(frame_indices[numeric_peak_offset])
            numeric_peak_metrics = frame_metrics[numeric_peak_offset]
            rgb_images = [
                decode_image(rgb_marker_dataset[frame_index])
                for frame_index in frame_indices
            ]
            baseline_frame = int(frame_indices[0])
            baseline_image = rgb_images[0]
            alignment_metrics = [
                rgb_numeric_alignment_metrics(
                    rgb_images[offset],
                    np.asarray(marker_dataset[frame_index][1]),
                )
                for offset, frame_index in enumerate(frame_indices)
            ]
            side_label = SIDE_LABELS[side]
            if numeric_only:
                rgb_peak_frame = numeric_peak_frame
                rgb_peak_metrics = {
                    "rgb_disp_p50_px": float("nan"),
                    "rgb_disp_p95_px": float("nan"),
                    "rgb_residual_p95_px": float("nan"),
                    "rgb_global_shift_x_px": float("nan"),
                    "rgb_global_shift_y_px": float("nan"),
                }
                temporal_jumps = []
                peak_image = rgb_images[numeric_peak_offset]
                flow = flow_image(np.asarray(marker_dataset[numeric_peak_frame]))
                peak_subtitle = (
                    f"frame {numeric_peak_frame} | p95 "
                    f"{numeric_peak_metrics['residual_p95_px']:.2f}px"
                )
                flow_title = f"{side_label} numeric flow"
                flow_subtitle = (
                    f"global ({numeric_peak_metrics['global_shift_x_px']:.1f}, "
                    f"{numeric_peak_metrics['global_shift_y_px']:.1f})px"
                )
            else:
                baseline_points = rgb_marker_points(baseline_image)
                rgb_points = [rgb_marker_points(image) for image in rgb_images]
                rgb_metrics = [
                    rgb_displacement_metrics(baseline_points, points)
                    for points in rgb_points
                ]
                rgb_peak_offset = int(
                    np.argmax([item["rgb_disp_p95_px"] for item in rgb_metrics])
                )
                rgb_peak_frame = int(frame_indices[rgb_peak_offset])
                rgb_peak_metrics = rgb_metrics[rgb_peak_offset]
                peak_image = rgb_images[rgb_peak_offset]
                peak_points = rgb_points[rgb_peak_offset]
                temporal_jumps = [
                    float(
                        np.percentile(
                            np.linalg.norm(
                                rgb_points[frame_index]
                                - rgb_points[frame_index - 1],
                                axis=-1,
                            ),
                            95,
                        )
                    )
                    for frame_index in range(1, frame_count)
                ]
                flow = rgb_flow_image(
                    baseline_image,
                    baseline_points,
                    peak_points,
                )
                peak_subtitle = (
                    f"frame {rgb_peak_frame} | p95 "
                    f"{rgb_peak_metrics['rgb_disp_p95_px']:.2f}px"
                )
                flow_title = f"{side_label} rgb flow"
                flow_subtitle = (
                    f"global ({rgb_peak_metrics['rgb_global_shift_x_px']:.1f}, "
                    f"{rgb_peak_metrics['rgb_global_shift_y_px']:.1f})px"
                )
            tiles.extend(
                (
                    title_tile(
                        baseline_image,
                        f"{side_label} baseline",
                        f"frame {baseline_frame} | step {int(steps[baseline_frame])}",
                        tile_size,
                    ),
                    title_tile(
                        peak_image,
                        f"{side_label} peak",
                        peak_subtitle,
                        tile_size,
                    ),
                    title_tile(
                        flow,
                        flow_title,
                        flow_subtitle,
                        tile_size,
                    ),
                )
            )
            rows.append(
                {
                    "task": task_name,
                    "hdf5": str(hdf5_path),
                    "side": side,
                    "frame_count": frame_count,
                    "first_frame": baseline_frame,
                    "first_step": int(steps[baseline_frame]),
                    "rgb_peak_frame": rgb_peak_frame,
                    "rgb_temporal_jump_p95_max_px": max(temporal_jumps, default=0.0),
                    **rgb_peak_metrics,
                    "numeric_peak_frame": numeric_peak_frame,
                    "numeric_residual_p95_max_px": numeric_peak_metrics[
                        "residual_p95_px"
                    ],
                    "numeric_global_shift_x_at_peak_px": numeric_peak_metrics[
                        "global_shift_x_px"
                    ],
                    "numeric_global_shift_y_at_peak_px": numeric_peak_metrics[
                        "global_shift_y_px"
                    ],
                    "rgb_dot_count_min": min(
                        item["rgb_dot_count"] for item in alignment_metrics
                    ),
                    "rgb_dot_count_max": max(
                        item["rgb_dot_count"] for item in alignment_metrics
                    ),
                    "rgb_to_numeric_p95_max_px": max(
                        item["rgb_to_numeric_p95_px"]
                        for item in alignment_metrics
                    ),
                    "numeric_to_rgb_p95_max_px": max(
                        item["numeric_to_rgb_p95_px"]
                        for item in alignment_metrics
                    ),
                    "reference_drift_max_px": reference_drift_px(
                        marker_dataset, frame_indices
                    ),
                    "temporal_jump_p95_max_px": temporal_jump_px(
                        marker_dataset, frame_indices
                    ),
                }
            )
    task_row = np.concatenate(tiles, axis=1)
    label_width = 260
    label = np.full((task_row.shape[0], label_width, 3), BACKGROUND, dtype=np.uint8)
    wrapped_name = task_name.replace("_", " ")
    words = wrapped_name.split()
    lines = []
    current_line = ""
    for word in words:
        candidate = f"{current_line} {word}".strip()
        if len(candidate) > 22 and current_line:
            lines.append(current_line)
            current_line = word
        else:
            current_line = candidate
    if current_line:
        lines.append(current_line)
    start_y = max(42, task_row.shape[0] // 2 - 18 * len(lines))
    for line_index, line in enumerate(lines):
        cv2.putText(
            label,
            line,
            (16, start_y + line_index * 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            TEXT,
            1,
            cv2.LINE_AA,
        )
    return np.concatenate((label, task_row), axis=1), rows


def main() -> None:
    args = parse_args()
    task_paths = []
    for item in args.tasks:
        if "=" not in item:
            raise ValueError(f"Invalid task mapping: {item}")
        task_name, path = item.split("=", 1)
        task_paths.append((task_name, Path(path)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_rows = []
    metrics_rows = []
    tile_size = (args.tile_width, args.tile_height)
    for task_name, hdf5_path in task_paths:
        task_row, task_metrics = audit_task(
            task_name,
            hdf5_path,
            tile_size,
            args.numeric_only,
            args.min_step,
        )
        task_rows.append(task_row)
        metrics_rows.extend(task_metrics)
    gap = np.full((8, task_rows[0].shape[1], 3), BACKGROUND, dtype=np.uint8)
    overview_rows = []
    for row_index, task_row in enumerate(task_rows):
        if row_index:
            overview_rows.append(gap)
        overview_rows.append(task_row)
    overview = np.concatenate(overview_rows, axis=0)
    overview_name = (
        "xense_marker_8tasks_numeric_overview.png"
        if args.numeric_only
        else "xense_marker_8tasks_rgb_overview.png"
    )
    cv2.imwrite(str(args.output_dir / overview_name), overview)
    with (args.output_dir / "xense_marker_8tasks_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(metrics_rows[0]))
        writer.writeheader()
        writer.writerows(metrics_rows)
    summary = {
        "tasks": len(task_paths),
        "sides": len(metrics_rows),
        "numeric_only": args.numeric_only,
        "max_reference_drift_px": max(
            item["reference_drift_max_px"] for item in metrics_rows
        ),
        "max_rgb_numeric_alignment_p95_px": max(
            max(
                item["rgb_to_numeric_p95_max_px"],
                item["numeric_to_rgb_p95_max_px"],
            )
            for item in metrics_rows
        ),
        "max_temporal_jump_p95_px": max(
            item["temporal_jump_p95_max_px"] for item in metrics_rows
        ),
        "max_rgb_temporal_jump_p95_px": (
            None
            if args.numeric_only
            else max(item["rgb_temporal_jump_p95_max_px"] for item in metrics_rows)
        ),
        "peak_rgb_displacement_p95_px": (
            None
            if args.numeric_only
            else {
                f"{item['task']}/{item['side']}": item["rgb_disp_p95_px"]
                for item in metrics_rows
            }
        ),
        "peak_numeric_residual_p95_px": {
            f"{item['task']}/{item['side']}": item[
                "numeric_residual_p95_max_px"
            ]
            for item in metrics_rows
        },
    }
    (args.output_dir / "xense_marker_8tasks_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
