import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


SIDES = ("left_tactile", "right_tactile")
SURFACE_DEPTH_MM = 28.0
CONTACT_THRESHOLD_MM = 0.05


def decode_image(dataset, frame_index):
    encoded = np.frombuffer(dataset[frame_index], dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not decode frame {frame_index} from {dataset.name}")
    return image


def finite_correlation(first, second):
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    valid = np.isfinite(first) & np.isfinite(second)
    first = first[valid]
    second = second[valid]
    if first.size < 3 or np.std(first) < 1e-12 or np.std(second) < 1e-12:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def percentile(values, quantile):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.percentile(values, quantile))


def sample_at_markers(image, marker_positions):
    height, width = image.shape[:2]
    x = np.clip(np.rint(marker_positions[:, 0]).astype(np.int64), 0, width - 1)
    y = np.clip(np.rint(marker_positions[:, 1]).astype(np.int64), 0, height - 1)
    return image[y, x]


def indentation_frame(file_handle, side, frame_index, depth=None):
    dataset_path = f"tactile/{side}/indentation"
    if dataset_path in file_handle:
        return np.asarray(file_handle[dataset_path][frame_index])
    if depth is None:
        depth = np.asarray(file_handle[f"tactile/{side}/depth"][frame_index])
    return np.maximum(SURFACE_DEPTH_MM - depth, 0.0)


def audit_side(file_handle, side):
    depth_dataset = file_handle[f"tactile/{side}/depth"]
    marker = np.asarray(file_handle[f"tactile/{side}/marker"])
    frame_count = marker.shape[0]
    displacement = np.linalg.norm(marker[:, 1] - marker[:, 0], axis=-1)
    marker_p95 = np.percentile(displacement, 95, axis=1)
    marker_max = displacement.max(axis=1)

    depth_min = np.empty(frame_count, dtype=np.float32)
    contact_fraction = np.empty(frame_count, dtype=np.float32)
    contact_strength = np.empty(frame_count, dtype=np.float32)
    marker_contact_correlations = []

    for frame_index in range(frame_count):
        depth = np.asarray(depth_dataset[frame_index])
        indentation = indentation_frame(file_handle, side, frame_index, depth)
        depth_min[frame_index] = float(depth.min())
        contact_fraction[frame_index] = float(np.mean(indentation > CONTACT_THRESHOLD_MM))
        contact_strength[frame_index] = float(np.percentile(indentation, 99))
        marker_indentation = sample_at_markers(indentation, marker[frame_index, 0])
        if np.count_nonzero(marker_indentation > CONTACT_THRESHOLD_MM) >= 3:
            correlation = finite_correlation(marker_indentation, displacement[frame_index])
            if correlation is not None:
                marker_contact_correlations.append(correlation)

    contact = contact_fraction > 0.0
    no_contact = ~contact
    if not np.any(no_contact):
        raise RuntimeError(f"No no-contact frame found for {side}")
    rgb_dataset = file_handle[f"tactile/{side}/rgb"]
    baseline = decode_image(rgb_dataset, int(np.flatnonzero(no_contact)[0])).astype(np.float32)
    rgb_no_contact_p95 = []
    rgb_contact_correlations = []
    rgb_inside_outside_ratios = []
    rgb_strength = np.empty(frame_count, dtype=np.float32)

    for frame_index in range(frame_count):
        tactile_rgb = decode_image(rgb_dataset, frame_index).astype(np.float32)
        response = np.mean(np.abs(tactile_rgb - baseline), axis=2)
        rgb_strength[frame_index] = float(np.percentile(response, 95))
        if no_contact[frame_index]:
            rgb_no_contact_p95.append(rgb_strength[frame_index])
        if contact[frame_index] and frame_index % 3 == 0:
            indentation = indentation_frame(file_handle, side, frame_index)
            correlation = finite_correlation(indentation, response)
            if correlation is not None:
                rgb_contact_correlations.append(correlation)
            contact_mask = indentation > CONTACT_THRESHOLD_MM
            if np.any(contact_mask) and np.any(~contact_mask):
                inside = float(np.mean(response[contact_mask]))
                outside = float(np.mean(response[~contact_mask]))
                if outside > 1e-6:
                    rgb_inside_outside_ratios.append(inside / outside)

    marker_peak_frame = int(np.argmax(marker_p95))
    contact_peak_frame = int(np.argmax(contact_fraction))
    contact_peak_indentation = indentation_frame(
        file_handle, side, contact_peak_frame
    )
    contact_peak_mask = contact_peak_indentation > CONTACT_THRESHOLD_MM
    contact_rows = np.flatnonzero(np.any(contact_peak_mask, axis=1))
    no_contact_peak_frame = int(np.flatnonzero(no_contact)[np.argmax(marker_p95[no_contact])])
    render_frames = sorted({0, no_contact_peak_frame, marker_peak_frame, contact_peak_frame, frame_count - 1})
    render_errors = []
    render_coverage = []
    marker_rgb_dataset = file_handle[f"tactile/{side}/rgb_marker"]

    for frame_index in render_frames:
        tactile_rgb = decode_image(rgb_dataset, frame_index).astype(np.float32)
        marker_rgb = decode_image(marker_rgb_dataset, frame_index).astype(np.float32)
        darkness = np.mean(np.maximum(tactile_rgb - marker_rgb, 0.0), axis=2)
        positions = marker[frame_index, 1]
        frame_errors = []
        for position in positions:
            center_x = int(round(float(position[0])))
            center_y = int(round(float(position[1])))
            radius = 7
            x0 = max(center_x - radius, 0)
            x1 = min(center_x + radius + 1, darkness.shape[1])
            y0 = max(center_y - radius, 0)
            y1 = min(center_y + radius + 1, darkness.shape[0])
            patch = darkness[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            weights = np.maximum(patch - np.percentile(patch, 35), 0.0)
            total = float(weights.sum())
            if total < 20.0:
                continue
            grid_y, grid_x = np.indices(weights.shape)
            observed_x = x0 + float(np.sum(grid_x * weights) / total)
            observed_y = y0 + float(np.sum(grid_y * weights) / total)
            frame_errors.append(float(np.hypot(observed_x - position[0], observed_y - position[1])))
        render_errors.extend(frame_errors)
        render_coverage.append(len(frame_errors) / len(positions))

    return {
        "frame_count": int(frame_count),
        "no_contact_frame_count": int(np.count_nonzero(no_contact)),
        "contact_frame_count": int(np.count_nonzero(contact)),
        "no_contact_marker_p95_px_median": percentile(marker_p95[no_contact], 50),
        "no_contact_marker_p95_px_max": percentile(marker_p95[no_contact], 100),
        "no_contact_marker_max_px_max": percentile(marker_max[no_contact], 100),
        "stationary_final_marker_p95_px": float(marker_p95[-1]),
        "contact_marker_p95_px_median": percentile(marker_p95[contact], 50),
        "contact_marker_max_px_max": percentile(marker_max[contact], 100),
        "contact_area_fraction_median": percentile(contact_fraction[contact], 50),
        "contact_area_fraction_max": percentile(contact_fraction[contact], 100),
        "contact_peak_row_min": int(contact_rows[0]) if contact_rows.size else None,
        "contact_peak_row_max": int(contact_rows[-1]) if contact_rows.size else None,
        "contact_peak_row_center": float(np.mean(contact_rows)) if contact_rows.size else None,
        "depth_marker_spatial_correlation_median": percentile(marker_contact_correlations, 50),
        "depth_marker_spatial_correlation_p10": percentile(marker_contact_correlations, 10),
        "no_contact_tactile_rgb_p95_delta_median": percentile(rgb_no_contact_p95, 50),
        "no_contact_tactile_rgb_p95_delta_max": percentile(rgb_no_contact_p95, 100),
        "depth_taxim_spatial_correlation_median": percentile(rgb_contact_correlations, 50),
        "depth_taxim_inside_outside_response_ratio_median": percentile(rgb_inside_outside_ratios, 50),
        "depth_taxim_temporal_correlation": finite_correlation(contact_strength, rgb_strength),
        "numeric_rendered_marker_error_px_median": percentile(render_errors, 50),
        "numeric_rendered_marker_error_px_p95": percentile(render_errors, 95),
        "numeric_rendered_marker_coverage": percentile(render_coverage, 50),
        "representative_frames": {
            "no_contact_peak": no_contact_peak_frame,
            "marker_peak": marker_peak_frame,
            "contact_peak": contact_peak_frame,
            "final": frame_count - 1,
        },
    }


def fit_panel(image, width, height, background=(15, 15, 15)):
    canvas = np.full((height, width, 3), background, dtype=np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    target_width = max(1, int(round(image.shape[1] * scale)))
    target_height = max(1, int(round(image.shape[0] * scale)))
    resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    x0 = (width - target_width) // 2
    y0 = (height - target_height) // 2
    canvas[y0:y0 + target_height, x0:x0 + target_width] = resized
    return canvas


def raw_depth_panel(depth):
    scaled = np.clip((30.0 - depth) / 6.0 * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(scaled, cv2.COLORMAP_VIRIDIS)


def indentation_panel(indentation):
    scaled = np.clip(indentation / 2.0 * 255.0, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    heatmap[indentation <= CONTACT_THRESHOLD_MM] = 0
    return heatmap


def flow_panel(marker_flow, image_shape, display_gain=10.0):
    height, width = image_shape
    canvas = np.full((height, width, 3), 238, dtype=np.uint8)
    initial = marker_flow[0]
    current = marker_flow[1]
    display_current = initial + display_gain * (current - initial)
    for start, end, actual in zip(initial, display_current, current):
        start_point = tuple(np.rint(start).astype(np.int32))
        end_point = tuple(np.rint(end).astype(np.int32))
        actual_point = tuple(np.rint(actual).astype(np.int32))
        cv2.circle(canvas, start_point, 2, (150, 150, 150), -1, cv2.LINE_AA)
        cv2.arrowedLine(canvas, start_point, end_point, (35, 65, 220), 1, cv2.LINE_AA, tipLength=0.25)
        cv2.circle(canvas, actual_point, 2, (20, 20, 20), -1, cv2.LINE_AA)
    return canvas


def draw_label(canvas, text, x, y):
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.64, (245, 245, 245), 2, cv2.LINE_AA)


def apply_optical_gain(image, background, gain):
    image = image.astype(np.float32)
    background = background.astype(np.float32)
    return np.clip(background + gain * (image - background), 0, 255).astype(np.uint8)


def build_video(file_handle, output_path, metrics, fps):
    frame_count = len(file_handle["step"])
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (1920, 1080),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    tags = np.asarray(file_handle["atom/tag"]).astype("U")
    head_dataset = file_handle["observation/head/rgb"]
    wrist_dataset = file_handle["observation/wrist/rgb"]

    for frame_index in range(frame_count):
        canvas = np.full((1080, 1920, 3), 12, dtype=np.uint8)
        head = fit_panel(decode_image(head_dataset, frame_index), 960, 540)
        wrist = fit_panel(decode_image(wrist_dataset, frame_index), 960, 540)
        canvas[:540, :960] = head
        canvas[540:, :960] = wrist
        draw_label(canvas, "Task / head camera", 18, 30)
        draw_label(canvas, "Task / wrist camera", 18, 570)
        draw_label(canvas, f"frame {frame_index:04d}  {tags[frame_index]}", 18, 520)

        for row_index, side in enumerate(SIDES):
            row_y = row_index * 540
            depth = np.asarray(file_handle[f"tactile/{side}/depth"][frame_index])
            indentation = indentation_frame(file_handle, side, frame_index, depth)
            marker = np.asarray(file_handle[f"tactile/{side}/marker"][frame_index])
            panels = (
                ("Raw Depth", raw_depth_panel(depth)),
                ("Indentation", indentation_panel(indentation)),
                ("Tactile RGB", decode_image(file_handle[f"tactile/{side}/rgb"], frame_index)),
                ("FEM flow x10", flow_panel(marker, depth.shape)),
            )
            for column_index, (label, panel) in enumerate(panels):
                x0 = 960 + column_index * 240
                canvas[row_y:row_y + 540, x0:x0 + 240] = fit_panel(panel, 240, 540)
                draw_label(canvas, label, x0 + 8, row_y + 28)
            marker_p95 = metrics[side]["contact_marker_p95_px_median"]
            side_label = "left" if side.startswith("left") else "right"
            draw_label(canvas, f"{side_label} XSense  contact median p95={marker_p95:.2f}px", 968, row_y + 520)

        writer.write(canvas)

    writer.release()


def build_gain_comparison_video(file_handle, output_path, fps):
    frame_count = len(file_handle["step"])
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (1920, 1080),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    tags = np.asarray(file_handle["atom/tag"]).astype("U")
    head_dataset = file_handle["observation/head/rgb"]
    wrist_dataset = file_handle["observation/wrist/rgb"]
    backgrounds = {
        side: decode_image(file_handle[f"tactile/{side}/rgb"], 0)
        for side in SIDES
    }
    gains = (1.0, 0.9, 0.8)

    for frame_index in range(frame_count):
        canvas = np.full((1080, 1920, 3), 12, dtype=np.uint8)
        canvas[:540, :960] = fit_panel(
            decode_image(head_dataset, frame_index), 960, 540
        )
        canvas[540:, :960] = fit_panel(
            decode_image(wrist_dataset, frame_index), 960, 540
        )
        draw_label(canvas, "Task / head camera", 18, 30)
        draw_label(canvas, "Task / wrist camera", 18, 570)
        draw_label(canvas, f"frame {frame_index:04d}  {tags[frame_index]}", 18, 520)

        for row_index, side in enumerate(SIDES):
            row_y = row_index * 540
            tactile_rgb = decode_image(
                file_handle[f"tactile/{side}/rgb"], frame_index
            )
            for column_index, gain in enumerate(gains):
                x0 = 960 + column_index * 320
                variant = apply_optical_gain(
                    tactile_rgb, backgrounds[side], gain
                )
                canvas[row_y:row_y + 540, x0:x0 + 320] = fit_panel(
                    variant, 320, 540
                )
                draw_label(
                    canvas,
                    f"Optical gain {gain:.1f}",
                    x0 + 8,
                    row_y + 28,
                )
            side_label = "left" if side.startswith("left") else "right"
            draw_label(canvas, f"{side_label} XSense", 968, row_y + 520)

        writer.write(canvas)

    writer.release()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("hdf5_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--skip-video", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.hdf5_path, "r") as file_handle:
        metrics = {side: audit_side(file_handle, side) for side in SIDES}
        metrics["task"] = {
            "frame_count": int(len(file_handle["step"])),
            "final_atom_tag": np.asarray(file_handle["atom/tag"])[-1].decode(),
            "hdf5_path": str(args.hdf5_path),
        }
        metrics_path = args.output_dir / "physical_chain_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        if not args.skip_video:
            build_video(file_handle, args.output_dir / "physical_chain_diagnostic.mp4", metrics, args.fps)
            build_gain_comparison_video(
                file_handle,
                args.output_dir / "optical_gain_comparison.mp4",
                args.fps,
            )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
