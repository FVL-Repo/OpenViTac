#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image


IMAGE_STREAMS = {
    "image": "observation/head/rgb",
    "wrist_image": "observation/wrist/rgb",
    "left_tactile_image": "tactile/left_tactile/rgb_marker",
    "right_tactile_image": "tactile/right_tactile/rgb_marker",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump OpenViTac HDF5 images as the OpenPI sim converter reads them."
    )
    parser.add_argument("hdf5_path", type=Path, help="输入 OpenViTac HDF5 episode 路径。")
    parser.add_argument("--frame", type=int, default=0, help="导出的帧号，默认 0。")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("debug/openpi_hdf5_images"),
        help="输出目录，默认 debug/openpi_hdf5_images。",
    )
    return parser.parse_args()


def decode_like_openpi_sim_converter(dataset: h5py.Dataset, frame_index: int) -> np.ndarray:
    """按 abs_joint sim 转换脚本当前逻辑解码图片。

    输入:
        dataset: HDF5 中的 JPEG byte stream dataset。
        frame_index: 要读取的帧号。

    输出:
        np.ndarray，dtype uint8，shape [H,W,3]。

    说明:
        cv2.imdecode 返回的数组不再做 BGR->RGB；这和
        policy/openpi/abs_joint/convert_insert_usb_sim_to_lerobot.py 保持一致。
    """

    buffer = np.frombuffer(dataset[frame_index], dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV 无法解码 {dataset.name}[{frame_index}]")
    return image


def save_rgb_and_bgr_pair(image: np.ndarray, output_dir: Path, frame: int, name: str) -> None:
    """保存同一个数组按 RGB/BGR 两种解释的对照图。

    输入:
        image: 解码后的 HWC 3 通道数组。
        output_dir: 输出目录。
        frame: 帧号。
        name: 图像流名字。

    输出:
        无。写出 *_as_rgb.png 和 *_as_bgr.png。
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{frame:06d}_{name}"
    Image.fromarray(image.astype(np.uint8), mode="RGB").save(output_dir / f"{prefix}_as_rgb.png")
    Image.fromarray(image[..., ::-1].astype(np.uint8), mode="RGB").save(output_dir / f"{prefix}_as_bgr.png")


def main() -> None:
    args = parse_args()
    hdf5_path = args.hdf5_path.expanduser().resolve()
    if not hdf5_path.is_file():
        raise FileNotFoundError(hdf5_path)

    with h5py.File(hdf5_path, "r") as h5_file:
        for name, h5_key in IMAGE_STREAMS.items():
            image = decode_like_openpi_sim_converter(h5_file[h5_key], args.frame)
            save_rgb_and_bgr_pair(image, args.output_dir, args.frame, name)
            print(f"{name}: {h5_key}[{args.frame}] shape={image.shape} dtype={image.dtype}")

    print(f"Saved debug images to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
