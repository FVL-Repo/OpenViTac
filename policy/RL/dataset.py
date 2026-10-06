from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import h5py
import numpy as np
import torch
from numpy.typing import ArrayLike
from torch.utils.data import Dataset

from .action import ARM_ACTION_DIM, target_qpos_to_action


@dataclass(frozen=True)
class EpisodeLayout:
    path: Path
    left_tactile_path: str
    right_tactile_path: str


def _decode_image(
    encoded_image,
    image_size: int,
) -> torch.Tensor:
    encoded_array = np.frombuffer(
        bytes(encoded_image),
        dtype=np.uint8,
    )
    image = cv2.imdecode(encoded_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not decode JPEG observation")

    image = cv2.resize(
        image,
        (image_size, image_size),
        interpolation=cv2.INTER_AREA,
    )
    chw_image = np.ascontiguousarray(
        image.transpose(2, 0, 1),
        dtype=np.uint8,
    )
    return torch.from_numpy(chw_image)


def _resolve_tactile_path(
    hdf5_file: h5py.File,
    side: str,
) -> str:
    candidates = (
        f"tactile/{side}_tactile/rgb_marker",
        f"tactile/{side}_gsmini/rgb_marker",
    )

    for candidate in candidates:
        if candidate in hdf5_file:
            return candidate

    raise KeyError(
        f"Could not find rgb_marker observation for {side} tactile sensor"
    )


def _validate_action_scale(
    action_scale: ArrayLike,
) -> np.ndarray:
    scale = np.asarray(action_scale, dtype=np.float32)

    if scale.shape != (ARM_ACTION_DIM,):
        raise ValueError(
            f"action_scale must have shape ({ARM_ACTION_DIM},), "
            f"got {scale.shape}"
        )

    if not np.all(np.isfinite(scale)):
        raise ValueError("action_scale contains NaN or infinite values")

    if np.any(scale <= 0.0):
        raise ValueError("action_scale must contain only positive values")

    return scale.copy()


class InsertUSBBCDataset(Dataset):
    """
    Single-step BC samples from the scripted USB insertion stage.
    """

    def __init__(
        self,
        hdf5_paths: Sequence[str | Path],
        action_scale: ArrayLike,
        image_size: int = 224,
        insertion_tag: str = "insert_usb_into_slot",
        require_policy_phase: bool = True,
        action_horizon: int = 1,
        zero_qpos: bool = False,
    ):
        self.paths = [Path(path) for path in hdf5_paths]

        if not self.paths:
            raise ValueError("No HDF5 episodes were provided")

        self.action_scale = _validate_action_scale(action_scale)
        self.image_size = int(image_size)
        self.action_horizon = int(action_horizon)
        self.zero_qpos = bool(zero_qpos)

        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if (
            isinstance(action_horizon, bool)
            or self.action_horizon < 1
        ):
            raise ValueError("action_horizon must be a positive integer")

        normalized_insertion_tags = tuple(
            dict.fromkeys(
                tag.strip().lower()
                for tag in insertion_tag.split(",")
                if tag.strip()
            )
        )
        if not normalized_insertion_tags:
            raise ValueError("insertion_tag must contain at least one tag")
        self.episodes: list[EpisodeLayout] = []
        self.records: list[tuple[int, int]] = []
        self.target_indices: list[int] = []

        for path in self.paths:
            if not path.is_file():
                raise FileNotFoundError(f"HDF5 episode does not exist: {path}")

            with h5py.File(path, "r") as hdf5_file:
                required_paths = (
                    "embodiment/joint",
                    "atom/tag",
                    "observation/head/rgb",
                    "observation/wrist/rgb",
                )
                for required_path in required_paths:
                    if required_path not in hdf5_file:
                        raise KeyError(
                            f"Missing {required_path} in {path}"
                        )

                left_tactile_path = _resolve_tactile_path(
                    hdf5_file,
                    "left",
                )
                right_tactile_path = _resolve_tactile_path(
                    hdf5_file,
                    "right",
                )

                joints = hdf5_file["embodiment/joint"]
                frame_count = len(joints)

                if frame_count <= self.action_horizon:
                    continue

                if joints.shape[-1] < ARM_ACTION_DIM:
                    raise ValueError(
                        f"Expected at least {ARM_ACTION_DIM} joints in {path}, "
                        f"got shape {joints.shape}"
                    )

                tags = hdf5_file["atom/tag"][()].astype("U")

                if len(tags) != frame_count:
                    raise ValueError(
                        f"atom/tag and embodiment/joint lengths differ in {path}"
                    )

                pair_count = frame_count - self.action_horizon
                pair_mask = np.ones(pair_count, dtype=bool)

                if "provenance/transition_valid" in hdf5_file:
                    transition_valid = np.asarray(
                        hdf5_file["provenance/transition_valid"],
                        dtype=bool,
                    )
                    if len(transition_valid) != frame_count:
                        raise ValueError(
                            "provenance/transition_valid and embodiment/joint "
                            f"lengths differ in {path}"
                        )
                    for offset in range(1, self.action_horizon + 1):
                        pair_mask &= transition_valid[
                            offset : offset + pair_count
                        ]

                if "phase/id" in hdf5_file:
                    phase_ids = hdf5_file["phase/id"][()]

                    if len(phase_ids) != frame_count:
                        raise ValueError(
                            f"phase/id and embodiment/joint lengths differ in {path}"
                        )

                    policy_id = int(
                        hdf5_file["phase"].attrs.get("policy_id", 1)
                    )
                    for offset in range(self.action_horizon + 1):
                        pair_mask &= (
                            phase_ids[offset : offset + pair_count]
                            == policy_id
                        )
                elif require_policy_phase:
                    raise KeyError(f"Missing phase/id in {path}")

                # Every destination frame in qpos[t] -> qpos[t+H] must remain
                # inside the selected policy suffix.
                for offset in range(1, self.action_horizon + 1):
                    destination_tags = np.char.lower(
                        tags[offset : offset + pair_count]
                    )
                    pair_mask &= np.isin(
                        destination_tags,
                        normalized_insertion_tags,
                    )

                episode_index = len(self.episodes)
                self.episodes.append(
                    EpisodeLayout(
                        path=path,
                        left_tactile_path=left_tactile_path,
                        right_tactile_path=right_tactile_path,
                    )
                )

                frame_indices = np.flatnonzero(pair_mask)
                self.records.extend(
                    (episode_index, int(frame_index))
                    for frame_index in frame_indices
                )
                self.target_indices.extend(
                    int(frame_index + self.action_horizon)
                    for frame_index in frame_indices
                )

        if not self.records:
            raise ValueError(
                f"No transitions matched insertion tag(s) "
                f"{insertion_tag!r}"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        episode_index, frame_index = self.records[index]
        target_frame_index = self.target_indices[index]
        layout = self.episodes[episode_index]

        with h5py.File(layout.path, "r") as hdf5_file:
            joints = hdf5_file["embodiment/joint"]

            current_qpos = np.asarray(
                joints[frame_index, :ARM_ACTION_DIM],
                dtype=np.float32,
            )
            target_qpos = np.asarray(
                joints[target_frame_index, :ARM_ACTION_DIM],
                dtype=np.float32,
            )

            observation = {
                "qpos": torch.from_numpy(
                    np.zeros_like(current_qpos)
                    if self.zero_qpos
                    else current_qpos.copy()
                ),
                "tac_left": _decode_image(
                    hdf5_file[layout.left_tactile_path][frame_index],
                    self.image_size,
                ),
                "tac_right": _decode_image(
                    hdf5_file[layout.right_tactile_path][frame_index],
                    self.image_size,
                ),
            }
            if not getattr(self, "_skip_camera_decode", False):
                observation["cam_high"] = _decode_image(
                    hdf5_file["observation/head/rgb"][frame_index],
                    self.image_size,
                )
                observation["cam_wrist"] = _decode_image(
                    hdf5_file["observation/wrist/rgb"][frame_index],
                    self.image_size,
                )

        normalized_action = target_qpos_to_action(
            current_qpos=current_qpos,
            target_qpos=target_qpos,
            action_scale=self.action_scale,
        )
        action = torch.from_numpy(normalized_action)

        return observation, action


class DINOv3PatchTokenCachedDataset(Dataset):
    """Replace camera JPEGs with cached frozen DINOv3 patch tokens."""

    def __init__(
        self,
        base_dataset: Dataset,
        patch_tokens: dict[str, torch.Tensor],
    ):
        self.base_dataset = base_dataset
        self.patch_tokens = dict(patch_tokens)
        if not self.patch_tokens:
            raise ValueError("patch_tokens must not be empty")
        expected_count = len(base_dataset)
        for key, tokens in self.patch_tokens.items():
            if tokens.ndim != 3 or tokens.shape[0] != expected_count:
                raise ValueError(
                    f"cached patch tokens for {key!r} must have shape "
                    f"(len(dataset), patches, hidden), got "
                    f"{tuple(tokens.shape)}"
                )
        dataset = base_dataset
        while hasattr(dataset, "dataset"):
            dataset = dataset.dataset
        if isinstance(dataset, InsertUSBBCDataset):
            dataset._skip_camera_decode = True

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        observation, action = self.base_dataset[index]
        for key, tokens in self.patch_tokens.items():
            observation[key] = tokens[index]
        return observation, action
