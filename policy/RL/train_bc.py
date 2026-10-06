from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence
import torch
import math
from torch.utils.data import DataLoader, Dataset
from dataclasses import asdict, dataclass
from tqdm.auto import tqdm

from policy.RL.checkpoint import (
    build_bc_checkpoint,
    save_bc_checkpoint,
)

from policy.RL.action_stats import (
    ActionStatistics,
    compute_action_statistics,
    split_episode_paths,
)
from policy.RL.dataset import (
    DINOv3PatchTokenCachedDataset,
    InsertUSBBCDataset,
)
from policy.RL.actor import GaussianActor
from policy.RL.bc import bc_update, compute_bc_loss

@dataclass(frozen=True)
class BCDatasetBundle:
    train_paths: tuple[Path, ...]
    validation_paths: tuple[Path, ...]
    action_statistics: ActionStatistics
    train_dataset: Dataset
    validation_dataset: Dataset

@dataclass(frozen=True)
class BCDataLoaders:
    train: DataLoader
    validation: DataLoader

@dataclass(frozen=True)
class BCEpochMetrics:
    loss: float
    action_mae: float
    per_joint_mae: tuple[float, ...]
    mean_grad_norm: float
    sample_count: int
    batch_count: int

@dataclass(frozen=True)
class BCValidationMetrics:
    loss: float
    action_mae: float
    per_joint_mae: tuple[float, ...]
    sample_count: int
    batch_count: int

@dataclass(frozen=True)
class BCTrainingResult:
    completed_epochs: int
    best_validation_loss: float
    train_history: tuple[BCEpochMetrics, ...]
    validation_history: tuple[BCValidationMetrics, ...]
    last_checkpoint_path: Path
    best_checkpoint_path: Path | None


def cache_dinov3_patch_tokens(
    actor: GaussianActor,
    datasets: BCDatasetBundle,
    device: torch.device | str,
    batch_size: int = 128,
    num_workers: int = 0,
) -> BCDatasetBundle:
    """Cache frozen DINOv3 patch tokens once for repeated BC epochs."""
    encoder = actor.encoder.visual_encoder
    if (
        actor.encoder.visual_backbone != "dinov3_vitb16"
        or encoder is None
        or not getattr(encoder, "freeze_backbone", False)
    ):
        raise ValueError(
            "DINOv3 patch caching requires a frozen DINOv3 visual encoder"
        )

    device = torch.device(device)
    cache_batch_size = int(batch_size)
    if cache_batch_size <= 0:
        raise ValueError("batch_size must be positive")

    def cache_dataset(dataset: Dataset) -> Dataset:
        loader = DataLoader(
            dataset,
            batch_size=cache_batch_size,
            shuffle=False,
            num_workers=int(num_workers),
            pin_memory=device.type == "cuda",
            persistent_workers=False,
        )
        token_cache: dict[str, torch.Tensor] = {}
        write_offset = 0
        for observation, _ in loader:
            images = torch.cat(
                [
                    observation[key]
                    for key in actor.encoder.camera_keys
                ],
                dim=0,
            ).to(device=device, non_blocking=True)
            tokens = encoder.encode_patch_tokens(images).cpu()
            batch_count = observation[
                actor.encoder.camera_keys[0]
            ].shape[0]
            split_tokens = tokens.split(batch_count, dim=0)
            if not token_cache:
                token_cache = {
                    key: torch.empty(
                        (len(dataset), *split.shape[1:]),
                        dtype=torch.float16,
                    )
                    for key, split in zip(
                        actor.encoder.camera_keys,
                        split_tokens,
                    )
                }
            for key, split in zip(
                actor.encoder.camera_keys,
                split_tokens,
            ):
                token_cache[key][
                    write_offset : write_offset + batch_count
                ].copy_(split.to(dtype=torch.float16))
            write_offset += batch_count

        if write_offset != len(dataset):
            raise RuntimeError(
                "DINOv3 cache length mismatch: "
                f"wrote {write_offset}, expected {len(dataset)}"
            )
        return DINOv3PatchTokenCachedDataset(
            base_dataset=dataset,
            patch_tokens=token_cache,
        )

    print(
        "[BC] caching frozen DINOv3 patch tokens "
        f"train={len(datasets.train_dataset)} "
        f"validation={len(datasets.validation_dataset)}"
    )
    return BCDatasetBundle(
        train_paths=datasets.train_paths,
        validation_paths=datasets.validation_paths,
        action_statistics=datasets.action_statistics,
        train_dataset=cache_dataset(datasets.train_dataset),
        validation_dataset=cache_dataset(
            datasets.validation_dataset
        ),
    )

def prepare_bc_datasets(
    hdf5_paths: Sequence[str | Path],
    validation_fraction: float = 0.1,
    seed: int = 0,
    image_size: int = 224,
    insertion_tag: str = "insert_usb_into_slot",
    require_policy_phase: bool = True,
    action_horizon: int = 1,
    zero_qpos: bool = False,
) -> BCDatasetBundle:
    train_paths, validation_paths = split_episode_paths(
        hdf5_paths=hdf5_paths,
        validation_fraction=validation_fraction,
        seed=seed,
    )

    action_statistics = compute_action_statistics(
        hdf5_paths=train_paths,
        insertion_tag=insertion_tag,
        require_policy_phase=require_policy_phase,
        action_horizon=action_horizon,
    )

    train_dataset = InsertUSBBCDataset(
        hdf5_paths=train_paths,
        action_scale=action_statistics.action_scale,
        image_size=image_size,
        insertion_tag=insertion_tag,
        require_policy_phase=require_policy_phase,
        action_horizon=action_horizon,
        zero_qpos=zero_qpos,
    )

    validation_dataset = InsertUSBBCDataset(
        hdf5_paths=validation_paths,
        action_scale=action_statistics.action_scale,
        image_size=image_size,
        insertion_tag=insertion_tag,
        require_policy_phase=require_policy_phase,
        action_horizon=action_horizon,
        zero_qpos=zero_qpos,
    )

    return BCDatasetBundle(
        train_paths=tuple(train_paths),
        validation_paths=tuple(validation_paths),
        action_statistics=action_statistics,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
    )

def move_batch_to_device(
    observation: dict[str, torch.Tensor],
    target_action: torch.Tensor,
    device: torch.device | str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    device = torch.device(device)

    device_observation = {
        key: value.to(
            device=device,
            non_blocking=True,
        )
        for key, value in observation.items()
    }

    device_target_action = target_action.to(
        device=device,
        non_blocking=True,
    )

    return device_observation, device_target_action

def create_bc_dataloaders(
    datasets: BCDatasetBundle,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 0,
) -> BCDataLoaders:
    batch_size = int(batch_size)
    num_workers = int(num_workers)
    seed = int(seed)

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if num_workers < 0:
        raise ValueError(
            "num_workers must be non-negative"
        )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        datasets.train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=generator,
    )

    validation_loader = DataLoader(
        datasets.validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    return BCDataLoaders(
        train=train_loader,
        validation=validation_loader,
    )

def train_bc_epoch(
    actor: GaussianActor,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device | str,
    max_grad_norm: float = 1.0,
) -> BCEpochMetrics:
    device = torch.device(device)

    total_loss = torch.zeros(
        (),
        dtype=torch.float64,
        device=device,
    )
    total_action_mae = torch.zeros(
        (),
        dtype=torch.float64,
        device=device,
    )
    total_per_joint_mae = torch.zeros(
        actor.action_dim,
        dtype=torch.float64,
        device=device,
    )
    total_grad_norm = torch.zeros(
        (),
        dtype=torch.float64,
        device=device,
    )

    sample_count = 0
    batch_count = 0

    for observation, target_action in data_loader:
        observation, target_action = move_batch_to_device(
            observation=observation,
            target_action=target_action,
            device=device,
        )

        update_output = bc_update(
            actor=actor,
            optimizer=optimizer,
            observation=observation,
            target_action=target_action,
            max_grad_norm=max_grad_norm,
        )

        batch_size = target_action.shape[0]

        total_loss += (
            update_output.loss.double() * batch_size
        )
        total_action_mae += (
            update_output.action_mae.double() * batch_size
        )
        total_per_joint_mae += (
            update_output.per_joint_mae.double()
            * batch_size
        )
        total_grad_norm += update_output.grad_norm.double()

        sample_count += batch_size
        batch_count += 1

    if sample_count == 0:
        raise ValueError(
            "train data_loader produced no samples"
        )

    per_joint_mae = (
        total_per_joint_mae / sample_count
    ).cpu().tolist()

    return BCEpochMetrics(
        loss=(total_loss / sample_count).item(),
        action_mae=(
            total_action_mae / sample_count
        ).item(),
        per_joint_mae=tuple(per_joint_mae),
        mean_grad_norm=(
            total_grad_norm / batch_count
        ).item(),
        sample_count=sample_count,
        batch_count=batch_count,
    )

def evaluate_bc_epoch(
    actor: GaussianActor,
    data_loader: DataLoader,
    device: torch.device | str,
) -> BCValidationMetrics:
    device = torch.device(device)

    total_loss = torch.zeros(
        (),
        dtype=torch.float64,
        device=device,
    )
    total_action_mae = torch.zeros(
        (),
        dtype=torch.float64,
        device=device,
    )
    total_per_joint_mae = torch.zeros(
        actor.action_dim,
        dtype=torch.float64,
        device=device,
    )

    sample_count = 0
    batch_count = 0

    actor.eval()

    with torch.no_grad():
        for observation, target_action in data_loader:
            observation, target_action = move_batch_to_device(
                observation=observation,
                target_action=target_action,
                device=device,
            )

            loss_output = compute_bc_loss(
                actor=actor,
                observation=observation,
                target_action=target_action,
            )

            batch_size = target_action.shape[0]

            total_loss += (
                loss_output.loss.double() * batch_size
            )
            total_action_mae += (
                loss_output.action_mae.double()
                * batch_size
            )
            total_per_joint_mae += (
                loss_output.per_joint_mae.double()
                * batch_size
            )

            sample_count += batch_size
            batch_count += 1

    if sample_count == 0:
        raise ValueError(
            "validation data_loader produced no samples"
        )

    per_joint_mae = (
        total_per_joint_mae / sample_count
    ).cpu().tolist()

    return BCValidationMetrics(
        loss=(total_loss / sample_count).item(),
        action_mae=(
            total_action_mae / sample_count
        ).item(),
        per_joint_mae=tuple(per_joint_mae),
        sample_count=sample_count,
        batch_count=batch_count,
    )

def fit_bc(
    actor: GaussianActor,
    optimizer: torch.optim.Optimizer,
    data_loaders: BCDataLoaders,
    datasets: BCDatasetBundle,
    device: torch.device | str,
    target_epochs: int,
    checkpoint_dir: str | Path,
    image_size: int,
    max_grad_norm: float = 1.0,
    start_epoch: int = 0,
    initial_best_validation_loss: float = float("inf"),
    insertion_tag: str = "insert_usb_into_slot",
    require_policy_phase: bool = True,
    action_horizon: int = 1,
    zero_qpos: bool = False,
    verbose: bool = True,
    metrics_path: str | Path | None = None,
) -> BCTrainingResult:
    device = torch.device(device)
    target_epochs = int(target_epochs)
    start_epoch = int(start_epoch)
    best_validation_loss = float(
        initial_best_validation_loss
    )

    if target_epochs <= 0:
        raise ValueError("target_epochs must be positive")

    if start_epoch < 0:
        raise ValueError("start_epoch must be non-negative")

    if start_epoch >= target_epochs:
        raise ValueError(
            "start_epoch must be smaller than target_epochs"
        )

    if (
        math.isnan(best_validation_loss)
        or best_validation_loss < 0.0
    ):
        raise ValueError(
            "initial_best_validation_loss must be "
            "non-negative or positive infinity"
        )

    checkpoint_dir = Path(checkpoint_dir)
    if metrics_path is not None:
        metrics_path = Path(metrics_path)
        metrics_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    last_checkpoint_path = (
        checkpoint_dir / "bc_last.pt"
    )
    best_checkpoint_candidate = (
        checkpoint_dir / "bc_best.pt"
    )

    actor.to(device)

    train_history = []
    validation_history = []

    best_checkpoint_exists = (
        best_checkpoint_candidate.is_file()
    )

    epoch_iterator = tqdm(
        range(start_epoch, target_epochs),
        total=target_epochs - start_epoch,
        desc="BC training",
        unit="epoch",
        dynamic_ncols=True,
        disable=not verbose,
    )

    for epoch_index in epoch_iterator:
        completed_epochs = epoch_index + 1

        train_metrics = train_bc_epoch(
            actor=actor,
            optimizer=optimizer,
            data_loader=data_loaders.train,
            device=device,
            max_grad_norm=max_grad_norm,
        )

        validation_metrics = evaluate_bc_epoch(
            actor=actor,
            data_loader=data_loaders.validation,
            device=device,
        )

        if not math.isfinite(validation_metrics.loss):
            raise FloatingPointError(
                "BC validation loss is not finite"
            )

        checkpoint = build_bc_checkpoint(
            actor=actor,
            optimizer=optimizer,
            action_statistics=(
                datasets.action_statistics
            ),
            train_paths=datasets.train_paths,
            validation_paths=datasets.validation_paths,
            completed_epochs=completed_epochs,
            train_metrics=asdict(train_metrics),
            validation_metrics=asdict(
                validation_metrics
            ),
            image_size=image_size,
            insertion_tag=insertion_tag,
            require_policy_phase=require_policy_phase,
            action_horizon=action_horizon,
            zero_qpos=zero_qpos,
        )

        save_bc_checkpoint(
            checkpoint=checkpoint,
            checkpoint_path=last_checkpoint_path,
        )

        is_best = (
            validation_metrics.loss
            < best_validation_loss
        )

        if is_best:
            best_validation_loss = (
                validation_metrics.loss
            )
            save_bc_checkpoint(
                checkpoint=checkpoint,
                checkpoint_path=(
                    best_checkpoint_candidate
                ),
            )
            best_checkpoint_exists = True

        train_history.append(train_metrics)
        validation_history.append(
            validation_metrics
        )

        if metrics_path is not None:
            metrics_record = {
                "completed_epochs": completed_epochs,
                "train": asdict(train_metrics),
                "validation": asdict(
                    validation_metrics
                ),
                "best_validation_loss": (
                    best_validation_loss
                ),
                "is_best": is_best,
            }

            with metrics_path.open(
                "a",
                encoding="utf-8",
            ) as metrics_file:
                json.dump(
                    metrics_record,
                    metrics_file,
                    sort_keys=True,
                )
                metrics_file.write("\n")

        if verbose:
            epoch_iterator.set_postfix(
                train_loss=f"{train_metrics.loss:.6f}",
                validation_loss=(
                    f"{validation_metrics.loss:.6f}"
                ),
                best=f"{best_validation_loss:.6f}",
            )

    return BCTrainingResult(
        completed_epochs=target_epochs,
        best_validation_loss=best_validation_loss,
        train_history=tuple(train_history),
        validation_history=tuple(
            validation_history
        ),
        last_checkpoint_path=last_checkpoint_path,
        best_checkpoint_path=(
            best_checkpoint_candidate
            if best_checkpoint_exists
            else None
        ),
    )
