#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import Subset


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from policy.RL.action_stats import ActionStatistics
from policy.RL.actor import GaussianActor
from policy.RL.checkpoint import (
    extract_action_horizon_from_bc_checkpoint,
    extract_action_scale_from_bc_checkpoint,
    load_bc_checkpoint,
    restore_actor_from_bc_checkpoint,
    restore_bc_training_state,
)
from policy.RL.dataset import InsertUSBBCDataset
from policy.RL.train_bc import (
    BCDatasetBundle,
    cache_dinov3_patch_tokens,
    create_bc_dataloaders,
    fit_bc,
    prepare_bc_datasets,
)


DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_IMAGE_SIZE = 224
DEFAULT_INSERTION_TAG = "insert_usb_into_slot"

JSON_CONFIGURATION_SCHEMA = {
    "experiment": {
        "output_dir": "output_dir",
        "overwrite_output": "overwrite_output",
        "resume": "resume",
        "initialize_from": "initialize_from",
        "seed": "seed",
        "device": "device",
    },
    "data": {
        "data_dir": "data_dir",
        "hdf5_pattern": "hdf5_pattern",
        "recursive": "recursive",
        "validation_fraction": "validation_fraction",
        "image_size": "image_size",
        "insertion_tag": "insertion_tag",
        "require_policy_phase": "require_policy_phase",
        "action_horizon": "action_horizon",
    },
    "policy": {
        "policy_class": "policy_class",
        "action_dim": "action_dim",
        "qpos_dim": "qpos_dim",
        "zero_qpos": "zero_qpos",
        "hidden_dim": "hidden_dim",
        "log_std_min": "log_std_min",
        "log_std_max": "log_std_max",
    },
    "training": {
        "epochs": "epochs",
        "batch_size": "batch_size",
        "num_workers": "num_workers",
        "learning_rate": "learning_rate",
        "weight_decay": "weight_decay",
        "max_grad_norm": "max_grad_norm",
    },
    "vision": {
        "backbone": "visual_backbone",
        "camera_names": "camera_names",
        "pretrained_path": "visual_pretrained_path",
        "freeze_backbone": "freeze_visual_backbone",
    },
    "tactile": {
        "backbone": "tactile_backbone",
        "tactile_names": "tactile_names",
        "checkpoint": "tactile_ckpt",
        "freeze_backbone": "freeze_tactile_backbone",
        "learning_rate": "tactile_learning_rate",
    },
}


@dataclass(frozen=True)
class TrainingSetup:
    actor: GaussianActor
    optimizer: torch.optim.Optimizer
    datasets: BCDatasetBundle
    start_epoch: int
    best_validation_loss: float
    image_size: int
    insertion_tag: str
    require_policy_phase: bool
    action_horizon: int
    mode: str


def _load_json_configuration(
    config_path: Path,
) -> dict[str, object]:
    config_path = config_path.expanduser().resolve()

    with config_path.open("r", encoding="utf-8") as config_file:
        configuration = json.load(config_file)

    if not isinstance(configuration, dict):
        raise TypeError(
            "BC training configuration must contain a JSON object"
        )

    unknown_sections = (
        set(configuration)
        - set(JSON_CONFIGURATION_SCHEMA)
    )
    if unknown_sections:
        raise KeyError(
            "Unknown JSON configuration sections: "
            f"{sorted(unknown_sections)}"
        )

    flattened_configuration = {}
    for section_name, section_schema in (
        JSON_CONFIGURATION_SCHEMA.items()
    ):
        section = configuration.get(section_name, {})
        if not isinstance(section, dict):
            raise TypeError(
                f"JSON configuration section {section_name!r} "
                "must contain an object"
            )

        unknown_keys = set(section) - set(section_schema)
        if unknown_keys:
            raise KeyError(
                f"Unknown keys in JSON section {section_name!r}: "
                f"{sorted(unknown_keys)}"
            )

        for key, value in section.items():
            flattened_configuration[
                section_schema[key]
            ] = value

    return flattened_configuration


def parse_arguments(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(
        add_help=False,
    )
    config_parser.add_argument("--config", type=Path)
    config_arguments, _ = config_parser.parse_known_args(
        argv,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Train or resume the single-step multimodal BC policy."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "JSON configuration file. Explicit CLI options "
            "override its values."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Directory containing HDF5 episodes for a fresh run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for bc_last.pt and bc_best.pt.",
    )
    parser.add_argument(
        "--overwrite-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Replace existing BC checkpoints and metrics for a fresh "
            "run. Other files in the output directory are preserved."
        ),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="BC checkpoint to resume. Its saved data split is reused.",
    )
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help=(
            "Initialize a fresh run from a BC actor checkpoint while "
            "recomputing the data split and action statistics."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Total target epoch count, not the number of extra epochs.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a device such as cuda:1.",
    )
    parser.add_argument("--image-size", type=int)
    parser.add_argument(
        "--policy-class",
        default="GaussianActor",
    )
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--qpos-dim", type=int, default=7)
    parser.add_argument(
        "--zero-qpos",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Replace policy qpos observations with a 7D zero vector.",
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--log-std-min", type=float, default=-5.0)
    parser.add_argument("--log-std-max", type=float, default=2.0)
    parser.add_argument("--insertion-tag")
    parser.add_argument(
        "--action-horizon",
        type=int,
        help=(
            "Number of saved frames between the observation and target "
            "joint state. Defaults to 1 for a fresh run."
        ),
    )
    parser.add_argument(
        "--require-policy-phase",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require phase/id metadata in fresh-run episodes.",
    )
    parser.add_argument(
        "--hdf5-pattern",
        default="*.hdf5",
        help="Pattern relative to --data-dir.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for HDF5 episodes recursively.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--visual-backbone",
        default="resnet18",
    )
    parser.add_argument(
        "--visual-pretrained-path",
        type=Path,
    )
    parser.add_argument(
        "--freeze-visual-backbone",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--camera-names",
        nargs="*",
        default=("cam_high", "cam_wrist"),
    )
    parser.add_argument(
        "--tactile-backbone",
        default="resnet18",
    )
    parser.add_argument(
        "--tactile-names",
        nargs="*",
        default=("tac_left", "tac_right"),
    )
    parser.add_argument("--tactile-ckpt", type=Path)
    parser.add_argument(
        "--tactile-learning-rate",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--freeze-tactile-backbone",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run one bounded epoch on small dataset subsets. "
            "Its checkpoints are diagnostic only."
        ),
    )
    parser.add_argument(
        "--smoke-train-samples",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--smoke-validation-samples",
        type=int,
        default=4,
    )

    if config_arguments.config is not None:
        try:
            configuration = _load_json_configuration(
                config_arguments.config
            )
        except (
            OSError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
        ) as error:
            parser.error(str(error))
        parser.set_defaults(**configuration)

    arguments = parser.parse_args(argv)

    for path_argument in (
        "config",
        "data_dir",
        "output_dir",
        "resume",
        "initialize_from",
        "visual_pretrained_path",
        "tactile_ckpt",
    ):
        value = getattr(arguments, path_argument)
        if value is not None and not isinstance(value, Path):
            setattr(
                arguments,
                path_argument,
                Path(value),
            )

    if arguments.smoke_test:
        if arguments.resume is not None:
            parser.error("--smoke-test cannot be combined with --resume")
        if arguments.epochs not in (None, 1):
            parser.error("--smoke-test requires --epochs 1")
        arguments.epochs = 1
    elif arguments.epochs is None:
        parser.error("--epochs is required unless --smoke-test is used")

    if arguments.resume is None and arguments.data_dir is None:
        parser.error("--data-dir is required for a fresh run")

    if arguments.resume is not None and arguments.data_dir is not None:
        parser.error("--data-dir cannot be combined with --resume")

    if arguments.resume is not None and arguments.overwrite_output:
        parser.error("--overwrite-output cannot be combined with --resume")

    if arguments.resume is not None and arguments.initialize_from is not None:
        parser.error("--resume cannot be combined with --initialize-from")

    if arguments.output_dir is None:
        parser.error("--output-dir is required")

    return arguments


def set_random_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        requested_device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    device = torch.device(requested_device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {requested_device!r} was requested, "
            "but CUDA is unavailable"
        )

    if (
        device.type == "cuda"
        and device.index is not None
        and device.index >= torch.cuda.device_count()
    ):
        raise RuntimeError(
            f"CUDA device index {device.index} is unavailable; "
            f"device_count={torch.cuda.device_count()}"
        )

    return device


def validate_numeric_arguments(arguments: argparse.Namespace) -> None:
    if arguments.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if arguments.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if arguments.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if (
        arguments.learning_rate is not None
        and (
            not math.isfinite(arguments.learning_rate)
            or arguments.learning_rate <= 0.0
        )
    ):
        raise ValueError("--learning-rate must be finite and positive")
    if (
        not math.isfinite(arguments.max_grad_norm)
        or arguments.max_grad_norm <= 0.0
    ):
        raise ValueError("--max-grad-norm must be finite and positive")
    if arguments.image_size is not None and arguments.image_size <= 0:
        raise ValueError("--image-size must be positive")
    if (
        arguments.action_horizon is not None
        and arguments.action_horizon < 1
    ):
        raise ValueError("--action-horizon must be positive")
    if arguments.hidden_dim <= 0:
        raise ValueError("--hidden-dim must be positive")
    if arguments.action_dim != 7:
        raise ValueError(
            "The current single-step dataset requires --action-dim 7"
        )
    if arguments.qpos_dim != 7:
        raise ValueError(
            "The current single-step dataset requires --qpos-dim 7"
        )
    if arguments.policy_class != "GaussianActor":
        raise ValueError(
            "Only --policy-class GaussianActor is supported"
        )
    if arguments.visual_backbone not in (
        "resnet18",
        "dinov3_vitb16",
    ):
        raise ValueError(
            "--visual-backbone must be resnet18 or dinov3_vitb16"
        )
    if arguments.visual_backbone == "dinov3_vitb16":
        if arguments.visual_pretrained_path is None:
            raise ValueError(
                "--visual-backbone dinov3_vitb16 requires "
                "--visual-pretrained-path"
            )
        if not arguments.visual_pretrained_path.expanduser().is_dir():
            raise FileNotFoundError(
                "--visual-pretrained-path is not a directory: "
                f"{arguments.visual_pretrained_path}"
            )
        if arguments.image_size not in (None, 224):
            raise ValueError(
                "dinov3_vitb16 requires --image-size 224"
            )
    elif arguments.visual_pretrained_path is not None:
        raise ValueError(
            "--visual-pretrained-path requires "
            "--visual-backbone dinov3_vitb16"
        )
    if arguments.tactile_backbone != "resnet18":
        raise ValueError(
            "Only --tactile-backbone resnet18 is supported"
        )
    if arguments.log_std_min >= arguments.log_std_max:
        raise ValueError(
            "--log-std-min must be smaller than --log-std-max"
        )
    if (
        not math.isfinite(arguments.weight_decay)
        or arguments.weight_decay < 0.0
    ):
        raise ValueError(
            "--weight-decay must be finite and non-negative"
        )
    for argument_name, names, allowed_names in (
        (
            "--camera-names",
            arguments.camera_names,
            {"cam_high", "cam_wrist"},
        ),
        (
            "--tactile-names",
            arguments.tactile_names,
            {"tac_left", "tac_right"},
        ),
    ):
        if (
            isinstance(names, str)
            or not all(
                isinstance(name, str) and name
                for name in names
            )
            or len(set(names)) != len(names)
            or not set(names).issubset(allowed_names)
        ):
            raise ValueError(
                f"{argument_name} contains invalid or duplicate names"
            )
    if not arguments.camera_names and not arguments.tactile_names:
        raise ValueError(
            "At least one camera or tactile observation is required"
        )
    if arguments.tactile_ckpt is not None and not arguments.tactile_names:
        raise ValueError(
            "--tactile-ckpt requires at least one tactile name"
        )
    if (
        arguments.freeze_tactile_backbone
        and arguments.tactile_ckpt is None
    ):
        raise ValueError(
            "--freeze-tactile-backbone requires --tactile-ckpt"
        )
    if (
        not math.isfinite(arguments.tactile_learning_rate)
        or arguments.tactile_learning_rate <= 0.0
    ):
        raise ValueError(
            "--tactile-learning-rate must be finite and positive"
        )
    if arguments.smoke_train_samples <= 0:
        raise ValueError("--smoke-train-samples must be positive")
    if arguments.smoke_validation_samples <= 0:
        raise ValueError("--smoke-validation-samples must be positive")


def create_bc_optimizer(
    actor: GaussianActor,
    learning_rate: float,
    tactile_learning_rate: float,
    weight_decay: float,
) -> torch.optim.Adam:
    tactile_parameters = []
    if actor.encoder.tactile_encoder is not None:
        tactile_parameters = [
            parameter
            for parameter in (
                actor.encoder.tactile_encoder.parameters()
            )
            if parameter.requires_grad
        ]

    tactile_parameter_ids = {
        id(parameter)
        for parameter in tactile_parameters
    }
    base_parameters = [
        parameter
        for parameter in actor.parameters()
        if (
            parameter.requires_grad
            and id(parameter) not in tactile_parameter_ids
        )
    ]

    parameter_groups = []
    if base_parameters:
        parameter_groups.append(
            {
                "name": "base",
                "params": base_parameters,
                "lr": learning_rate,
            }
        )
    if tactile_parameters:
        parameter_groups.append(
            {
                "name": "tactile_backbone",
                "params": tactile_parameters,
                "lr": tactile_learning_rate,
            }
        )
    if not parameter_groups:
        raise ValueError("Actor has no trainable parameters")

    return torch.optim.Adam(
        parameter_groups,
        weight_decay=weight_decay,
    )


def configure_bc_optimizer(
    optimizer: torch.optim.Optimizer,
    learning_rate: float,
    tactile_learning_rate: float,
    weight_decay: float,
) -> None:
    expected_group_names = {
        "base",
        "tactile_backbone",
    }
    for parameter_group in optimizer.param_groups:
        group_name = parameter_group.get("name")
        if group_name not in expected_group_names:
            raise ValueError(
                "Unexpected optimizer parameter group: "
                f"{group_name!r}"
            )
        parameter_group["lr"] = (
            tactile_learning_rate
            if group_name == "tactile_backbone"
            else learning_rate
        )
        parameter_group["weight_decay"] = weight_decay


def find_hdf5_paths(
    data_dir: Path,
    pattern: str,
    recursive: bool,
) -> tuple[Path, ...]:
    data_dir = data_dir.expanduser().resolve()

    if not data_dir.is_dir():
        raise NotADirectoryError(
            f"Dataset directory does not exist: {data_dir}"
        )

    iterator = (
        data_dir.rglob(pattern)
        if recursive
        else data_dir.glob(pattern)
    )
    paths = tuple(
        sorted(
            {
                path.resolve()
                for path in iterator
                if path.is_file()
            },
            key=str,
        )
    )

    if len(paths) < 2:
        raise ValueError(
            "At least two HDF5 episodes are required, "
            f"but found {len(paths)} in {data_dir}"
        )

    return paths


def _require_mapping(
    parent: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise TypeError(f"checkpoint[{key!r}] must be a mapping")
    return value


def _statistics_vector(
    statistics_payload: Mapping[str, object],
    key: str,
    action_dim: int,
) -> np.ndarray:
    value = statistics_payload.get(key)
    if not torch.is_tensor(value):
        raise TypeError(f"action_statistics[{key!r}] must be a tensor")

    array = value.detach().cpu().numpy().astype(
        np.float32,
        copy=True,
    )
    if array.shape != (action_dim,):
        raise ValueError(
            f"action_statistics[{key!r}] must have shape "
            f"({action_dim},), got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"action_statistics[{key!r}] contains non-finite values"
        )
    return array


def restore_action_statistics(
    checkpoint: Mapping[str, object],
    action_dim: int,
) -> ActionStatistics:
    payload = _require_mapping(checkpoint, "action_statistics")
    transition_count = payload.get("transition_count")

    if (
        not isinstance(transition_count, int)
        or isinstance(transition_count, bool)
        or transition_count <= 0
    ):
        raise ValueError(
            "action_statistics.transition_count must be positive"
        )

    statistics = ActionStatistics(
        transition_count=transition_count,
        delta_mean=_statistics_vector(payload, "delta_mean", action_dim),
        delta_std=_statistics_vector(payload, "delta_std", action_dim),
        delta_abs_p95=_statistics_vector(
            payload,
            "delta_abs_p95",
            action_dim,
        ),
        delta_abs_p99=_statistics_vector(
            payload,
            "delta_abs_p99",
            action_dim,
        ),
        delta_abs_max=_statistics_vector(
            payload,
            "delta_abs_max",
            action_dim,
        ),
        action_scale=_statistics_vector(
            payload,
            "action_scale",
            action_dim,
        ),
    )

    if np.any(statistics.action_scale <= 0.0):
        raise ValueError("checkpoint action_scale must be positive")

    return statistics


def datasets_from_checkpoint(
    checkpoint: Mapping[str, object],
    action_dim: int,
    zero_qpos: bool = False,
) -> tuple[BCDatasetBundle, int, str, bool]:
    data_split = _require_mapping(checkpoint, "data_split")
    observation_contract = _require_mapping(
        checkpoint,
        "observation_contract",
    )

    train_path_values = data_split.get("train_paths")
    validation_path_values = data_split.get("validation_paths")
    if not isinstance(train_path_values, (tuple, list)):
        raise TypeError("data_split.train_paths must be a sequence")
    if not isinstance(validation_path_values, (tuple, list)):
        raise TypeError("data_split.validation_paths must be a sequence")

    train_paths = tuple(Path(path) for path in train_path_values)
    validation_paths = tuple(
        Path(path) for path in validation_path_values
    )
    if not train_paths or not validation_paths:
        raise ValueError("checkpoint data split must not be empty")

    image_size = observation_contract.get("image_size")
    if (
        not isinstance(image_size, int)
        or isinstance(image_size, bool)
        or image_size <= 0
    ):
        raise ValueError("checkpoint image_size must be positive")

    insertion_tag = data_split.get("insertion_tag")
    if not isinstance(insertion_tag, str) or not insertion_tag:
        raise ValueError("checkpoint insertion_tag must not be empty")

    require_policy_phase = data_split.get("require_policy_phase")
    if not isinstance(require_policy_phase, bool):
        raise TypeError(
            "checkpoint require_policy_phase must be boolean"
        )

    statistics = restore_action_statistics(
        checkpoint=checkpoint,
        action_dim=action_dim,
    )
    action_horizon = extract_action_horizon_from_bc_checkpoint(
        checkpoint
    )
    train_dataset = InsertUSBBCDataset(
        hdf5_paths=train_paths,
        action_scale=statistics.action_scale,
        image_size=image_size,
        insertion_tag=insertion_tag,
        require_policy_phase=require_policy_phase,
        action_horizon=action_horizon,
        zero_qpos=zero_qpos,
    )
    validation_dataset = InsertUSBBCDataset(
        hdf5_paths=validation_paths,
        action_scale=statistics.action_scale,
        image_size=image_size,
        insertion_tag=insertion_tag,
        require_policy_phase=require_policy_phase,
        action_horizon=action_horizon,
        zero_qpos=zero_qpos,
    )

    return (
        BCDatasetBundle(
            train_paths=train_paths,
            validation_paths=validation_paths,
            action_statistics=statistics,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
        ),
        image_size,
        insertion_tag,
        require_policy_phase,
    )


def checkpoint_validation_loss(
    checkpoint: Mapping[str, object],
) -> float:
    training_state = _require_mapping(checkpoint, "training_state")
    validation_metrics = _require_mapping(
        training_state,
        "validation_metrics",
    )
    loss = float(validation_metrics.get("loss"))

    if not math.isfinite(loss) or loss < 0.0:
        raise ValueError(
            "checkpoint validation loss must be finite and non-negative"
        )
    return loss


def best_validation_loss_for_resume(
    checkpoint: Mapping[str, object],
    checkpoint_path: Path,
    action_dim: int,
) -> float:
    current_loss = checkpoint_validation_loss(checkpoint)
    best_path = checkpoint_path.with_name("bc_best.pt")

    if not best_path.is_file():
        return current_loss

    best_checkpoint = load_bc_checkpoint(best_path, map_location="cpu")

    for section_name in (
        "data_split",
        "action_contract",
        "observation_contract",
    ):
        if dict(_require_mapping(best_checkpoint, section_name)) != dict(
            _require_mapping(checkpoint, section_name)
        ):
            raise ValueError(
                f"{best_path} does not match the resume checkpoint "
                f"section {section_name!r}"
            )

    current_scale = extract_action_scale_from_bc_checkpoint(
        checkpoint,
        expected_action_dim=action_dim,
    )
    best_scale = extract_action_scale_from_bc_checkpoint(
        best_checkpoint,
        expected_action_dim=action_dim,
    )
    if not torch.equal(current_scale, best_scale):
        raise ValueError(
            f"{best_path} uses a different action_scale"
        )

    return min(
        current_loss,
        checkpoint_validation_loss(best_checkpoint),
    )


def prepare_fresh_training(
    arguments: argparse.Namespace,
    device: torch.device,
) -> TrainingSetup:
    paths = find_hdf5_paths(
        data_dir=arguments.data_dir,
        pattern=arguments.hdf5_pattern,
        recursive=arguments.recursive,
    )
    image_size = arguments.image_size or DEFAULT_IMAGE_SIZE
    insertion_tag = arguments.insertion_tag or DEFAULT_INSERTION_TAG
    require_policy_phase = arguments.require_policy_phase
    action_horizon = arguments.action_horizon or 1

    datasets = prepare_bc_datasets(
        hdf5_paths=paths,
        validation_fraction=arguments.validation_fraction,
        seed=arguments.seed,
        image_size=image_size,
        insertion_tag=insertion_tag,
        require_policy_phase=require_policy_phase,
        action_horizon=action_horizon,
        zero_qpos=arguments.zero_qpos,
    )
    if arguments.initialize_from is None:
        actor = GaussianActor(
            action_dim=arguments.action_dim,
            hidden_dim=arguments.hidden_dim,
            log_std_min=arguments.log_std_min,
            log_std_max=arguments.log_std_max,
            qpos_dim=arguments.qpos_dim,
            camera_keys=arguments.camera_names,
            tactile_keys=arguments.tactile_names,
            visual_backbone=arguments.visual_backbone,
            visual_pretrained_path=(
                arguments.visual_pretrained_path
            ),
            freeze_visual_backbone=(
                arguments.freeze_visual_backbone
            ),
            tactile_backbone=arguments.tactile_backbone,
            tactile_normalization=(
                "frozen_batch_norm"
                if arguments.tactile_ckpt is not None
                else "group_norm"
            ),
            tactile_output_projection=(
                arguments.tactile_ckpt is not None
            ),
            freeze_tactile_backbone=(
                arguments.freeze_tactile_backbone
            ),
        ).to(device)
        if arguments.tactile_ckpt is not None:
            actor.encoder.load_tactile_checkpoint(
                arguments.tactile_ckpt
            )
        mode = "fresh"
    else:
        initialization_checkpoint = load_bc_checkpoint(
            arguments.initialize_from.expanduser().resolve(),
            map_location="cpu",
        )
        actor = restore_actor_from_bc_checkpoint(
            initialization_checkpoint,
            device=device,
        )
        expected_actor_contract = {
            "action_dim": arguments.action_dim,
            "hidden_dim": arguments.hidden_dim,
            "log_std_min": arguments.log_std_min,
            "log_std_max": arguments.log_std_max,
            "qpos_dim": arguments.qpos_dim,
            "camera_keys": tuple(arguments.camera_names),
            "tactile_keys": tuple(arguments.tactile_names),
            "visual_backbone": arguments.visual_backbone,
            "visual_pretrained_path": (
                None
                if arguments.visual_pretrained_path is None
                else str(
                    arguments.visual_pretrained_path
                    .expanduser()
                    .resolve()
                )
            ),
            "freeze_visual_backbone": (
                arguments.freeze_visual_backbone
            ),
            "tactile_backbone": arguments.tactile_backbone,
            "freeze_tactile_backbone": arguments.freeze_tactile_backbone,
        }
        actual_actor_contract = {
            "action_dim": actor.action_dim,
            "hidden_dim": actor.mu_head.in_features,
            "log_std_min": actor.log_std_min,
            "log_std_max": actor.log_std_max,
            "qpos_dim": actor.encoder.qpos_dim,
            "camera_keys": tuple(actor.encoder.camera_keys),
            "tactile_keys": tuple(actor.encoder.tactile_keys),
            "visual_backbone": actor.encoder.visual_backbone,
            "visual_pretrained_path": (
                actor.encoder.visual_pretrained_path
            ),
            "freeze_visual_backbone": (
                actor.encoder.freeze_visual_backbone
            ),
            "tactile_backbone": actor.encoder.tactile_backbone,
            "freeze_tactile_backbone": actor.encoder.freeze_tactile_backbone,
        }
        if actual_actor_contract != expected_actor_contract:
            raise ValueError(
                "--initialize-from actor does not match the requested "
                f"policy configuration: expected={expected_actor_contract}, "
                f"actual={actual_actor_contract}"
            )
        mode = "initialize"
    learning_rate = (
        arguments.learning_rate
        if arguments.learning_rate is not None
        else DEFAULT_LEARNING_RATE
    )
    optimizer = create_bc_optimizer(
        actor=actor,
        learning_rate=learning_rate,
        tactile_learning_rate=(
            arguments.tactile_learning_rate
        ),
        weight_decay=arguments.weight_decay,
    )

    return TrainingSetup(
        actor=actor,
        optimizer=optimizer,
        datasets=datasets,
        start_epoch=0,
        best_validation_loss=float("inf"),
        image_size=image_size,
        insertion_tag=insertion_tag,
        require_policy_phase=require_policy_phase,
        action_horizon=action_horizon,
        mode=mode,
    )


def prepare_resumed_training(
    arguments: argparse.Namespace,
    device: torch.device,
) -> TrainingSetup:
    checkpoint_path = arguments.resume.expanduser().resolve()
    checkpoint = load_bc_checkpoint(
        checkpoint_path,
        map_location="cpu",
    )
    actor = restore_actor_from_bc_checkpoint(
        checkpoint,
        device=device,
    )
    learning_rate = (
        arguments.learning_rate
        if arguments.learning_rate is not None
        else DEFAULT_LEARNING_RATE
    )
    optimizer = create_bc_optimizer(
        actor=actor,
        learning_rate=learning_rate,
        tactile_learning_rate=(
            arguments.tactile_learning_rate
        ),
        weight_decay=arguments.weight_decay,
    )
    start_epoch = restore_bc_training_state(
        checkpoint,
        optimizer=optimizer,
    )

    configure_bc_optimizer(
        optimizer=optimizer,
        learning_rate=learning_rate,
        tactile_learning_rate=(
            arguments.tactile_learning_rate
        ),
        weight_decay=arguments.weight_decay,
    )

    (
        datasets,
        image_size,
        insertion_tag,
        require_policy_phase,
    ) = datasets_from_checkpoint(
        checkpoint=checkpoint,
        action_dim=actor.action_dim,
        zero_qpos=arguments.zero_qpos,
    )
    action_horizon = extract_action_horizon_from_bc_checkpoint(
        checkpoint
    )

    if (
        arguments.image_size is not None
        and arguments.image_size != image_size
    ):
        raise ValueError(
            "--image-size does not match the resume checkpoint"
        )
    if (
        arguments.insertion_tag is not None
        and arguments.insertion_tag != insertion_tag
    ):
        raise ValueError(
            "--insertion-tag does not match the resume checkpoint"
        )
    if (
        arguments.require_policy_phase
        != require_policy_phase
    ):
        raise ValueError(
            "--require-policy-phase does not match the resume checkpoint"
        )
    if (
        arguments.action_horizon is not None
        and arguments.action_horizon != action_horizon
    ):
        raise ValueError(
            "--action-horizon does not match the resume checkpoint"
        )

    best_validation_loss = best_validation_loss_for_resume(
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        action_dim=actor.action_dim,
    )

    return TrainingSetup(
        actor=actor,
        optimizer=optimizer,
        datasets=datasets,
        start_epoch=start_epoch,
        best_validation_loss=best_validation_loss,
        image_size=image_size,
        insertion_tag=insertion_tag,
        require_policy_phase=require_policy_phase,
        action_horizon=action_horizon,
        mode="resume",
    )


def limit_datasets_for_smoke_test(
    datasets: BCDatasetBundle,
    train_sample_limit: int,
    validation_sample_limit: int,
) -> BCDatasetBundle:
    train_count = min(
        len(datasets.train_dataset),
        train_sample_limit,
    )
    validation_count = min(
        len(datasets.validation_dataset),
        validation_sample_limit,
    )

    return replace(
        datasets,
        train_dataset=Subset(
            datasets.train_dataset,
            range(train_count),
        ),
        validation_dataset=Subset(
            datasets.validation_dataset,
            range(validation_count),
        ),
    )


def find_existing_training_artifacts(
    output_dir: Path,
) -> tuple[Path, ...]:
    return tuple(
        path
        for path in (
            output_dir / "bc_last.pt",
            output_dir / "bc_best.pt",
            output_dir / "metrics.jsonl",
        )
        if path.exists()
    )


def remove_training_artifacts(
    artifact_paths: tuple[Path, ...],
) -> None:
    for artifact_path in artifact_paths:
        artifact_path.unlink(missing_ok=True)


def main() -> None:
    arguments = parse_arguments()
    validate_numeric_arguments(arguments)
    set_random_seed(arguments.seed)
    device = resolve_device(arguments.device)
    output_dir = arguments.output_dir.expanduser().resolve()

    if arguments.resume is None:
        existing_artifacts = find_existing_training_artifacts(
            output_dir
        )
        if existing_artifacts and not arguments.overwrite_output:
            raise FileExistsError(
                "Fresh training would overwrite existing checkpoints: "
                f"{list(existing_artifacts)}. Use --resume, a new output "
                "dir, or --overwrite-output."
            )
        setup = prepare_fresh_training(arguments, device)
        if existing_artifacts:
            remove_training_artifacts(existing_artifacts)
            print(
                "[BC] overwrite_output=true removed="
                f"{[path.name for path in existing_artifacts]}"
            )
    else:
        resume_path = arguments.resume.expanduser().resolve()
        if output_dir != resume_path.parent:
            raise ValueError(
                "A resumed run must use the checkpoint directory as "
                "--output-dir so bc_best.pt remains consistent"
            )
        setup = prepare_resumed_training(arguments, device)

    if arguments.epochs <= setup.start_epoch:
        raise ValueError(
            f"--epochs ({arguments.epochs}) must exceed completed "
            f"epochs ({setup.start_epoch})"
        )

    datasets = setup.datasets
    if arguments.smoke_test:
        datasets = limit_datasets_for_smoke_test(
            datasets=datasets,
            train_sample_limit=arguments.smoke_train_samples,
            validation_sample_limit=(
                arguments.smoke_validation_samples
            ),
        )
        print(
            "[BC] SMOKE TEST: generated checkpoints are diagnostic "
            "only and must not initialize SAC."
        )

    if (
        setup.actor.encoder.visual_backbone == "dinov3_vitb16"
        and setup.actor.encoder.freeze_visual_backbone
    ):
        datasets = cache_dinov3_patch_tokens(
            actor=setup.actor,
            datasets=datasets,
            device=device,
            batch_size=arguments.batch_size,
            num_workers=arguments.num_workers,
        )

    data_loaders = create_bc_dataloaders(
        datasets=datasets,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
        pin_memory=device.type == "cuda",
        seed=arguments.seed,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in setup.actor.parameters()
    )
    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in setup.actor.parameters()
        if parameter.requires_grad
    )
    print(
        f"[BC] mode={setup.mode} device={device} "
        f"epochs={setup.start_epoch}->{arguments.epochs} "
        f"action_horizon={setup.action_horizon}"
    )
    print(
        f"[BC] train_samples={len(datasets.train_dataset)} "
        f"validation_samples={len(datasets.validation_dataset)} "
        f"parameters={parameter_count:,} "
        f"trainable_parameters={trainable_parameter_count:,}"
    )
    if (
        setup.actor.encoder.visual_encoder is not None
        and setup.actor.encoder.freeze_visual_backbone
    ):
        print("[BC] visual_backbone=frozen")
    if (
        setup.actor.encoder.tactile_encoder is not None
        and setup.actor.encoder.freeze_tactile_backbone
    ):
        print("[BC] tactile_backbone=frozen")
    if (
        arguments.resume is None
        and arguments.initialize_from is None
        and arguments.tactile_ckpt is not None
    ):
        print(
            "[BC] tactile_checkpoint="
            f"{arguments.tactile_ckpt.expanduser().resolve()}"
        )
    if arguments.initialize_from is not None:
        print(
            "[BC] initialized_actor="
            f"{arguments.initialize_from.expanduser().resolve()}"
        )
    print(f"[BC] output_dir={output_dir}")

    result = fit_bc(
        actor=setup.actor,
        optimizer=setup.optimizer,
        data_loaders=data_loaders,
        datasets=datasets,
        device=device,
        target_epochs=arguments.epochs,
        checkpoint_dir=output_dir,
        image_size=setup.image_size,
        max_grad_norm=arguments.max_grad_norm,
        start_epoch=setup.start_epoch,
        initial_best_validation_loss=(
            setup.best_validation_loss
        ),
        insertion_tag=setup.insertion_tag,
        require_policy_phase=setup.require_policy_phase,
        action_horizon=setup.action_horizon,
        zero_qpos=arguments.zero_qpos,
        verbose=True,
        metrics_path=output_dir / "metrics.jsonl",
    )

    print(
        f"[BC] completed_epochs={result.completed_epochs} "
        f"best_validation_loss={result.best_validation_loss:.6f}"
    )
    print(f"[BC] last_checkpoint={result.last_checkpoint_path}")
    print(f"[BC] best_checkpoint={result.best_checkpoint_path}")


if __name__ == "__main__":
    main()
