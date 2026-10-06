from __future__ import annotations

import os
import tempfile

from pathlib import Path
from typing import Mapping, Sequence

import h5py
import torch

from policy.RL.action_stats import ActionStatistics
from policy.RL.actor import GaussianActor


BC_CHECKPOINT_KIND = "single_step_bc"
BC_CHECKPOINT_FORMAT_VERSION = 1


def _state_dict_to_cpu(
    module: torch.nn.Module,
    omitted_prefixes: Sequence[str] = (),
) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
        if not any(
            name.startswith(prefix)
            for prefix in omitted_prefixes
        )
    }


def build_actor_checkpoint_payload(
    actor: GaussianActor,
) -> dict[str, object]:
    actor_config = {
        "action_dim": int(actor.action_dim),
        "hidden_dim": int(actor.mu_head.in_features),
        "log_std_min": float(actor.log_std_min),
        "log_std_max": float(actor.log_std_max),
    }

    observation_config = {
        "qpos_dim": int(actor.encoder.qpos_dim),
        "feature_dim": int(actor.encoder.feature_dim),
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
        "tactile_normalization": (
            actor.encoder.tactile_normalization
        ),
        "tactile_output_projection": (
            actor.encoder.tactile_output_projection
        ),
        "freeze_tactile_backbone": (
            actor.encoder.freeze_tactile_backbone
        ),
    }

    omitted_state_dict_prefixes = ()
    if (
        actor.encoder.visual_backbone == "dinov3_vitb16"
        and actor.encoder.freeze_visual_backbone
    ):
        omitted_state_dict_prefixes = (
            "encoder.visual_encoder.backbone.",
        )

    return {
        "class_name": "GaussianActor",
        "config": actor_config,
        "observation_config": observation_config,
        "state_dict": _state_dict_to_cpu(
            actor,
            omitted_prefixes=omitted_state_dict_prefixes,
        ),
        "omitted_state_dict_prefixes": (
            omitted_state_dict_prefixes
        ),
    }

def _as_cpu_float_tensor(
    value,
) -> torch.Tensor:
    return torch.as_tensor(
        value,
        dtype=torch.float32,
    ).detach().cpu().clone()


def build_action_statistics_payload(
    statistics: ActionStatistics,
) -> dict[str, object]:
    return {
        "transition_count": int(
            statistics.transition_count
        ),
        "delta_mean": _as_cpu_float_tensor(
            statistics.delta_mean
        ),
        "delta_std": _as_cpu_float_tensor(
            statistics.delta_std
        ),
        "delta_abs_p95": _as_cpu_float_tensor(
            statistics.delta_abs_p95
        ),
        "delta_abs_p99": _as_cpu_float_tensor(
            statistics.delta_abs_p99
        ),
        "delta_abs_max": _as_cpu_float_tensor(
            statistics.delta_abs_max
        ),
        "action_scale": _as_cpu_float_tensor(
            statistics.action_scale
        ),
    }


def read_common_save_frequency(
    hdf5_paths: Sequence[str | Path],
) -> int:
    paths = [Path(path) for path in hdf5_paths]

    if not paths:
        raise ValueError(
            "At least one HDF5 episode is required"
        )

    frequencies = []

    for path in paths:
        with h5py.File(path, "r") as hdf5_file:
            if (
                "phase" not in hdf5_file
                or "save_frequency"
                not in hdf5_file["phase"].attrs
            ):
                raise KeyError(
                    "Missing phase.attrs['save_frequency'] "
                    f"in {path}"
                )

            save_frequency = int(
                hdf5_file["phase"].attrs[
                    "save_frequency"
                ]
            )

        if save_frequency <= 0:
            raise ValueError(
                f"save_frequency must be positive in {path}"
            )

        frequencies.append(save_frequency)

    unique_frequencies = set(frequencies)

    if len(unique_frequencies) != 1:
        raise ValueError(
            "Episodes use inconsistent save_frequency "
            f"values: {sorted(unique_frequencies)}"
        )

    return frequencies[0]

def _checkpoint_value_to_cpu(
    value,
):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()

    if isinstance(value, dict):
        return {
            key: _checkpoint_value_to_cpu(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _checkpoint_value_to_cpu(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return tuple(
            _checkpoint_value_to_cpu(item)
            for item in value
        )

    return value

def build_bc_checkpoint(
    actor: GaussianActor,
    optimizer: torch.optim.Optimizer,
    action_statistics: ActionStatistics,
    train_paths: Sequence[str | Path],
    validation_paths: Sequence[str | Path],
    completed_epochs: int,
    train_metrics: Mapping[str, object],
    validation_metrics: Mapping[str, object],
    image_size: int,
    insertion_tag: str = "insert_usb_into_slot",
    require_policy_phase: bool = True,
    action_horizon: int = 1,
    zero_qpos: bool = False,
) -> dict[str, object]:
    completed_epochs = int(completed_epochs)
    image_size = int(image_size)
    action_horizon = int(action_horizon)

    if completed_epochs <= 0:
        raise ValueError(
            "completed_epochs must be positive"
        )

    if image_size <= 0:
        raise ValueError("image_size must be positive")

    if action_horizon < 1:
        raise ValueError("action_horizon must be positive")

    if not insertion_tag:
        raise ValueError("insertion_tag must not be empty")

    train_paths = tuple(
        Path(path) for path in train_paths
    )
    validation_paths = tuple(
        Path(path) for path in validation_paths
    )

    if not train_paths:
        raise ValueError("train_paths must not be empty")

    if not validation_paths:
        raise ValueError(
            "validation_paths must not be empty"
        )

    resolved_train_paths = {
        path.resolve() for path in train_paths
    }
    resolved_validation_paths = {
        path.resolve() for path in validation_paths
    }

    if len(resolved_train_paths) != len(train_paths):
        raise ValueError(
            "train_paths contains duplicate episodes"
        )

    if (
        len(resolved_validation_paths)
        != len(validation_paths)
    ):
        raise ValueError(
            "validation_paths contains duplicate episodes"
        )

    if resolved_train_paths.intersection(
        resolved_validation_paths
    ):
        raise ValueError(
            "train_paths and validation_paths overlap"
        )

    source_save_frequency = read_common_save_frequency(
        train_paths + validation_paths
    )

    actor_payload = build_actor_checkpoint_payload(
        actor
    )
    statistics_payload = (
        build_action_statistics_payload(
            action_statistics
        )
    )

    return {
        "kind": BC_CHECKPOINT_KIND,
        "format_version": BC_CHECKPOINT_FORMAT_VERSION,
        "actor": actor_payload,
        "action_statistics": statistics_payload,
        "action_contract": {
            "representation": (
                "normalized_relative_joint_delta"
            ),
            "action_dim": int(actor.action_dim),
            "normalized_bounds": (-1.0, 1.0),
            "target_source": (
                "next_saved_measured_qpos"
                if action_horizon == 1
                else "future_saved_measured_qpos"
            ),
            "saved_frame_stride": action_horizon,
            "source_save_frequency": (
                source_save_frequency
            ),
        },
        "observation_contract": {
            "qpos_key": "qpos",
            "qpos_dim": int(actor.encoder.qpos_dim),
            "camera_keys": tuple(
                actor.encoder.camera_keys
            ),
            "tactile_keys": tuple(
                actor.encoder.tactile_keys
            ),
            "image_size": image_size,
            "zero_qpos": bool(zero_qpos),
        },
        "data_split": {
            "train_paths": tuple(
                str(path) for path in train_paths
            ),
            "validation_paths": tuple(
                str(path)
                for path in validation_paths
            ),
            "insertion_tag": insertion_tag,
            "require_policy_phase": bool(
                require_policy_phase
            ),
        },
        "training_state": {
            "completed_epochs": completed_epochs,
            "optimizer_class": (
                optimizer.__class__.__name__
            ),
            "optimizer_state_dict": (
                _checkpoint_value_to_cpu(
                    optimizer.state_dict()
                )
            ),
            "train_metrics": (
                _checkpoint_value_to_cpu(
                    dict(train_metrics)
                )
            ),
            "validation_metrics": (
                _checkpoint_value_to_cpu(
                    dict(validation_metrics)
                )
            ),
        },
    }

def save_bc_checkpoint(
    checkpoint: Mapping[str, object],
    checkpoint_path: str | Path,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=checkpoint_path.parent,
            prefix=f".{checkpoint_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(
                temporary_file.name
            )

            torch.save(
                dict(checkpoint),
                temporary_file,
            )
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        os.replace(
            temporary_path,
            checkpoint_path,
        )

    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(
                missing_ok=True,
            )
        raise

    return checkpoint_path

def load_bc_checkpoint(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> dict[str, object]:
    checkpoint_path = Path(checkpoint_path)

    loaded = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=True,
    )

    if not isinstance(loaded, Mapping):
        raise TypeError(
            "BC checkpoint must contain a mapping, "
            f"got {type(loaded).__name__}"
        )

    checkpoint = dict(loaded)

    checkpoint_kind = checkpoint.get("kind")
    if checkpoint_kind != BC_CHECKPOINT_KIND:
        raise ValueError(
            "Unexpected checkpoint kind: "
            f"expected {BC_CHECKPOINT_KIND!r}, "
            f"got {checkpoint_kind!r}"
        )

    format_version = checkpoint.get(
        "format_version"
    )
    if (
        format_version
        != BC_CHECKPOINT_FORMAT_VERSION
    ):
        raise ValueError(
            "Unsupported BC checkpoint format version: "
            f"expected {BC_CHECKPOINT_FORMAT_VERSION}, "
            f"got {format_version!r}"
        )

    required_sections = (
        "actor",
        "action_statistics",
        "action_contract",
        "observation_contract",
        "data_split",
        "training_state",
    )

    missing_sections = [
        section_name
        for section_name in required_sections
        if section_name not in checkpoint
    ]

    if missing_sections:
        raise KeyError(
            "BC checkpoint is missing required sections: "
            f"{missing_sections}"
        )

    for section_name in required_sections:
        if not isinstance(
            checkpoint[section_name],
            Mapping,
        ):
            raise TypeError(
                f"BC checkpoint section "
                f"{section_name!r} must be a mapping"
            )

    return checkpoint

def restore_actor_from_bc_checkpoint(
    checkpoint: Mapping[str, object],
    device: str | torch.device = "cpu",
) -> GaussianActor:
    actor_payload = checkpoint.get("actor")

    if not isinstance(actor_payload, Mapping):
        raise TypeError(
            "checkpoint['actor'] must be a mapping"
        )

    class_name = actor_payload.get("class_name")
    if class_name != "GaussianActor":
        raise ValueError(
            "Unsupported actor class: "
            f"expected 'GaussianActor', "
            f"got {class_name!r}"
        )

    actor_config = actor_payload.get("config")
    if not isinstance(actor_config, Mapping):
        raise TypeError(
            "actor config must be a mapping"
        )

    required_config_keys = {
        "action_dim",
        "hidden_dim",
        "log_std_min",
        "log_std_max",
    }

    actual_config_keys = set(actor_config)
    if actual_config_keys != required_config_keys:
        missing_keys = (
            required_config_keys
            - actual_config_keys
        )
        unexpected_keys = (
            actual_config_keys
            - required_config_keys
        )
        raise KeyError(
            "Invalid actor config keys: "
            f"missing={sorted(missing_keys)}, "
            f"unexpected={sorted(unexpected_keys)}"
        )

    observation_config = actor_payload.get(
        "observation_config"
    )
    if not isinstance(observation_config, Mapping):
        raise TypeError(
            "actor observation_config must be a mapping"
        )

    required_observation_config_keys = {
        "qpos_dim",
        "feature_dim",
        "camera_keys",
        "tactile_keys",
        "visual_backbone",
        "tactile_backbone",
        "tactile_normalization",
        "tactile_output_projection",
        "freeze_tactile_backbone",
    }
    optional_observation_config_keys = {
        "visual_pretrained_path",
        "freeze_visual_backbone",
    }
    actual_observation_config_keys = set(
        observation_config
    )
    if (
        not required_observation_config_keys.issubset(
            actual_observation_config_keys
        )
        or not actual_observation_config_keys.issubset(
            required_observation_config_keys
            | optional_observation_config_keys
        )
    ):
        raise KeyError(
            "Invalid actor observation config keys: "
            f"missing={sorted(required_observation_config_keys - actual_observation_config_keys)}, "
            f"unexpected={sorted(actual_observation_config_keys - required_observation_config_keys - optional_observation_config_keys)}"
        )

    normalized_observation_config = dict(observation_config)
    normalized_observation_config.setdefault(
        "visual_pretrained_path",
        None,
    )
    normalized_observation_config.setdefault(
        "freeze_visual_backbone",
        False,
    )

    actor = GaussianActor(
        action_dim=actor_config["action_dim"],
        hidden_dim=actor_config["hidden_dim"],
        log_std_min=actor_config["log_std_min"],
        log_std_max=actor_config["log_std_max"],
        qpos_dim=normalized_observation_config["qpos_dim"],
        camera_keys=normalized_observation_config["camera_keys"],
        tactile_keys=normalized_observation_config["tactile_keys"],
        visual_backbone=(
            normalized_observation_config["visual_backbone"]
        ),
        visual_pretrained_path=(
            normalized_observation_config[
                "visual_pretrained_path"
            ]
        ),
        freeze_visual_backbone=(
            normalized_observation_config[
                "freeze_visual_backbone"
            ]
        ),
        tactile_backbone=(
            normalized_observation_config["tactile_backbone"]
        ),
        tactile_normalization=(
            normalized_observation_config["tactile_normalization"]
        ),
        tactile_output_projection=(
            normalized_observation_config[
                "tactile_output_projection"
            ]
        ),
        freeze_tactile_backbone=(
            normalized_observation_config[
                "freeze_tactile_backbone"
            ]
        ),
    )

    expected_observation_config = {
        "qpos_dim": int(actor.encoder.qpos_dim),
        "feature_dim": int(
            actor.encoder.feature_dim
        ),
        "camera_keys": tuple(
            actor.encoder.camera_keys
        ),
        "tactile_keys": tuple(
            actor.encoder.tactile_keys
        ),
        "visual_backbone": actor.encoder.visual_backbone,
        "visual_pretrained_path": (
            actor.encoder.visual_pretrained_path
        ),
        "freeze_visual_backbone": (
            actor.encoder.freeze_visual_backbone
        ),
        "tactile_backbone": actor.encoder.tactile_backbone,
        "tactile_normalization": (
            actor.encoder.tactile_normalization
        ),
        "tactile_output_projection": (
            actor.encoder.tactile_output_projection
        ),
        "freeze_tactile_backbone": (
            actor.encoder.freeze_tactile_backbone
        ),
    }

    if normalized_observation_config != (
        expected_observation_config
    ):
        raise ValueError(
            "Actor observation configuration is "
            "incompatible with GaussianActor"
        )

    state_dict = actor_payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise TypeError(
            "actor state_dict must be a mapping"
        )

    omitted_state_dict_prefixes = actor_payload.get(
        "omitted_state_dict_prefixes",
        (),
    )
    if not isinstance(
        omitted_state_dict_prefixes,
        (tuple, list),
    ) or not all(
        isinstance(prefix, str) and prefix
        for prefix in omitted_state_dict_prefixes
    ):
        raise TypeError(
            "actor omitted_state_dict_prefixes must be a sequence "
            "of non-empty strings"
        )

    load_result = actor.load_state_dict(
        dict(state_dict),
        strict=not omitted_state_dict_prefixes,
    )
    if omitted_state_dict_prefixes:
        unexpected_keys = load_result.unexpected_keys
        invalid_missing_keys = [
            key
            for key in load_result.missing_keys
            if not any(
                key.startswith(prefix)
                for prefix in omitted_state_dict_prefixes
            )
        ]
        if unexpected_keys or invalid_missing_keys:
            raise RuntimeError(
                "Actor state_dict does not match checkpoint: "
                f"missing={invalid_missing_keys}, "
                f"unexpected={unexpected_keys}"
            )

    return actor.to(device)

def extract_action_horizon_from_bc_checkpoint(
    checkpoint: Mapping[str, object],
) -> int:
    action_contract = checkpoint.get("action_contract")
    if not isinstance(action_contract, Mapping):
        raise TypeError(
            "checkpoint['action_contract'] must be a mapping"
        )

    target_source = action_contract.get("target_source")
    saved_frame_stride = action_contract.get("saved_frame_stride")
    if (
        not isinstance(saved_frame_stride, int)
        or isinstance(saved_frame_stride, bool)
        or saved_frame_stride < 1
    ):
        raise ValueError(
            "saved_frame_stride must be a positive integer"
        )

    expected_target_source = (
        "next_saved_measured_qpos"
        if saved_frame_stride == 1
        else "future_saved_measured_qpos"
    )
    if target_source != expected_target_source:
        raise ValueError(
            "Unsupported action target source for "
            f"saved_frame_stride={saved_frame_stride}: "
            f"{target_source!r}"
        )
    return saved_frame_stride


def extract_action_scale_from_bc_checkpoint(
    checkpoint: Mapping[str, object],
    expected_action_dim: int,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    if (
        not isinstance(expected_action_dim, int)
        or isinstance(expected_action_dim, bool)
        or expected_action_dim <= 0
    ):
        raise ValueError(
            "expected_action_dim must be a positive integer"
        )

    action_contract = checkpoint.get(
        "action_contract"
    )
    if not isinstance(action_contract, Mapping):
        raise TypeError(
            "checkpoint['action_contract'] "
            "must be a mapping"
        )

    representation = action_contract.get(
        "representation"
    )
    if (
        representation
        != "normalized_relative_joint_delta"
    ):
        raise ValueError(
            "Unsupported action representation: "
            f"{representation!r}"
        )

    contract_action_dim = action_contract.get(
        "action_dim"
    )
    if contract_action_dim != expected_action_dim:
        raise ValueError(
            "Action dimension mismatch: "
            f"actor={expected_action_dim}, "
            f"checkpoint={contract_action_dim!r}"
        )

    normalized_bounds = action_contract.get(
        "normalized_bounds"
    )
    if (
        not isinstance(
            normalized_bounds,
            (tuple, list),
        )
        or tuple(normalized_bounds) != (-1.0, 1.0)
    ):
        raise ValueError(
            "Expected normalized action bounds "
            "(-1.0, 1.0)"
        )

    extract_action_horizon_from_bc_checkpoint(checkpoint)

    source_save_frequency = action_contract.get(
        "source_save_frequency"
    )
    if (
        not isinstance(source_save_frequency, int)
        or isinstance(source_save_frequency, bool)
        or source_save_frequency <= 0
    ):
        raise ValueError(
            "source_save_frequency must be "
            "a positive integer"
        )

    statistics = checkpoint.get(
        "action_statistics"
    )
    if not isinstance(statistics, Mapping):
        raise TypeError(
            "checkpoint['action_statistics'] "
            "must be a mapping"
        )

    action_scale = statistics.get(
        "action_scale"
    )
    if not torch.is_tensor(action_scale):
        raise TypeError(
            "action_scale must be a tensor"
        )

    action_scale = (
        action_scale.detach()
        .to(
            device=device,
            dtype=torch.float32,
        )
        .clone()
    )

    if (
        action_scale.ndim != 1
        or action_scale.numel()
        != expected_action_dim
    ):
        raise ValueError(
            "action_scale must have shape "
            f"({expected_action_dim},), "
            f"got {tuple(action_scale.shape)}"
        )

    if not torch.isfinite(action_scale).all().item():
        raise ValueError(
            "action_scale must contain only "
            "finite values"
        )

    if not (action_scale > 0).all().item():
        raise ValueError(
            "action_scale values must be positive"
        )

    return action_scale

def restore_bc_training_state(
    checkpoint: Mapping[str, object],
    optimizer: torch.optim.Optimizer,
) -> int:
    training_state = checkpoint.get(
        "training_state"
    )
    if not isinstance(training_state, Mapping):
        raise TypeError(
            "checkpoint['training_state'] "
            "must be a mapping"
        )

    saved_optimizer_class = training_state.get(
        "optimizer_class"
    )
    current_optimizer_class = (
        optimizer.__class__.__name__
    )

    if (
        saved_optimizer_class
        != current_optimizer_class
    ):
        raise ValueError(
            "Optimizer class mismatch: "
            f"checkpoint={saved_optimizer_class!r}, "
            f"current={current_optimizer_class!r}"
        )

    optimizer_state_dict = training_state.get(
        "optimizer_state_dict"
    )
    if not isinstance(
        optimizer_state_dict,
        Mapping,
    ):
        raise TypeError(
            "optimizer_state_dict must be a mapping"
        )

    required_optimizer_keys = {
        "state",
        "param_groups",
    }
    missing_optimizer_keys = (
        required_optimizer_keys
        - set(optimizer_state_dict)
    )

    if missing_optimizer_keys:
        raise KeyError(
            "Optimizer state is missing keys: "
            f"{sorted(missing_optimizer_keys)}"
        )

    completed_epochs = training_state.get(
        "completed_epochs"
    )
    if (
        not isinstance(completed_epochs, int)
        or isinstance(completed_epochs, bool)
        or completed_epochs <= 0
    ):
        raise ValueError(
            "completed_epochs must be "
            "a positive integer"
        )

    optimizer.load_state_dict(
        dict(optimizer_state_dict)
    )

    return completed_epochs
