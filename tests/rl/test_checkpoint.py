import h5py
import numpy as np
import pytest
import torch

from policy.RL.actor import GaussianActor
from policy.RL.action_stats import ActionStatistics
from policy.RL.checkpoint import (
    BC_CHECKPOINT_FORMAT_VERSION,
    BC_CHECKPOINT_KIND,
    build_action_statistics_payload,
    build_actor_checkpoint_payload,
    build_bc_checkpoint,
    extract_action_horizon_from_bc_checkpoint,
    extract_action_scale_from_bc_checkpoint,
    load_bc_checkpoint,
    read_common_save_frequency,
    restore_actor_from_bc_checkpoint,
    restore_bc_training_state,
    save_bc_checkpoint,
)


def _make_observation(batch_size=1):
    return {
        "qpos": torch.randn(batch_size, 7),
        "cam_high": torch.randint(
            0, 256, (batch_size, 3, 64, 64), dtype=torch.uint8
        ),
        "cam_wrist": torch.randint(
            0, 256, (batch_size, 3, 64, 64), dtype=torch.uint8
        ),
        "tac_left": torch.randint(
            0, 256, (batch_size, 3, 64, 64), dtype=torch.uint8
        ),
        "tac_right": torch.randint(
            0, 256, (batch_size, 3, 64, 64), dtype=torch.uint8
        ),
    }


def _make_frequency_episode(path, save_frequency=None):
    with h5py.File(path, "w") as hdf5_file:
        phase = hdf5_file.create_group("phase")
        if save_frequency is not None:
            phase.attrs["save_frequency"] = save_frequency


def _make_action_statistics():
    return ActionStatistics(
        transition_count=20,
        delta_mean=np.zeros(7, dtype=np.float32),
        delta_std=np.full(7, 0.1, dtype=np.float32),
        delta_abs_p95=np.full(7, 0.2, dtype=np.float32),
        delta_abs_p99=np.full(7, 0.3, dtype=np.float32),
        delta_abs_max=np.full(7, 0.4, dtype=np.float32),
        action_scale=np.linspace(0.01, 0.07, 7, dtype=np.float32),
    )


def _make_minimal_bc_checkpoint():
    return {
        "kind": BC_CHECKPOINT_KIND,
        "format_version": BC_CHECKPOINT_FORMAT_VERSION,
        "actor": {},
        "action_statistics": {},
        "action_contract": {},
        "observation_contract": {},
        "data_split": {},
        "training_state": {},
    }


def _make_action_scale_checkpoint(action_scale=None):
    if action_scale is None:
        action_scale = torch.linspace(
            0.01,
            0.07,
            7,
        )

    checkpoint = _make_minimal_bc_checkpoint()
    checkpoint["action_contract"] = {
        "representation": "normalized_relative_joint_delta",
        "action_dim": 7,
        "normalized_bounds": (-1.0, 1.0),
        "target_source": "next_saved_measured_qpos",
        "saved_frame_stride": 1,
        "source_save_frequency": 2,
    }
    checkpoint["action_statistics"] = {
        "action_scale": action_scale,
    }
    return checkpoint


def test_actor_checkpoint_payload_contains_reconstruction_contract():
    actor = GaussianActor(
        action_dim=7,
        hidden_dim=128,
        log_std_min=-6.0,
        log_std_max=1.5,
    )

    payload = build_actor_checkpoint_payload(actor)

    assert payload["class_name"] == "GaussianActor"
    assert payload["config"] == {
        "action_dim": 7,
        "hidden_dim": 128,
        "log_std_min": -6.0,
        "log_std_max": 1.5,
    }
    assert payload["observation_config"] == {
        "qpos_dim": 7,
        "feature_dim": 512,
        "camera_keys": ("cam_high", "cam_wrist"),
        "tactile_keys": ("tac_left", "tac_right"),
        "visual_backbone": "resnet18",
        "visual_pretrained_path": None,
        "freeze_visual_backbone": False,
        "tactile_backbone": "resnet18",
        "tactile_normalization": "group_norm",
        "tactile_output_projection": False,
        "freeze_tactile_backbone": False,
    }

    state_dict = payload["state_dict"]
    assert state_dict.keys() == actor.state_dict().keys()
    assert all(tensor.device.type == "cpu" for tensor in state_dict.values())
    assert all(not tensor.requires_grad for tensor in state_dict.values())


def test_actor_checkpoint_payload_is_an_independent_snapshot():
    actor = GaussianActor()
    payload = build_actor_checkpoint_payload(actor)
    parameter_name = next(iter(payload["state_dict"]))
    saved_parameter = payload["state_dict"][parameter_name].clone()

    with torch.no_grad():
        actor.state_dict()[parameter_name].add_(1.0)

    torch.testing.assert_close(
        payload["state_dict"][parameter_name],
        saved_parameter,
    )


def test_actor_payload_round_trips_with_safe_torch_load(tmp_path):
    torch.manual_seed(13)
    actor = GaussianActor(hidden_dim=64).eval()
    payload = build_actor_checkpoint_payload(actor)
    checkpoint_path = tmp_path / "bc_checkpoint.pt"
    torch.save(
        {
            "kind": BC_CHECKPOINT_KIND,
            "format_version": BC_CHECKPOINT_FORMAT_VERSION,
            "actor": payload,
        },
        checkpoint_path,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    actor_payload = checkpoint["actor"]
    restored_actor = GaussianActor(
        **actor_payload["config"]
    ).eval()
    load_result = restored_actor.load_state_dict(
        actor_payload["state_dict"],
        strict=True,
    )

    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []

    observation = _make_observation(batch_size=2)
    with torch.no_grad():
        expected_action = actor.deterministic_action(observation)
        restored_action = restored_actor.deterministic_action(observation)

    torch.testing.assert_close(restored_action, expected_action)


def test_action_statistics_payload_is_safe_cpu_snapshot():
    arrays = [
        np.linspace(index, index + 0.6, 7, dtype=np.float32)
        for index in range(6)
    ]
    statistics = ActionStatistics(
        transition_count=123,
        delta_mean=arrays[0],
        delta_std=arrays[1],
        delta_abs_p95=arrays[2],
        delta_abs_p99=arrays[3],
        delta_abs_max=arrays[4],
        action_scale=arrays[5],
    )

    payload = build_action_statistics_payload(statistics)

    assert payload["transition_count"] == 123
    for key in (
        "delta_mean",
        "delta_std",
        "delta_abs_p95",
        "delta_abs_p99",
        "delta_abs_max",
        "action_scale",
    ):
        assert payload[key].shape == (7,)
        assert payload[key].dtype == torch.float32
        assert payload[key].device.type == "cpu"
        assert not payload[key].requires_grad

    saved_scale = payload["action_scale"].clone()
    statistics.action_scale[:] = -1.0
    torch.testing.assert_close(payload["action_scale"], saved_scale)


def test_read_common_save_frequency_accepts_consistent_episodes(tmp_path):
    paths = [tmp_path / f"{index}.hdf5" for index in range(3)]
    for path in paths:
        _make_frequency_episode(path, save_frequency=2)

    assert read_common_save_frequency(paths) == 2


def test_read_common_save_frequency_rejects_empty_paths():
    with pytest.raises(ValueError, match="At least one"):
        read_common_save_frequency([])


def test_read_common_save_frequency_requires_metadata(tmp_path):
    path = tmp_path / "missing_frequency.hdf5"
    _make_frequency_episode(path)

    with pytest.raises(KeyError, match="save_frequency"):
        read_common_save_frequency([path])


@pytest.mark.parametrize("save_frequency", [0, -1])
def test_read_common_save_frequency_rejects_nonpositive_value(
    tmp_path,
    save_frequency,
):
    path = tmp_path / "invalid_frequency.hdf5"
    _make_frequency_episode(path, save_frequency=save_frequency)

    with pytest.raises(ValueError, match="must be positive"):
        read_common_save_frequency([path])


def test_read_common_save_frequency_rejects_mixed_values(tmp_path):
    first_path = tmp_path / "first.hdf5"
    second_path = tmp_path / "second.hdf5"
    _make_frequency_episode(first_path, save_frequency=2)
    _make_frequency_episode(second_path, save_frequency=4)

    with pytest.raises(ValueError, match="inconsistent"):
        read_common_save_frequency([first_path, second_path])


def test_full_bc_checkpoint_is_safe_and_sac_compatible(tmp_path):
    train_paths = [tmp_path / "train_0.hdf5", tmp_path / "train_1.hdf5"]
    validation_paths = [tmp_path / "validation_0.hdf5"]
    for path in train_paths + validation_paths:
        _make_frequency_episode(path, save_frequency=2)

    torch.manual_seed(17)
    actor = GaussianActor(hidden_dim=64)
    optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    optimizer.zero_grad(set_to_none=True)
    actor.mu_head.weight.square().mean().backward()
    optimizer.step()

    checkpoint = build_bc_checkpoint(
        actor=actor,
        optimizer=optimizer,
        action_statistics=_make_action_statistics(),
        train_paths=train_paths,
        validation_paths=validation_paths,
        completed_epochs=5,
        train_metrics={"loss": 0.25, "per_joint_mae": (0.1,) * 7},
        validation_metrics={"loss": 0.5, "per_joint_mae": (0.2,) * 7},
        image_size=96,
    )

    assert checkpoint["kind"] == BC_CHECKPOINT_KIND
    assert checkpoint["format_version"] == BC_CHECKPOINT_FORMAT_VERSION
    assert checkpoint["action_contract"] == {
        "representation": "normalized_relative_joint_delta",
        "action_dim": 7,
        "normalized_bounds": (-1.0, 1.0),
        "target_source": "next_saved_measured_qpos",
        "saved_frame_stride": 1,
        "source_save_frequency": 2,
    }
    assert checkpoint["observation_contract"]["image_size"] == 96
    assert checkpoint["data_split"]["train_paths"] == tuple(
        str(path) for path in train_paths
    )
    assert checkpoint["data_split"]["validation_paths"] == tuple(
        str(path) for path in validation_paths
    )
    assert checkpoint["training_state"]["completed_epochs"] == 5
    assert checkpoint["training_state"]["optimizer_class"] == "Adam"

    checkpoint_path = tmp_path / "full_bc_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    loaded = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    restored_actor = GaussianActor(
        **loaded["actor"]["config"]
    ).eval()
    load_result = restored_actor.load_state_dict(
        loaded["actor"]["state_dict"],
        strict=True,
    )
    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []
    torch.testing.assert_close(
        loaded["action_statistics"]["action_scale"],
        torch.linspace(0.01, 0.07, 7),
    )

    actor.eval()
    observation = _make_observation(batch_size=1)
    with torch.no_grad():
        expected_action = actor.deterministic_action(observation)
        restored_action = restored_actor.deterministic_action(observation)
    torch.testing.assert_close(restored_action, expected_action)

    optimizer_state = checkpoint["training_state"]["optimizer_state_dict"]
    first_state = next(iter(optimizer_state["state"].values()))
    saved_exp_avg = first_state["exp_avg"].clone()
    optimizer.zero_grad(set_to_none=True)
    actor.mu_head.weight.square().mean().backward()
    optimizer.step()
    torch.testing.assert_close(first_state["exp_avg"], saved_exp_avg)


def test_checkpoint_records_future_action_horizon(tmp_path):
    train_path = tmp_path / "train.hdf5"
    validation_path = tmp_path / "validation.hdf5"
    for path in (train_path, validation_path):
        _make_frequency_episode(path, save_frequency=2)
    actor = GaussianActor(hidden_dim=32)
    checkpoint = build_bc_checkpoint(
        actor=actor,
        optimizer=torch.optim.Adam(actor.parameters()),
        action_statistics=_make_action_statistics(),
        train_paths=[train_path],
        validation_paths=[validation_path],
        completed_epochs=1,
        train_metrics={"loss": 1.0},
        validation_metrics={"loss": 1.0},
        image_size=64,
        action_horizon=4,
    )

    assert checkpoint["action_contract"]["target_source"] == (
        "future_saved_measured_qpos"
    )
    assert checkpoint["action_contract"]["saved_frame_stride"] == 4
    assert extract_action_horizon_from_bc_checkpoint(checkpoint) == 4


def test_build_bc_checkpoint_rejects_invalid_path_splits(tmp_path):
    first_path = tmp_path / "first.hdf5"
    second_path = tmp_path / "second.hdf5"
    _make_frequency_episode(first_path, save_frequency=2)
    _make_frequency_episode(second_path, save_frequency=2)
    actor = GaussianActor(hidden_dim=32)
    optimizer = torch.optim.Adam(actor.parameters())
    common_kwargs = {
        "actor": actor,
        "optimizer": optimizer,
        "action_statistics": _make_action_statistics(),
        "completed_epochs": 1,
        "train_metrics": {"loss": 1.0},
        "validation_metrics": {"loss": 1.0},
        "image_size": 64,
    }

    with pytest.raises(ValueError, match="duplicate"):
        build_bc_checkpoint(
            train_paths=[first_path, first_path],
            validation_paths=[second_path],
            **common_kwargs,
        )

    with pytest.raises(ValueError, match="overlap"):
        build_bc_checkpoint(
            train_paths=[first_path],
            validation_paths=[first_path],
            **common_kwargs,
        )


@pytest.mark.parametrize(
    "override",
    [
        {"completed_epochs": 0},
        {"image_size": 0},
        {"insertion_tag": ""},
        {"train_paths": []},
        {"validation_paths": []},
    ],
)
def test_build_bc_checkpoint_rejects_invalid_metadata(tmp_path, override):
    train_path = tmp_path / "train.hdf5"
    validation_path = tmp_path / "validation.hdf5"
    _make_frequency_episode(train_path, save_frequency=2)
    _make_frequency_episode(validation_path, save_frequency=2)
    actor = GaussianActor(hidden_dim=32)
    kwargs = {
        "actor": actor,
        "optimizer": torch.optim.Adam(actor.parameters()),
        "action_statistics": _make_action_statistics(),
        "train_paths": [train_path],
        "validation_paths": [validation_path],
        "completed_epochs": 1,
        "train_metrics": {"loss": 1.0},
        "validation_metrics": {"loss": 1.0},
        "image_size": 64,
    }
    kwargs.update(override)

    with pytest.raises(ValueError):
        build_bc_checkpoint(**kwargs)


def test_save_bc_checkpoint_creates_parent_and_is_safe_to_load(tmp_path):
    checkpoint = {
        "kind": BC_CHECKPOINT_KIND,
        "format_version": BC_CHECKPOINT_FORMAT_VERSION,
        "tensor": torch.arange(7, dtype=torch.float32),
    }
    checkpoint_path = tmp_path / "nested" / "bc.pt"

    returned_path = save_bc_checkpoint(
        checkpoint,
        checkpoint_path,
    )

    assert returned_path == checkpoint_path
    loaded = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    assert loaded["kind"] == BC_CHECKPOINT_KIND
    torch.testing.assert_close(
        loaded["tensor"],
        checkpoint["tensor"],
    )
    assert list(
        checkpoint_path.parent.glob(
            f".{checkpoint_path.name}.*.tmp"
        )
    ) == []


def test_save_bc_checkpoint_replaces_existing_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "bc.pt"
    save_bc_checkpoint(
        {"epoch": 1},
        checkpoint_path,
    )

    save_bc_checkpoint(
        {"epoch": 2},
        checkpoint_path,
    )

    loaded = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    assert loaded == {"epoch": 2}


def test_save_bc_checkpoint_preserves_old_file_on_write_failure(
    tmp_path,
    monkeypatch,
):
    checkpoint_path = tmp_path / "bc.pt"
    old_contents = b"previous checkpoint"
    checkpoint_path.write_bytes(old_contents)

    def fail_after_partial_write(checkpoint, file_object):
        del checkpoint
        file_object.write(b"partial new checkpoint")
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(
        "policy.RL.checkpoint.torch.save",
        fail_after_partial_write,
    )

    with pytest.raises(RuntimeError, match="simulated write failure"):
        save_bc_checkpoint(
            {"epoch": 2},
            checkpoint_path,
        )

    assert checkpoint_path.read_bytes() == old_contents
    assert list(
        checkpoint_path.parent.glob(
            f".{checkpoint_path.name}.*.tmp"
        )
    ) == []


def test_load_bc_checkpoint_round_trips_valid_payload(tmp_path):
    checkpoint = _make_minimal_bc_checkpoint()
    checkpoint["actor"]["state_dict"] = {
        "weight": torch.arange(4, dtype=torch.float32),
    }
    checkpoint_path = tmp_path / "bc.pt"
    save_bc_checkpoint(checkpoint, checkpoint_path)

    loaded = load_bc_checkpoint(checkpoint_path)

    assert loaded.keys() == checkpoint.keys()
    torch.testing.assert_close(
        loaded["actor"]["state_dict"]["weight"],
        checkpoint["actor"]["state_dict"]["weight"],
    )


def test_load_bc_checkpoint_forces_safe_torch_load(monkeypatch):
    call = {}

    def fake_torch_load(path, map_location, weights_only):
        call["path"] = path
        call["map_location"] = map_location
        call["weights_only"] = weights_only
        return _make_minimal_bc_checkpoint()

    monkeypatch.setattr(
        "policy.RL.checkpoint.torch.load",
        fake_torch_load,
    )

    load_bc_checkpoint(
        "checkpoint.pt",
        map_location="cuda:0",
    )

    assert call["path"].name == "checkpoint.pt"
    assert call["map_location"] == "cuda:0"
    assert call["weights_only"] is True


def test_load_bc_checkpoint_rejects_non_mapping_payload(tmp_path):
    checkpoint_path = tmp_path / "not_a_mapping.pt"
    torch.save(["not", "a", "mapping"], checkpoint_path)

    with pytest.raises(TypeError, match="must contain a mapping"):
        load_bc_checkpoint(checkpoint_path)


@pytest.mark.parametrize(
    ("key", "invalid_value", "error_match"),
    [
        ("kind", "single_step_sac", "checkpoint kind"),
        ("format_version", 999, "format version"),
    ],
)
def test_load_bc_checkpoint_rejects_wrong_identity(
    tmp_path,
    key,
    invalid_value,
    error_match,
):
    checkpoint = _make_minimal_bc_checkpoint()
    checkpoint[key] = invalid_value
    checkpoint_path = tmp_path / f"wrong_{key}.pt"
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match=error_match):
        load_bc_checkpoint(checkpoint_path)


def test_load_bc_checkpoint_rejects_missing_sections(tmp_path):
    checkpoint = _make_minimal_bc_checkpoint()
    del checkpoint["action_contract"]
    checkpoint_path = tmp_path / "missing_section.pt"
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(KeyError, match="action_contract"):
        load_bc_checkpoint(checkpoint_path)


@pytest.mark.parametrize(
    "section_name",
    [
        "actor",
        "action_statistics",
        "action_contract",
        "observation_contract",
        "data_split",
        "training_state",
    ],
)
def test_load_bc_checkpoint_requires_mapping_sections(
    tmp_path,
    section_name,
):
    checkpoint = _make_minimal_bc_checkpoint()
    checkpoint[section_name] = []
    checkpoint_path = tmp_path / f"invalid_{section_name}.pt"
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(TypeError, match=section_name):
        load_bc_checkpoint(checkpoint_path)


def test_restore_actor_reconstructs_identical_policy():
    torch.manual_seed(29)
    original_actor = GaussianActor(
        hidden_dim=32,
        log_std_min=-6.0,
        log_std_max=1.0,
        tactile_normalization="frozen_batch_norm",
        tactile_output_projection=True,
        freeze_tactile_backbone=True,
    ).eval()
    checkpoint = _make_minimal_bc_checkpoint()
    checkpoint["actor"] = build_actor_checkpoint_payload(
        original_actor
    )

    restored_actor = restore_actor_from_bc_checkpoint(
        checkpoint,
        device="cpu",
    ).eval()

    assert restored_actor.action_dim == original_actor.action_dim
    assert restored_actor.mu_head.in_features == 32
    assert restored_actor.log_std_min == -6.0
    assert restored_actor.log_std_max == 1.0
    assert (
        restored_actor.encoder.tactile_normalization
        == "frozen_batch_norm"
    )
    assert restored_actor.encoder.tactile_output_projection
    assert restored_actor.encoder.freeze_tactile_backbone
    assert all(
        not parameter.requires_grad
        for parameter in (
            restored_actor.encoder.tactile_encoder.parameters()
        )
    )
    assert all(
        parameter.device.type == "cpu"
        for parameter in restored_actor.parameters()
    )

    observation = _make_observation(batch_size=2)
    with torch.no_grad():
        expected_action = original_actor.deterministic_action(
            observation
        )
        restored_action = restored_actor.deterministic_action(
            observation
        )

    torch.testing.assert_close(
        restored_action,
        expected_action,
    )


def test_restore_actor_rejects_unsupported_actor_class():
    checkpoint = _make_minimal_bc_checkpoint()
    checkpoint["actor"] = {
        "class_name": "DifferentActor",
    }

    with pytest.raises(ValueError, match="Unsupported actor class"):
        restore_actor_from_bc_checkpoint(checkpoint)


@pytest.mark.parametrize("config_change", ["missing", "unexpected"])
def test_restore_actor_requires_exact_config_keys(config_change):
    actor = GaussianActor(hidden_dim=32)
    actor_payload = build_actor_checkpoint_payload(actor)
    actor_config = dict(actor_payload["config"])

    if config_change == "missing":
        del actor_config["hidden_dim"]
    else:
        actor_config["extra_option"] = True

    actor_payload["config"] = actor_config
    checkpoint = _make_minimal_bc_checkpoint()
    checkpoint["actor"] = actor_payload

    with pytest.raises(KeyError, match="Invalid actor config keys"):
        restore_actor_from_bc_checkpoint(checkpoint)


def test_restore_actor_rejects_incompatible_observation_config():
    actor = GaussianActor(hidden_dim=32)
    actor_payload = build_actor_checkpoint_payload(actor)
    observation_config = dict(
        actor_payload["observation_config"]
    )
    observation_config["feature_dim"] = 256
    actor_payload["observation_config"] = observation_config
    checkpoint = _make_minimal_bc_checkpoint()
    checkpoint["actor"] = actor_payload

    with pytest.raises(ValueError, match="observation configuration"):
        restore_actor_from_bc_checkpoint(checkpoint)


def test_restore_actor_uses_strict_state_dict_loading():
    actor = GaussianActor(hidden_dim=32)
    actor_payload = build_actor_checkpoint_payload(actor)
    state_dict = dict(actor_payload["state_dict"])
    removed_key = next(iter(state_dict))
    del state_dict[removed_key]
    state_dict["unexpected.weight"] = torch.zeros(1)
    actor_payload["state_dict"] = state_dict
    checkpoint = _make_minimal_bc_checkpoint()
    checkpoint["actor"] = actor_payload

    with pytest.raises(RuntimeError):
        restore_actor_from_bc_checkpoint(checkpoint)


def test_extract_action_scale_returns_independent_float32_tensor():
    source_scale = torch.linspace(
        0.01,
        0.07,
        7,
        dtype=torch.float64,
        requires_grad=True,
    )
    checkpoint = _make_action_scale_checkpoint(source_scale)

    action_scale = extract_action_scale_from_bc_checkpoint(
        checkpoint,
        expected_action_dim=7,
        device="cpu",
    )

    assert action_scale.shape == (7,)
    assert action_scale.dtype == torch.float32
    assert action_scale.device.type == "cpu"
    assert not action_scale.requires_grad
    expected_scale = source_scale.detach().float().clone()
    torch.testing.assert_close(action_scale, expected_scale)

    with torch.no_grad():
        source_scale.fill_(1.0)
    torch.testing.assert_close(action_scale, expected_scale)


@pytest.mark.parametrize(
    ("contract_key", "invalid_value", "error_match"),
    [
        ("representation", "absolute_qpos", "representation"),
        ("action_dim", 8, "dimension mismatch"),
        ("normalized_bounds", (0.0, 1.0), "bounds"),
        ("target_source", "planned_qpos", "target source"),
        ("saved_frame_stride", 2, "saved_frame_stride"),
        ("source_save_frequency", 0, "source_save_frequency"),
    ],
)
def test_extract_action_scale_rejects_incompatible_contract(
    contract_key,
    invalid_value,
    error_match,
):
    checkpoint = _make_action_scale_checkpoint()
    checkpoint["action_contract"][contract_key] = invalid_value

    with pytest.raises(ValueError, match=error_match):
        extract_action_scale_from_bc_checkpoint(
            checkpoint,
            expected_action_dim=7,
        )


@pytest.mark.parametrize(
    ("action_scale", "error_type", "error_match"),
    [
        ([0.1] * 7, TypeError, "must be a tensor"),
        (torch.ones(1, 7), ValueError, "shape"),
        (
            torch.tensor([0.1, 0.1, 0.1, float("nan"), 0.1, 0.1, 0.1]),
            ValueError,
            "finite",
        ),
        (
            torch.tensor([0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1]),
            ValueError,
            "positive",
        ),
    ],
)
def test_extract_action_scale_rejects_invalid_scale(
    action_scale,
    error_type,
    error_match,
):
    checkpoint = _make_action_scale_checkpoint(action_scale)

    with pytest.raises(error_type, match=error_match):
        extract_action_scale_from_bc_checkpoint(
            checkpoint,
            expected_action_dim=7,
        )


def test_restore_bc_training_state_restores_adam_and_epoch():
    source_model = torch.nn.Linear(3, 2)
    source_optimizer = torch.optim.Adam(
        source_model.parameters(),
        lr=3e-4,
        betas=(0.8, 0.95),
    )
    source_optimizer.zero_grad(set_to_none=True)
    source_model(torch.ones(4, 3)).square().mean().backward()
    source_optimizer.step()
    source_state = source_optimizer.state_dict()

    checkpoint = _make_minimal_bc_checkpoint()
    checkpoint["training_state"] = {
        "completed_epochs": 7,
        "optimizer_class": "Adam",
        "optimizer_state_dict": source_state,
    }

    restored_model = torch.nn.Linear(3, 2)
    restored_optimizer = torch.optim.Adam(
        restored_model.parameters(),
        lr=1e-2,
    )

    completed_epochs = restore_bc_training_state(
        checkpoint,
        restored_optimizer,
    )

    assert completed_epochs == 7
    restored_state = restored_optimizer.state_dict()
    assert restored_state["param_groups"] == source_state["param_groups"]
    assert restored_state["state"].keys() == source_state["state"].keys()
    for parameter_id, expected_parameter_state in source_state["state"].items():
        actual_parameter_state = restored_state["state"][parameter_id]
        assert actual_parameter_state.keys() == expected_parameter_state.keys()
        for state_name, expected_value in expected_parameter_state.items():
            torch.testing.assert_close(
                actual_parameter_state[state_name],
                expected_value,
            )


def test_restore_bc_training_state_rejects_optimizer_class_mismatch():
    model = torch.nn.Linear(3, 2)
    checkpoint = _make_minimal_bc_checkpoint()
    checkpoint["training_state"] = {
        "completed_epochs": 1,
        "optimizer_class": "Adam",
        "optimizer_state_dict": {
            "state": {},
            "param_groups": [],
        },
    }
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    with pytest.raises(ValueError, match="Optimizer class mismatch"):
        restore_bc_training_state(checkpoint, optimizer)


@pytest.mark.parametrize("completed_epochs", [0, -1, True, None])
def test_restore_bc_training_state_rejects_invalid_completed_epochs(
    completed_epochs,
):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    checkpoint = _make_minimal_bc_checkpoint()
    checkpoint["training_state"] = {
        "completed_epochs": completed_epochs,
        "optimizer_class": "Adam",
        "optimizer_state_dict": optimizer.state_dict(),
    }

    with pytest.raises(ValueError, match="completed_epochs"):
        restore_bc_training_state(checkpoint, optimizer)
