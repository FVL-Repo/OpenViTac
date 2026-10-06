import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import policy.RL.train_bc as train_bc_module
from policy.RL.train_bc import (
    create_bc_dataloaders,
    evaluate_bc_epoch,
    fit_bc,
    move_batch_to_device,
    prepare_bc_datasets,
    train_bc_epoch,
)


class StubDataset:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class IndexDataset(torch.utils.data.Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return index


class TinyBCDataset(torch.utils.data.Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {
            "qpos": torch.full((7,), float(index)),
        }, torch.zeros(7)


class TinyEvaluationActor(torch.nn.Module):
    action_dim = 7

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))


@pytest.mark.parametrize("device", ["cpu", torch.device("cpu")])
def test_move_batch_to_device_preserves_batch_contract(device):
    observation = {
        "qpos": torch.randn(2, 7, dtype=torch.float32),
        "cam_high": torch.randint(
            0, 256, (2, 3, 16, 16), dtype=torch.uint8
        ),
        "cam_wrist": torch.randint(
            0, 256, (2, 3, 16, 16), dtype=torch.uint8
        ),
        "tac_left": torch.randint(
            0, 256, (2, 3, 16, 16), dtype=torch.uint8
        ),
        "tac_right": torch.randint(
            0, 256, (2, 3, 16, 16), dtype=torch.uint8
        ),
    }
    target_action = torch.randn(2, 7, dtype=torch.float32)
    original_observation = observation.copy()

    device_observation, device_target_action = move_batch_to_device(
        observation=observation,
        target_action=target_action,
        device=device,
    )

    assert device_observation is not observation
    assert observation == original_observation
    assert device_observation.keys() == observation.keys()

    for key, original_value in observation.items():
        moved_value = device_observation[key]
        assert moved_value.device.type == "cpu"
        assert moved_value.dtype == original_value.dtype
        torch.testing.assert_close(moved_value, original_value)

    assert device_target_action.device.type == "cpu"
    assert device_target_action.dtype == torch.float32
    torch.testing.assert_close(device_target_action, target_action)


def test_prepare_bc_datasets_uses_only_train_paths_for_statistics(
    monkeypatch,
    tmp_path,
):
    all_paths = [
        tmp_path / f"{episode_index}.hdf5"
        for episode_index in range(10)
    ]
    action_scale = np.linspace(0.1, 0.7, 7, dtype=np.float32)
    action_statistics = SimpleNamespace(action_scale=action_scale)
    statistics_calls = []

    def fake_compute_action_statistics(**kwargs):
        statistics_calls.append(kwargs)
        return action_statistics

    monkeypatch.setattr(
        train_bc_module,
        "compute_action_statistics",
        fake_compute_action_statistics,
    )
    monkeypatch.setattr(
        train_bc_module,
        "InsertUSBBCDataset",
        StubDataset,
    )

    bundle = prepare_bc_datasets(
        hdf5_paths=all_paths,
        validation_fraction=0.2,
        seed=7,
        image_size=96,
        insertion_tag="insertion",
        require_policy_phase=False,
    )

    assert len(bundle.train_paths) == 8
    assert len(bundle.validation_paths) == 2
    assert set(bundle.train_paths).isdisjoint(bundle.validation_paths)
    assert set(bundle.train_paths + bundle.validation_paths) == set(all_paths)

    assert len(statistics_calls) == 1
    statistics_kwargs = statistics_calls[0]
    assert tuple(statistics_kwargs["hdf5_paths"]) == bundle.train_paths
    assert not set(statistics_kwargs["hdf5_paths"]).intersection(
        bundle.validation_paths
    )
    assert statistics_kwargs["insertion_tag"] == "insertion"
    assert statistics_kwargs["require_policy_phase"] is False

    assert bundle.action_statistics is action_statistics

    train_kwargs = bundle.train_dataset.kwargs
    validation_kwargs = bundle.validation_dataset.kwargs
    assert tuple(train_kwargs["hdf5_paths"]) == bundle.train_paths
    assert tuple(validation_kwargs["hdf5_paths"]) == bundle.validation_paths
    assert train_kwargs["action_scale"] is action_scale
    assert validation_kwargs["action_scale"] is action_scale

    for dataset_kwargs in (train_kwargs, validation_kwargs):
        assert dataset_kwargs["image_size"] == 96
        assert dataset_kwargs["insertion_tag"] == "insertion"
        assert dataset_kwargs["require_policy_phase"] is False


def test_create_bc_dataloaders_batching_and_order_are_reproducible():
    datasets = SimpleNamespace(
        train_dataset=IndexDataset(10),
        validation_dataset=IndexDataset(10),
    )

    first = create_bc_dataloaders(
        datasets=datasets,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        seed=11,
    )
    second = create_bc_dataloaders(
        datasets=datasets,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        seed=11,
    )

    first_train_batches = [batch.tolist() for batch in first.train]
    second_train_batches = [batch.tolist() for batch in second.train]
    validation_batches = [batch.tolist() for batch in first.validation]

    assert first_train_batches == second_train_batches
    assert [len(batch) for batch in first_train_batches] == [4, 4, 2]
    assert sorted(sum(first_train_batches, [])) == list(range(10))
    assert sum(first_train_batches, []) != list(range(10))

    assert validation_batches == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9],
    ]
    assert first.train.persistent_workers is False
    assert first.validation.persistent_workers is False


def test_create_bc_dataloaders_enables_persistent_workers():
    datasets = SimpleNamespace(
        train_dataset=IndexDataset(2),
        validation_dataset=IndexDataset(2),
    )

    loaders = create_bc_dataloaders(
        datasets=datasets,
        batch_size=1,
        num_workers=1,
    )

    assert loaders.train.persistent_workers is True
    assert loaders.validation.persistent_workers is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"batch_size": -1},
        {"num_workers": -1},
    ],
)
def test_create_bc_dataloaders_rejects_invalid_arguments(kwargs):
    datasets = SimpleNamespace(
        train_dataset=IndexDataset(2),
        validation_dataset=IndexDataset(2),
    )

    with pytest.raises(ValueError):
        create_bc_dataloaders(
            datasets=datasets,
            **kwargs,
        )


def test_train_bc_epoch_weights_metrics_by_sample_count(monkeypatch):
    update_outputs = [
        SimpleNamespace(
            loss=torch.tensor(1.0),
            action_mae=torch.tensor(2.0),
            per_joint_mae=torch.full((7,), 3.0),
            grad_norm=torch.tensor(4.0),
        ),
        SimpleNamespace(
            loss=torch.tensor(7.0),
            action_mae=torch.tensor(8.0),
            per_joint_mae=torch.full((7,), 9.0),
            grad_norm=torch.tensor(10.0),
        ),
    ]
    update_calls = []

    def fake_bc_update(**kwargs):
        update_calls.append(kwargs)
        return update_outputs[len(update_calls) - 1]

    monkeypatch.setattr(
        train_bc_module,
        "bc_update",
        fake_bc_update,
    )

    actor = SimpleNamespace(action_dim=7)
    optimizer = object()
    data_loader = torch.utils.data.DataLoader(
        TinyBCDataset(3),
        batch_size=2,
        shuffle=False,
    )

    metrics = train_bc_epoch(
        actor=actor,
        optimizer=optimizer,
        data_loader=data_loader,
        device="cpu",
        max_grad_norm=0.5,
    )

    assert metrics.loss == pytest.approx(3.0)
    assert metrics.action_mae == pytest.approx(4.0)
    assert metrics.per_joint_mae == pytest.approx((5.0,) * 7)
    assert metrics.mean_grad_norm == pytest.approx(7.0)
    assert metrics.sample_count == 3
    assert metrics.batch_count == 2

    assert [
        call["target_action"].shape[0]
        for call in update_calls
    ] == [2, 1]
    assert all(call["actor"] is actor for call in update_calls)
    assert all(call["optimizer"] is optimizer for call in update_calls)
    assert all(call["max_grad_norm"] == 0.5 for call in update_calls)
    assert all(
        call["target_action"].device.type == "cpu"
        for call in update_calls
    )


def test_train_bc_epoch_rejects_empty_data_loader(monkeypatch):
    def fail_if_called(**kwargs):
        raise AssertionError("bc_update must not run for an empty loader")

    monkeypatch.setattr(
        train_bc_module,
        "bc_update",
        fail_if_called,
    )
    data_loader = torch.utils.data.DataLoader(
        TinyBCDataset(0),
        batch_size=2,
    )

    with pytest.raises(ValueError, match="produced no samples"):
        train_bc_epoch(
            actor=SimpleNamespace(action_dim=7),
            optimizer=object(),
            data_loader=data_loader,
            device="cpu",
        )


def test_evaluate_bc_epoch_is_read_only_and_sample_weighted(monkeypatch):
    loss_outputs = [
        SimpleNamespace(
            loss=torch.tensor(1.0),
            action_mae=torch.tensor(2.0),
            per_joint_mae=torch.full((7,), 3.0),
        ),
        SimpleNamespace(
            loss=torch.tensor(7.0),
            action_mae=torch.tensor(8.0),
            per_joint_mae=torch.full((7,), 9.0),
        ),
    ]
    loss_calls = []

    def fake_compute_bc_loss(**kwargs):
        assert not torch.is_grad_enabled()
        assert not kwargs["actor"].training
        loss_calls.append(kwargs)
        return loss_outputs[len(loss_calls) - 1]

    def fail_if_updated(**kwargs):
        raise AssertionError("bc_update must not run during validation")

    monkeypatch.setattr(
        train_bc_module,
        "compute_bc_loss",
        fake_compute_bc_loss,
    )
    monkeypatch.setattr(
        train_bc_module,
        "bc_update",
        fail_if_updated,
    )

    actor = TinyEvaluationActor()
    actor.weight.grad = torch.tensor(3.0)
    parameter_before = actor.weight.detach().clone()
    gradient_before = actor.weight.grad.detach().clone()
    data_loader = torch.utils.data.DataLoader(
        TinyBCDataset(3),
        batch_size=2,
        shuffle=False,
    )

    metrics = evaluate_bc_epoch(
        actor=actor,
        data_loader=data_loader,
        device="cpu",
    )

    assert metrics.loss == pytest.approx(3.0)
    assert metrics.action_mae == pytest.approx(4.0)
    assert metrics.per_joint_mae == pytest.approx((5.0,) * 7)
    assert metrics.sample_count == 3
    assert metrics.batch_count == 2
    assert not actor.training
    torch.testing.assert_close(actor.weight, parameter_before)
    torch.testing.assert_close(actor.weight.grad, gradient_before)
    assert [
        call["target_action"].shape[0]
        for call in loss_calls
    ] == [2, 1]


def test_evaluate_bc_epoch_rejects_empty_data_loader(monkeypatch):
    def fail_if_called(**kwargs):
        raise AssertionError("compute_bc_loss must not run for an empty loader")

    monkeypatch.setattr(
        train_bc_module,
        "compute_bc_loss",
        fail_if_called,
    )
    data_loader = torch.utils.data.DataLoader(
        TinyBCDataset(0),
        batch_size=2,
    )

    with pytest.raises(ValueError, match="produced no samples"):
        evaluate_bc_epoch(
            actor=TinyEvaluationActor(),
            data_loader=data_loader,
            device="cpu",
        )


def test_fit_bc_runs_epochs_and_saves_last_and_best_checkpoints(
    monkeypatch,
    tmp_path,
):
    train_metrics = [
        train_bc_module.BCEpochMetrics(
            loss=loss,
            action_mae=loss + 0.1,
            per_joint_mae=(loss,) * 7,
            mean_grad_norm=1.0,
            sample_count=8,
            batch_count=2,
        )
        for loss in (1.0, 0.8, 0.7)
    ]
    validation_metrics = [
        train_bc_module.BCValidationMetrics(
            loss=loss,
            action_mae=loss + 0.1,
            per_joint_mae=(loss,) * 7,
            sample_count=4,
            batch_count=1,
        )
        for loss in (0.6, 0.4, 0.5)
    ]
    train_calls = []
    validation_calls = []
    build_calls = []
    save_calls = []

    def fake_train_bc_epoch(**kwargs):
        train_calls.append(kwargs)
        return train_metrics[len(train_calls) - 1]

    def fake_evaluate_bc_epoch(**kwargs):
        validation_calls.append(kwargs)
        return validation_metrics[len(validation_calls) - 1]

    def fake_build_bc_checkpoint(**kwargs):
        build_calls.append(kwargs)
        return {
            "completed_epochs": kwargs["completed_epochs"],
        }

    def fake_save_bc_checkpoint(checkpoint, checkpoint_path):
        save_calls.append(
            (checkpoint["completed_epochs"], checkpoint_path.name)
        )
        return checkpoint_path

    monkeypatch.setattr(
        train_bc_module,
        "train_bc_epoch",
        fake_train_bc_epoch,
    )
    monkeypatch.setattr(
        train_bc_module,
        "evaluate_bc_epoch",
        fake_evaluate_bc_epoch,
    )
    monkeypatch.setattr(
        train_bc_module,
        "build_bc_checkpoint",
        fake_build_bc_checkpoint,
    )
    monkeypatch.setattr(
        train_bc_module,
        "save_bc_checkpoint",
        fake_save_bc_checkpoint,
    )

    actor = TinyEvaluationActor()
    optimizer = object()
    data_loaders = SimpleNamespace(
        train=object(),
        validation=object(),
    )
    datasets = SimpleNamespace(
        action_statistics=object(),
        train_paths=(tmp_path / "train.hdf5",),
        validation_paths=(tmp_path / "validation.hdf5",),
    )

    metrics_path = tmp_path / "metrics.jsonl"
    result = fit_bc(
        actor=actor,
        optimizer=optimizer,
        data_loaders=data_loaders,
        datasets=datasets,
        device="cpu",
        target_epochs=5,
        checkpoint_dir=tmp_path,
        image_size=96,
        max_grad_norm=0.5,
        start_epoch=2,
        initial_best_validation_loss=0.7,
        insertion_tag="insertion",
        require_policy_phase=False,
        verbose=False,
        metrics_path=metrics_path,
    )

    assert len(train_calls) == 3
    assert len(validation_calls) == 3
    assert all(call["actor"] is actor for call in train_calls)
    assert all(call["optimizer"] is optimizer for call in train_calls)
    assert all(call["max_grad_norm"] == 0.5 for call in train_calls)
    assert [call["completed_epochs"] for call in build_calls] == [3, 4, 5]
    assert all(call["image_size"] == 96 for call in build_calls)
    assert all(call["insertion_tag"] == "insertion" for call in build_calls)
    assert all(
        call["require_policy_phase"] is False
        for call in build_calls
    )
    assert build_calls[0]["train_metrics"]["loss"] == 1.0
    assert build_calls[1]["validation_metrics"]["loss"] == 0.4
    assert save_calls == [
        (3, "bc_last.pt"),
        (3, "bc_best.pt"),
        (4, "bc_last.pt"),
        (4, "bc_best.pt"),
        (5, "bc_last.pt"),
    ]
    assert result.completed_epochs == 5
    assert result.best_validation_loss == pytest.approx(0.4)
    assert result.train_history == tuple(train_metrics)
    assert result.validation_history == tuple(validation_metrics)
    assert result.last_checkpoint_path == tmp_path / "bc_last.pt"
    assert result.best_checkpoint_path == tmp_path / "bc_best.pt"

    metric_records = [
        json.loads(line)
        for line in metrics_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert [
        record["completed_epochs"]
        for record in metric_records
    ] == [3, 4, 5]
    assert [
        record["is_best"]
        for record in metric_records
    ] == [True, True, False]
    assert metric_records[-1][
        "best_validation_loss"
    ] == pytest.approx(0.4)


@pytest.mark.parametrize(
    "target_epochs,start_epoch,initial_best_validation_loss",
    [
        (0, 0, float("inf")),
        (2, -1, float("inf")),
        (2, 2, float("inf")),
        (2, 0, float("nan")),
        (2, 0, -1.0),
    ],
)
def test_fit_bc_rejects_invalid_training_state(
    target_epochs,
    start_epoch,
    initial_best_validation_loss,
    tmp_path,
):
    with pytest.raises(ValueError):
        fit_bc(
            actor=object(),
            optimizer=object(),
            data_loaders=object(),
            datasets=object(),
            device="cpu",
            target_epochs=target_epochs,
            checkpoint_dir=tmp_path,
            image_size=96,
            start_epoch=start_epoch,
            initial_best_validation_loss=(
                initial_best_validation_loss
            ),
            verbose=False,
        )


def test_fit_bc_rejects_non_finite_validation_loss(
    monkeypatch,
    tmp_path,
):
    train_metrics = train_bc_module.BCEpochMetrics(
        loss=1.0,
        action_mae=1.0,
        per_joint_mae=(1.0,) * 7,
        mean_grad_norm=1.0,
        sample_count=2,
        batch_count=1,
    )
    validation_metrics = train_bc_module.BCValidationMetrics(
        loss=float("nan"),
        action_mae=1.0,
        per_joint_mae=(1.0,) * 7,
        sample_count=2,
        batch_count=1,
    )

    monkeypatch.setattr(
        train_bc_module,
        "train_bc_epoch",
        lambda **kwargs: train_metrics,
    )
    monkeypatch.setattr(
        train_bc_module,
        "evaluate_bc_epoch",
        lambda **kwargs: validation_metrics,
    )

    def fail_if_checkpointed(**kwargs):
        raise AssertionError(
            "a non-finite validation result must not be checkpointed"
        )

    monkeypatch.setattr(
        train_bc_module,
        "build_bc_checkpoint",
        fail_if_checkpointed,
    )

    with pytest.raises(
        FloatingPointError,
        match="validation loss is not finite",
    ):
        fit_bc(
            actor=TinyEvaluationActor(),
            optimizer=object(),
            data_loaders=SimpleNamespace(
                train=object(),
                validation=object(),
            ),
            datasets=object(),
            device="cpu",
            target_epochs=1,
            checkpoint_dir=tmp_path,
            image_size=96,
            verbose=False,
        )
