import json

import pytest

from policy.RL.actor import GaussianActor
from scripts import train_single_step_bc


def test_parse_arguments_loads_json_and_allows_cli_overrides(
    tmp_path,
):
    config_path = tmp_path / "train.json"
    config_path.write_text(
        json.dumps(
            {
                "experiment": {
                    "output_dir": "checkpoints",
                    "overwrite_output": True,
                    "initialize_from": "old_bc.pt",
                    "device": "cpu",
                },
                "data": {
                    "data_dir": "episodes",
                    "action_horizon": 4,
                },
                "training": {
                    "epochs": 50,
                    "batch_size": 8,
                },
                "tactile": {
                    "checkpoint": "tactile.pth",
                    "freeze_backbone": True,
                },
            }
        ),
        encoding="utf-8",
    )

    arguments = train_single_step_bc.parse_arguments(
        [
            "--config",
            str(config_path),
            "--epochs",
            "12",
            "--batch-size",
            "4",
        ]
    )

    assert arguments.config == config_path
    assert arguments.data_dir.name == "episodes"
    assert arguments.action_horizon == 4
    assert arguments.output_dir.name == "checkpoints"
    assert arguments.overwrite_output is True
    assert arguments.initialize_from.name == "old_bc.pt"
    assert arguments.epochs == 12
    assert arguments.batch_size == 4
    assert arguments.device == "cpu"
    assert arguments.tactile_ckpt.name == "tactile.pth"
    assert arguments.freeze_tactile_backbone is True


def test_parse_arguments_rejects_unknown_json_keys(
    tmp_path,
):
    config_path = tmp_path / "train.json"
    config_path.write_text(
        json.dumps(
            {
                "experiment": {
                    "output_dir": "checkpoints",
                },
                "training": {
                    "unknown_option": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit):
        train_single_step_bc.parse_arguments(
            ["--config", str(config_path)]
        )


def test_parse_arguments_rejects_overwrite_when_resuming(tmp_path):
    with pytest.raises(SystemExit):
        train_single_step_bc.parse_arguments(
            [
                "--resume",
                str(tmp_path / "bc_last.pt"),
                "--output-dir",
                str(tmp_path),
                "--epochs",
                "2",
                "--overwrite-output",
            ]
        )


def test_remove_training_artifacts_preserves_other_files(tmp_path):
    artifact_names = (
        "bc_last.pt",
        "bc_best.pt",
        "metrics.jsonl",
    )
    for artifact_name in artifact_names:
        (tmp_path / artifact_name).write_text(
            "old",
            encoding="utf-8",
        )
    preserved_path = tmp_path / "notes.txt"
    preserved_path.write_text("keep", encoding="utf-8")

    artifact_paths = (
        train_single_step_bc.find_existing_training_artifacts(
            tmp_path
        )
    )
    train_single_step_bc.remove_training_artifacts(
        artifact_paths
    )

    assert {path.name for path in artifact_paths} == set(
        artifact_names
    )
    assert all(not path.exists() for path in artifact_paths)
    assert preserved_path.read_text(encoding="utf-8") == "keep"


def test_create_bc_optimizer_uses_separate_tactile_learning_rate():
    actor = GaussianActor()

    optimizer = train_single_step_bc.create_bc_optimizer(
        actor=actor,
        learning_rate=3e-4,
        tactile_learning_rate=1e-5,
        weight_decay=1e-4,
    )

    groups = {
        group["name"]: group
        for group in optimizer.param_groups
    }
    assert set(groups) == {"base", "tactile_backbone"}
    assert groups["base"]["lr"] == pytest.approx(3e-4)
    assert groups["tactile_backbone"]["lr"] == pytest.approx(1e-5)

    grouped_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    trainable_parameters = [
        parameter
        for parameter in actor.parameters()
        if parameter.requires_grad
    ]
    grouped_parameter_ids = [
        id(parameter)
        for parameter in grouped_parameters
    ]
    assert len(grouped_parameter_ids) == len(set(grouped_parameter_ids))
    assert set(grouped_parameter_ids) == {
        id(parameter)
        for parameter in trainable_parameters
    }


def test_configure_bc_optimizer_reapplies_group_hyperparameters():
    actor = GaussianActor()
    optimizer = train_single_step_bc.create_bc_optimizer(
        actor=actor,
        learning_rate=3e-4,
        tactile_learning_rate=1e-5,
        weight_decay=1e-4,
    )
    saved_state = optimizer.state_dict()

    restored_optimizer = train_single_step_bc.create_bc_optimizer(
        actor=actor,
        learning_rate=9e-4,
        tactile_learning_rate=9e-5,
        weight_decay=0.0,
    )
    restored_optimizer.load_state_dict(saved_state)
    train_single_step_bc.configure_bc_optimizer(
        optimizer=restored_optimizer,
        learning_rate=2e-4,
        tactile_learning_rate=2e-5,
        weight_decay=2e-4,
    )

    restored_groups = {
        group["name"]: group
        for group in restored_optimizer.param_groups
    }
    assert restored_groups["base"]["lr"] == pytest.approx(2e-4)
    assert restored_groups["tactile_backbone"]["lr"] == pytest.approx(2e-5)
    assert all(
        group["weight_decay"] == pytest.approx(2e-4)
        for group in restored_groups.values()
    )
