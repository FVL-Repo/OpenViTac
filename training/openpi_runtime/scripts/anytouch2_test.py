"""CPU regressions for the concat baseline; no pretrained weights required."""

import importlib.util
from pathlib import Path
import types

import numpy as np
import pytest
import torch

from openpi import transforms
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.training.config import make_anytouch2_config


def test_relative_chunk_is_based_on_current_observation_and_roundtrips():
    state = np.arange(8, dtype=np.float32)
    targets = np.stack([state + 1, state + 3, state + 7])
    mask = transforms.make_bool_mask(7, -1)
    relative = transforms.DeltaActions(mask)({"state": state.copy(), "actions": targets.copy()})
    np.testing.assert_array_equal(relative["actions"][:, :7], np.array([[1] * 7, [3] * 7, [7] * 7]))
    np.testing.assert_array_equal(relative["actions"][:, 7], targets[:, 7])
    absolute = transforms.AbsoluteActions(mask)(relative)
    np.testing.assert_allclose(absolute["actions"], targets)


def test_prefix_concatenates_tactile_before_vision_and_language():
    cfg = make_anytouch2_config("simulation/unseen_task")
    model = types.SimpleNamespace(
        tactile_image_keys=set(cfg.model.tactile_image_keys),
        tactile_encoder_type="anytouch2",
        anytouch2_encoder=lambda image: torch.full((1, 36, 4), image),
        paligemma_with_expert=types.SimpleNamespace(
            embed_image=lambda image: torch.full((1, 2, 4), image),
            embed_language_tokens=lambda tokens: torch.full((1, 1, 4), 5.0),
        ),
        _apply_checkpoint=lambda fn, *args: fn(*args),
    )
    keys = cfg.model.image_keys
    embeddings, padding, attention = PI0Pytorch.embed_prefix(
        model,
        list(zip(keys, [1.0, 2.0, 3.0, 4.0], strict=True)),
        {key: torch.tensor([True]) for key in keys},
        torch.zeros(1, 1, dtype=torch.long),
        torch.ones(1, 1, dtype=torch.bool),
    )
    assert embeddings.shape == (1, 77, 4)
    assert embeddings[0, :, 0].tolist() == [1.0] * 36 + [2.0] * 36 + [3.0] * 2 + [4.0] * 2 + [10.0]
    assert padding.all()
    assert not attention.any()  # prefix shares bidirectional attention


def test_generic_config_matches_existing_concat_architecture():
    config = make_anytouch2_config("simulation/new_task", anytouch_checkpoint_path="/tmp/weights.pth")
    assert config.model.pi05
    assert not config.model.discrete_state_input
    assert config.model.action_horizon == 50
    assert config.model.anytouch_checkpoint_path == "/tmp/weights.pth"
    assert config.model.anytouch_trainable
    assert config.model.pytorch_compile_mode is None
    assert config.data.repo_id == "simulation/new_task"
    assert config.data.delta_joint_actions
    assert config.data.tactile_history_offsets == (-6, -4, -2, 0)
    assert config.data.assets.assets_dir is None


@pytest.mark.parametrize("repo_id", ["", "/absolute", "../escape"])
def test_reject_invalid_dataset_ids(repo_id):
    with pytest.raises(ValueError, match="repo_id"):
        make_anytouch2_config(repo_id)


def test_hdf5_converter_targets_next_raw_frame_and_clamps_terminal(tmp_path):
    import h5py

    path = Path(__file__).resolve().parents[1] / "examples/UniVTAC/convert_sim_to_lerobot.py"
    spec = importlib.util.spec_from_file_location("base_converter", path)
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    joints = np.arange(45, dtype=np.float32).reshape(5, 9)
    with h5py.File(tmp_path / "episode.hdf5", "w") as episode:
        episode["embodiment/joint"] = joints
        states, targets = converter.make_states_and_actions_for_mode(
            episode, "abs_joint", np.array([0, 2, 4]), 1, "left"
        )
    np.testing.assert_array_equal(states, joints[[0, 2, 4], :8])
    np.testing.assert_array_equal(targets, joints[[1, 3, 4], :8])


def test_lazy_history_timestamps(monkeypatch):
    from openpi.training import data_loader
    from openpi.training.config import DataConfig

    seen = {}
    monkeypatch.setattr(
        data_loader.lerobot_dataset, "LeRobotDatasetMetadata", lambda repo: types.SimpleNamespace(fps=60)
    )

    def dataset(repo_id, **kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(data_loader.lerobot_dataset, "LeRobotDataset", dataset)
    cfg = make_anytouch2_config("simulation/custom")
    data = DataConfig(repo_id=cfg.data.repo_id, observation_history_offsets={"left_tactile_image": (-6, -4, -2, 0)})
    data_loader.create_torch_dataset(data, 50, cfg.model)
    assert seen["delta_timestamps"]["left_tactile_image"] == [-6 / 60, -4 / 60, -2 / 60, 0]
    assert seen["delta_timestamps"]["actions"] == [i / 60 for i in range(50)]


@pytest.mark.parametrize("distributed", [False, True])
def test_existing_run_is_rejected_on_every_rank(tmp_path, monkeypatch, distributed):
    import dataclasses

    from scripts import train_pytorch

    config = dataclasses.replace(
        make_anytouch2_config("simulation/test"),
        checkpoint_base_dir=str(tmp_path),
        exp_name="existing",
        wandb_enabled=False,
    )
    if not distributed:
        config.checkpoint_dir.mkdir(parents=True)
    monkeypatch.setattr(train_pytorch, "setup_ddp", lambda: (distributed, 0, torch.device("cpu")))
    monkeypatch.setattr(train_pytorch, "set_seed", lambda *args: None)
    monkeypatch.setattr(train_pytorch.dist, "get_rank", lambda: 1)

    def broadcast_rank_zero_decision(values, src):
        assert src == 0
        values[0] = True

    monkeypatch.setattr(train_pytorch.dist, "broadcast_object_list", broadcast_rank_zero_decision)
    with pytest.raises(FileExistsError, match="use --resume"):
        train_pytorch.train_loop(config)
