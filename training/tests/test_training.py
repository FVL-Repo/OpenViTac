"""CPU checks for task configuration and simulated HDF5 conversion."""

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))
import config
import run


def load_converter(monkeypatch):
    for name in ("lerobot", "lerobot.common", "lerobot.common.datasets", "lerobot.common.datasets.lerobot_dataset"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["lerobot.common.datasets.lerobot_dataset"].LeRobotDataset = object
    path = config.RUNTIME / "examples/anytouch2/convert_univtac_sim_delta_eef_to_lerobot.py"
    spec = importlib.util.spec_from_file_location("training_converter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_episode(path, instruction, offset):
    image = np.full((8, 10, 3), (30, 90, 150), dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    with h5py.File(path, "w") as episode:
        episode.attrs["instruction"] = instruction
        episode["step"] = [0, 2, 4]
        episode["embodiment/joint"] = np.arange(27, dtype=np.float32).reshape(3, 9) + offset
        for key in ("observation/head/rgb", "observation/wrist/rgb", "tactile/left_tactile/rgb_marker",
                    "tactile/right_tactile/rgb_marker"):
            episode[key] = np.asarray([encoded.tobytes()] * 3, dtype=f"S{len(encoded)}")


def test_conversion_preserves_episodes_actions_and_images(tmp_path, monkeypatch):
    converter = load_converter(monkeypatch)
    class Dataset:
        def __init__(self):
            self.pending = []
            self.episodes = []
        def add_frame(self, frame):
            self.pending.append(frame)
        def save_episode(self):
            self.episodes.append(self.pending)
            self.pending = []
    dataset = Dataset()
    args = types.SimpleNamespace(image_size=(10, 8), control_mode="abs_joint", task_prompt=None,
                                 frame_stride=1, target_offset=1, gripper_mode="left",
                                 read_retries=0, read_retry_delay=0)
    for i in range(2):
        path = tmp_path / f"{i}.hdf5"
        write_episode(path, f"task {i}", i * 100)
        assert converter.convert_episode(dataset, path, args) == 3
    assert len(dataset.episodes) == 2
    for i, episode in enumerate(dataset.episodes):
        assert all(frame["task"] == f"task {i}" for frame in episode)
        expected = np.arange(27, dtype=np.float32).reshape(3, 9) + i * 100
        for t, frame in enumerate(episode):
            np.testing.assert_array_equal(frame["state"], expected[t, :8])
            np.testing.assert_array_equal(frame["actions"], expected[min(t + 1, 2), :8])
            for key in ("image", "wrist_image", "left_tactile_image", "right_tactile_image"):
                assert frame[key].shape == (8, 10, 3)
                np.testing.assert_allclose(frame[key][0, 0], (30, 90, 150), atol=3)


def test_tasks_and_cli_defaults():
    assert len(config.TASKS) == 11
    for task in config.TASKS:
        assert config.steps_for(task) == (20_000 if task == "in_b_v2" else 10_000)
        for model in config.MODELS:
            output = subprocess.run([sys.executable, str(TRAINING / "run.py"), "show", "--task", task,
                                     "--model", model], capture_output=True, text=True, check=True)
            assert '"batch_size": 64' in output.stdout
            assert '"action_horizon": 50' in output.stdout
    with pytest.raises(ValueError):
        config.steps_for("unknown")


def test_missing_inputs_fail_before_loading_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "DATA", tmp_path / "data")
    monkeypatch.setattr(run, "WEIGHTS", tmp_path / "weights")
    monkeypatch.setattr(sys, "argv", ["run.py", "train", "--task", "ga"])
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        run.main()
    (tmp_path / "data/lerobot/ga").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="norm_stats.json"):
        run.main()
    stats = tmp_path / "data/stats/openvtla/ga/norm_stats.json"
    stats.parent.mkdir(parents=True)
    stats.touch()
    with pytest.raises(FileNotFoundError, match="checkpoint-4frames.pth"):
        run.main()


def test_relative_action_round_trip():
    # Load only the two numerical transforms so this check does not need JAX.
    import ast
    import dataclasses
    source = (config.RUNTIME / "src/openpi/transforms.py").read_text()
    tree = ast.parse(source)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name in {"DeltaActions", "AbsoluteActions"}]
    namespace = {"dataclasses": dataclasses, "np": np, "DataTransformFn": object}
    code = compile(ast.fix_missing_locations(ast.Module(body=classes, type_ignores=[])), "transforms.py", "exec",
                   flags=__import__("__future__").annotations.compiler_flag)
    exec(code, namespace)
    state = np.arange(8, dtype=np.float32)
    targets = np.stack([state + 1, state + 3])
    data = {"state": state, "actions": targets.copy()}
    mask = (True,) * 7 + (False,)
    data = namespace["DeltaActions"](mask)(data)
    np.testing.assert_array_equal(data["actions"][:, :7], [[1] * 7, [3] * 7])
    np.testing.assert_array_equal(data["actions"][:, 7], targets[:, 7])
    data = namespace["AbsoluteActions"](mask)(data)
    np.testing.assert_array_equal(data["actions"], targets)


def test_prefix_order_uses_shared_tactile_encoder():
    import ast
    import math
    torch = pytest.importorskip("torch")
    source = (config.RUNTIME / "src/openpi/models_pytorch/pi0_pytorch.py").read_text()
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PI0Pytorch")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "embed_prefix")
    namespace = {"torch": torch, "math": math}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), "pi0_pytorch.py", "exec"), namespace)
    model = types.SimpleNamespace(
        tactile_image_keys={"left", "right"}, tactile_encoder_type="anytouch2",
        anytouch2_encoder=lambda image: torch.full((1, 36, 4), image),
        paligemma_with_expert=types.SimpleNamespace(
            embed_image=lambda image: torch.full((1, 2, 4), image),
            embed_language_tokens=lambda _: torch.full((1, 1, 4), 5.0)),
        _apply_checkpoint=lambda fn, *args: fn(*args))
    keys = ("left", "right", "head", "wrist")
    embeddings, padding, attention = namespace["embed_prefix"](
        model, list(zip(keys, (1.0, 2.0, 3.0, 4.0))), {key: torch.tensor([True]) for key in keys},
        torch.zeros(1, 1, dtype=torch.long), torch.ones(1, 1, dtype=torch.bool))
    assert embeddings.shape == (1, 77, 4)
    assert embeddings[0, :, 0].tolist() == [1.0] * 36 + [2.0] * 36 + [3.0] * 2 + [4.0] * 2 + [10.0]
    assert padding.all() and not attention.any()


def test_build_config_for_both_models(monkeypatch):
    import dataclasses

    @dataclasses.dataclass
    class Model:
        image_keys: tuple = ("left_tactile_0_rgb", "right_tactile_0_rgb", "base_0_rgb", "wrist_0_rgb")
        tactile_encoder_type: str = "anytouch2"
        anytouch_trainable: bool = True
        action_horizon: int = 50

    @dataclasses.dataclass
    class Train:
        model: object = dataclasses.field(default_factory=Model)
        data: object = None
        name: str = ""
        assets_base_dir: str = ""
        checkpoint_base_dir: str = ""
        exp_name: str = ""
        batch_size: int = 0
        num_workers: int = 0
        num_train_steps: int = 0
        resume: bool = False
        wandb_enabled: bool = False

    cfg = types.SimpleNamespace(
        make_anytouch2_config=lambda *args, **kwargs: Train(),
        LeRobotSimVisionDataConfig=lambda **kwargs: types.SimpleNamespace(**kwargs),
        DataConfig=lambda **kwargs: types.SimpleNamespace(**kwargs),
    )
    monkeypatch.setitem(sys.modules, "openpi", types.ModuleType("openpi"))
    policies = types.ModuleType("openpi.policies")
    policies.sim_policy = types.SimpleNamespace(VISION_IMAGE_KEYS=("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"))
    monkeypatch.setitem(sys.modules, "openpi.policies", policies)
    training = types.ModuleType("openpi.training")
    training.config = cfg
    monkeypatch.setitem(sys.modules, "openpi.training", training)
    for task in config.TASKS:
        for model in config.MODELS:
            args = types.SimpleNamespace(task=task, model=model, exp_name="run1", num_workers=2, resume=False)
            result = run.build_config(args)
            assert result.num_train_steps == config.steps_for(task)
            assert result.batch_size == 64
            assert result.exp_name == f"{task}/run1"
            assert result.model.action_horizon == 50
            assert result.model.tactile_encoder_type == ("siglip" if model == "pi05" else "anytouch2")
            if model == "pi05":
                assert result.data.delta_joint_actions
                assert result.data.repo_id == task
            else:
                assert result.model.anytouch_trainable


def test_lerobot_history_stays_in_episode():
    torch = pytest.importorskip("torch")
    module = pytest.importorskip("lerobot.common.datasets.lerobot_dataset")
    dataset = types.SimpleNamespace(
        episode_data_index={"from": torch.tensor([0, 3]), "to": torch.tensor([3, 6])},
        delta_indices={"left_tactile_image": [-6, -4, -2, 0]},
    )
    indices, padding = module.LeRobotDataset._get_query_indices(dataset, 3, 1)
    assert indices["left_tactile_image"] == [3, 3, 3, 3]
    assert padding["left_tactile_image_is_pad"].tolist() == [True, True, True, False]
