from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import yaml


def _task_config_path(task_config: str | Path) -> Path:
    path = Path(task_config)
    if path.suffix in (".yml", ".yaml"):
        return path
    return Path("task_config") / f"{task_config}.yml"


def _set_if_present(target: object, name: str, value: Any) -> None:
    if hasattr(target, name):
        setattr(target, name, value)


def _resolve_task_type(task_name: str, task_module, task_variant: str | None):
    task_type = task_module.Task
    if task_variant is None:
        return task_type
    if task_variant not in {"rl", "rfcl"}:
        raise ValueError(f"Unknown task variant: {task_variant!r}")
    if task_name != "insert_USB":
        raise ValueError(f"RL task variant is not implemented for {task_name!r}")

    from policy.RL.tasks.insert_usb import (
        build_insert_usb_rfcl_task_type,
        build_insert_usb_rl_task_type,
    )

    builder = (
        build_insert_usb_rfcl_task_type
        if task_variant == "rfcl"
        else build_insert_usb_rl_task_type
    )
    return builder(task_type, task_module)


def create_task(
    task_name: str,
    task_config: str | Path,
    save_dir: str | Path,
    *,
    video_frequency: int = 0,
    step_limit: int = 120,
    mode: str = "eval",
    save_pre_move: bool | None = None,
    insert_usb_fixed_target_slot: bool = False,
    task_variant: str | None = None,
    device: str | None = None,
):
    config_path = _task_config_path(task_config)
    with config_path.open("r", encoding="utf-8") as config_file:
        source = yaml.load(config_file, Loader=yaml.FullLoader) or {}

    task_module = importlib.import_module(f"envs.{task_name}")
    env_config = task_module.TaskCfg()

    env_config.tactile_sensor_type = source.get(
        "sensor_type",
        env_config.tactile_sensor_type,
    )
    env_config.save_dir = Path(save_dir)
    env_config.decimation = int(source.get("decimation", env_config.decimation))
    env_config.save_frequency = int(
        source.get("save_frequency", env_config.save_frequency)
    )
    env_config.video_frequency = int(video_frequency)
    env_config.render_frequency = 0
    env_config.obs_data_type = source.get("observations", {})
    env_config.random_texture = bool(source.get("random_texture", False))
    env_config.step_lim = int(step_limit)
    env_config.scene.num_envs = 1
    env_config.eval_start_delay_steps = 0
    effective_task_variant = task_variant
    if insert_usb_fixed_target_slot and effective_task_variant is None:
        effective_task_variant = "rl"
    if insert_usb_fixed_target_slot and task_name != "insert_USB":
        raise ValueError("Fixed target slot is only supported for insert_USB")

    if device is not None:
        _set_if_present(env_config.sim, "device", str(device))

    if save_pre_move is None:
        save_pre_move = source.get("save_pre_move", False)
    env_config.save_pre_move = bool(save_pre_move)
    env_config.skip_pre_move = bool(
        source.get("skip_pre_move", getattr(env_config, "skip_pre_move", False))
    )
    env_config.eval_start_delay_steps = int(source.get("eval_start_delay_steps", 0))

    for name, caster in (
        ("dense_gelpad", bool),
        ("use_adaptive_grasp", bool),
        ("adaptive_grasp_depth_threshold", float),
        ("tactile_video_key", str),
        ("reset_after_actor_steps", int),
        ("reset_time_limit", float),
    ):
        if name in source and hasattr(env_config, name):
            setattr(env_config, name, caster(source[name]))

    for name in ("block_base_pose_indices", "cup_base_pose_indices"):
        if name in source and hasattr(env_config, name):
            setattr(env_config, name, tuple(int(index) for index in source[name]))

    for name in ("rough_block_side", "initial_grasp_side"):
        if name in source and hasattr(env_config, name):
            setattr(env_config, name, str(source[name]))

    task_type = _resolve_task_type(task_name, task_module, effective_task_variant)
    task_kwargs = {}
    if effective_task_variant in {"rl", "rfcl"}:
        task_kwargs["fixed_target_slot"] = bool(insert_usb_fixed_target_slot)
    return task_type(env_config, mode=mode, **task_kwargs)
