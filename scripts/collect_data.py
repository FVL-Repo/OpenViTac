import os
import sys
import time
import yaml
import json
import torch
import argparse
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Literal

sys.path.append('.')

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Collect data"
)
parser.add_argument(
    "task",
    type=str,
    help="Task file name",
)
parser.add_argument(
    "config",
    type=str,
    help="Config file name",
)
parser.add_argument(
    "--episode_num",
    type=int,
    default=-1,
)
parser.add_argument(
    "--start_seed",
    type=int,
    default=-1,
)
parser.add_argument(
    "--max_seed",
    type=int,
    default=-1,
)
parser.add_argument(
    "--gpu",
    type=str,
    default=None,
)
parser.add_argument(
    "--background",
    type=str,
    default=None,
    help="Override env_cfg.background. Use base0/base1/... or an .exr filename under assets/scene.",
)
parser.add_argument(
    "--target_block",
    type=str,
    default=None,
    help="Override env_cfg.target_block when the selected task supports it.",
)
parser.add_argument(
    "--block_base_pose_indices",
    type=str,
    default=None,
    help=(
        "Override env_cfg.block_base_pose_indices when supported. "
        "Accepts comma-separated values such as 0,1,4 or a YAML/JSON list in the config."
    ),
)
parser.add_argument(
    "--target_cup",
    type=str,
    default=None,
    help="Override env_cfg.target_cup when the selected task supports it.",
)
parser.add_argument(
    "--reference_cup",
    type=str,
    default=None,
    help="Override env_cfg.reference_cup when the selected task supports it.",
)
parser.add_argument(
    "--placement_side",
    type=str,
    default=None,
    help="Override env_cfg.placement_side when the selected task supports it.",
)
parser.add_argument(
    "--cup_base_pose_indices",
    type=str,
    default=None,
    help=(
        "Override env_cfg.cup_base_pose_indices when supported. "
        "Accepts comma-separated values such as 0,1,2 or a YAML/JSON list in the config."
    ),
)
parser.add_argument(
    "--target_area",
    type=str,
    default=None,
    help="Override env_cfg.target_area when the selected task supports it.",
)
parser.add_argument(
    "--frame_order",
    type=str,
    default=None,
    help="Override env_cfg.frame_order when the selected task supports it.",
)
parser.add_argument(
    "--rough_block_side",
    type=str,
    default=None,
    help="Override env_cfg.rough_block_side when the selected task supports it.",
)
parser.add_argument(
    "--initial_grasp_side",
    type=str,
    default=None,
    help="Override env_cfg.initial_grasp_side when the selected task supports it.",
)
parser.add_argument(
    "--weight_label",
    type=str,
    default=None,
    choices=("random", "light", "heavy"),
    help="Override env_cfg.weight_label when the selected task supports it.",
)
parser.add_argument(
    "--roughness_label",
    type=str,
    default=None,
    choices=("random", "smooth", "rough"),
    help="Override env_cfg.roughness_label when the selected task supports it.",
)
parser.add_argument(
    "--hardness_label",
    type=str,
    default=None,
    choices=("random", "soft", "hard"),
    help="Override env_cfg.hardness_label when the selected task supports it.",
)
parser.add_argument(
    "--empty_can",
    type=str,
    default=None,
    choices=("random", "coke", "fanta", "7up", "pepsi"),
    help="Override env_cfg.empty_can when the selected task supports it.",
)

args_cli = parser.parse_args()
if args_cli.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args_cli.gpu

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli.enable_cameras = True
args_cli.num_envs = 1

def get_config(file, default_root:Path, type:Literal['yaml', 'json']):
    if type == 'yaml':
        if file.endswith('.yml') or file.endswith('.yaml'):
            file = Path(file)
        else:
            file = default_root / f'{file}.yml'
        with open(file, 'r') as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        return config, file
    else:
        if file.endswith('.json'):
            file = Path(file)
        else:
            file = default_root / f'{file}.json'
        with open(file, 'r') as f:
            config = json.load(f)
        return config, file

def parse_int_tuple(value, name: str):
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.startswith("["):
            value = json.loads(text)
        else:
            value = text.replace(",", " ").split()
    try:
        parsed = tuple(int(item) for item in value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a list or comma-separated string, got {value!r}") from exc
    except ValueError as exc:
        raise ValueError(f"{name} must contain only integers, got {value!r}") from exc
    if not parsed:
        raise ValueError(f"{name} must not be empty")
    return parsed

task_config, task_config_file = get_config(
    args_cli.config, 
    default_root=Path(__file__).parent.parent / 'task_config', 
    type='yaml'
)

if task_config.get('render_frequency', 1) == 0:
    args_cli.livestream = 2

is_force_task = args_cli.task in ("grasp_fragile_chip", "bulb_tightening", "tension_strap", "wipe_vase")
if is_force_task:
    from envs._force_task_utils import (
        dispatch_force_task_seeds, launch_force_task_app,
        prepare_force_task_config, run_force_task_episode,
    )
    exit_code = dispatch_force_task_seeds(
        "collect", args_cli, config=task_config, config_path=task_config_file)
    if exit_code is not None:
        raise SystemExit(exit_code)

# launch omniverse app, must done before importing anything from omni.isaac
app_launcher = launch_force_task_app(AppLauncher, args_cli) if is_force_task else AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib
if TYPE_CHECKING:
    from envs._base_task import BaseTask, BaseTaskCfg

log_path = Path('./log')
def log(msg):
    global log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    msg = f"[{time.strftime(r'%Y-%m-%d %H:%M:%S')}] {msg}"
    with open(log_path, 'a') as f:
        f.write(msg + '\n')
    print(msg)

def run(task: 'BaseTask', episode_num, use_seed, start_seed, max_seed):
    if is_force_task:
        try:
            return run_force_task_episode(task, seed=start_seed)
        finally:
            task.close()
            simulation_app.close()

    suc_num, seed = 0, 0
    suc_map = []
    
    if start_seed != -1:
        seed = start_seed
        log(f"Starting from seed {seed}.")
    elif use_seed:
        suc_map_path = task.save_root / 'suc_map.txt'
        if suc_map_path.exists():
            with open(suc_map_path, 'r') as f:
                suc_map = f.read().strip().split(' ')
            suc_num = sum([1 for s in suc_map if s == '1'])
            seed = len(suc_map)
            log(f"Use seed with {suc_num} successful episodes. Starting from seed {seed}.")

    mean_steps = 0.0
    while suc_num < episode_num and (max_seed == -1 or seed <= max_seed):
        try:
            start_t = time.perf_counter()
            task.reset(seed=seed)
            task.play_once()
            cost_t = time.perf_counter() - start_t
        except Exception as e:
            log(f"[{suc_num:<3d}] Seed {seed} failed with error: {traceback.format_exc()}")
            suc_map.append('0')
            task.clean_cache(mean_steps=mean_steps, result='error')
        else:
            if task.plan_success and task.check_success() and not task.check_early_stop():
                task.save_to_hdf5()
                log(f"[{suc_num:<3d}] Seed {seed} success in {cost_t:.2f} s.\n"
                    f"steps: {task.step_count:<5d}, save frames: {task.save_count:<5d}.\n")
                suc_num += 1
                suc_map.append('1')
                if mean_steps > 0: 
                    mean_steps = ((suc_num - 1) * mean_steps + task.step_count) / suc_num
                else:
                    mean_steps = task.step_count
                task.clean_cache(mean_steps=mean_steps, result='success')
            else:
                log(f"[{suc_num:<3d}] Seed {seed} failed in {cost_t:.2f} s.\n"
                    f"Plan {task.plan_success}, Check {task.check_success()}")
                suc_map.append('0')
                task.clean_cache(mean_steps=mean_steps, result='fail')
        
        with open(task.save_root / 'suc_map.txt', 'w') as f:
            f.write(' '.join([s for s in suc_map]))
        
        seed += 1
    
    log(f'Complete collection, success rate: {suc_num}/{seed} ({(suc_num / seed) * 100:.2f}%)')

    task.close()
    simulation_app.close()

def main():
    global args_cli, task_config, task_config_file, log_path
    task_file_name = args_cli.task

    episode_num = task_config.get("episode_num", -1)
    if args_cli.episode_num != -1:
        episode_num = args_cli.episode_num
    start_seed = task_config.get("start_seed", -1)
    if args_cli.start_seed != -1:
        start_seed = args_cli.start_seed
    max_seed = task_config.get("max_seed", -1)
    if args_cli.max_seed != -1:
        max_seed = args_cli.max_seed
    
    task_config.update({
        "episode_num": episode_num,
        "start_seed": start_seed,
        "max_seed": max_seed,
    })
    if args_cli.background is not None:
        task_config["background"] = args_cli.background

    task_module = importlib.import_module(f"envs.{task_file_name}")
    env_cfg:'BaseTaskCfg' = task_module.TaskCfg()
    if "background" in task_config:
        env_cfg.background = str(task_config["background"])
    if hasattr(env_cfg, "target_block"):
        target_block = args_cli.target_block
        if target_block is None:
            target_block = task_config.get("target_block", None)
        if target_block is not None:
            env_cfg.target_block = str(target_block)
            task_config["target_block"] = env_cfg.target_block
    if hasattr(env_cfg, "block_base_pose_indices"):
        pose_indices = args_cli.block_base_pose_indices
        if pose_indices is None:
            pose_indices = task_config.get("block_base_pose_indices", None)
        pose_indices = parse_int_tuple(pose_indices, "block_base_pose_indices")
        if pose_indices is not None:
            env_cfg.block_base_pose_indices = pose_indices
            task_config["block_base_pose_indices"] = list(pose_indices)
    for key in (
        "target_cup",
        "reference_cup",
        "placement_side",
        "target_area",
        "frame_order",
        "rough_block_side",
        "initial_grasp_side",
        "weight_label",
        "roughness_label",
        "hardness_label",
        "empty_can",
    ):
        if hasattr(env_cfg, key):
            value = getattr(args_cli, key)
            if value is None:
                value = task_config.get(key, None)
            if value is not None:
                setattr(env_cfg, key, str(value))
                task_config[key] = str(value)
    if hasattr(env_cfg, "cup_base_pose_indices"):
        pose_indices = args_cli.cup_base_pose_indices
        if pose_indices is None:
            pose_indices = task_config.get("cup_base_pose_indices", None)
        pose_indices = parse_int_tuple(pose_indices, "cup_base_pose_indices")
        if pose_indices is not None:
            env_cfg.cup_base_pose_indices = pose_indices
            task_config["cup_base_pose_indices"] = list(pose_indices)
    env_cfg.tactile_sensor_type = task_config.get('sensor_type', 'gsmini')
    env_cfg.dense_gelpad = bool(task_config.get('dense_gelpad', env_cfg.dense_gelpad))
    env_cfg.force_field_grid = tuple(task_config.get('force_field_grid', env_cfg.force_field_grid))
    if "save_dir_exact" in task_config:
        env_cfg.save_dir = Path(task_config["save_dir_exact"])
    else:
        env_cfg.save_dir = Path(task_config.get("save_dir", "./data")) / task_file_name / task_config_file.stem
    env_cfg.decimation = task_config.get("decimation", env_cfg.decimation)
    env_cfg.save_frequency = task_config.get("save_frequency", env_cfg.save_frequency)
    env_cfg.video_frequency = task_config.get("video_frequency", env_cfg.video_frequency)
    env_cfg.render_frequency = task_config.get("render_frequency", env_cfg.render_frequency)
    if "reset_time_limit" in task_config:
        env_cfg.reset_time_limit = float(task_config["reset_time_limit"])
    for key in (
        "reset_first_frame_steps",
        "reset_after_actor_steps",
        "reset_final_steps",
        "reset_render_warmup_steps",
    ):
        if key in task_config:
            setattr(env_cfg, key, int(task_config[key]))
    if "video_size" in task_config:
        env_cfg.video_size = tuple(task_config["video_size"])
    env_cfg.obs_data_type = task_config.get("observations", {})
    if task_config.get("gel_particle", False):
        tactile_obs = env_cfg.obs_data_type.setdefault("tactile", [])
        if "gel_particle" not in tactile_obs:
            tactile_obs.append("gel_particle")
    env_cfg.random_texture = task_config.get("random_texture", False)
    env_cfg.save_pre_move = task_config.get("save_pre_move", getattr(env_cfg, "save_pre_move", False))
    env_cfg.tactile_video_key = task_config.get("tactile_video_key", env_cfg.tactile_video_key)
    if "use_adaptive_grasp" in task_config:
        env_cfg.use_adaptive_grasp = bool(task_config["use_adaptive_grasp"])
    if "adaptive_grasp_depth_threshold" in task_config:
        env_cfg.adaptive_grasp_depth_threshold = float(task_config["adaptive_grasp_depth_threshold"])
    xense_tuning_types = {
        "xense_usb_close_percent": float,
        "xense_half_cylinder_close_percent": float,
        "xense_insert_half_cylinder_close_percent": float,
        "xense_cube_close_percent": float,
        "xense_cup_close_percent": float,
        "xense_cup_min_principal_ratio": float,
        "xense_cup_max_nonrigid_error": float,
        "xense_pour_cup_close_percent": float,
        "xense_pour_ball_friction_ratio": float,
        "xense_pour_grip_friction_ratio": float,
        "xense_pour_wrist_angle_deg": float,
        "xense_pour_wrist_steps": int,
        "xense_pour_wrist_translation_x": float,
        "xense_pour_wrist_translation_y": float,
        "xense_pour_wrist_translation_z": float,
        "xense_pour_actor_tilt_deg": float,
        "xense_pour_actor_tilt_axis_x": float,
        "xense_pour_actor_tilt_axis_y": float,
        "xense_pour_actor_tilt_axis_z": float,
        "xense_pour_carry_segments": int,
        "xense_pour_carry_settle_steps": int,
        "xense_pour_hold_actor_during_carry": bool,
        "xense_pour_target_y_offset": float,
        "xense_pour_target_z_offset": float,
        "xense_pour_release_lift": float,
        "xense_pour_release_snap_angle_deg": float,
        "xense_pour_release_snap_steps": int,
        "xense_pour_release_snap_cycles": int,
        "xense_pour_fix_cup_during_release": bool,
        "xense_pour_release_retract_x": float,
        "xense_pour_release_carry_y": float,
        "xense_drawer_close_percent": float,
        "xense_gear_close_percent": float,
        "xense_half_cylinder_grasp_height_bias": float,
        "xense_insert_half_cylinder_grasp_height_bias": float,
        "xense_cube_grasp_height_bias": float,
        "xense_cup_grasp_height_bias": float,
        "xense_pour_cup_grasp_height_bias": float,
        "xense_pour_cup_grasp_world_x_bias": float,
        "xense_drawer_grasp_z_bias": float,
        "xense_half_cylinder_grasp_world_y_bias": float,
        "xense_insert_half_cylinder_grasp_world_y_bias": float,
        "xense_cube_grasp_world_y_bias": float,
        "xense_initial_settle_steps": int,
        "xense_half_cylinder_initial_settle_steps": int,
        "xense_cup_initial_settle_steps": int,
        "xense_pour_initial_settle_steps": int,
        "xense_drawer_initial_settle_steps": int,
        "xense_insert_half_cylinder_initial_settle_steps": int,
        "xense_cube_initial_settle_steps": int,
        "xense_gear_grasp_world_y_bias": float,
        "xense_carry_time_dilation": float,
        "xense_carry_segments": int,
        "xense_carry_max_step": float,
        "xense_post_close_settle_steps": int,
        "xense_adaptive_grasp_max_steps": int,
        "xense_adaptive_grasp_tail_steps": int,
        "xense_adaptive_grasp_check_interval": int,
        "xense_adaptive_grasp_target_tolerance": float,
        "xense_adaptive_grasp_hold_margin": float,
        "xense_adaptive_grasp_hold_velocity": float,
        "xense_usb_post_close_settle_steps": int,
        "xense_adaptive_grasp_min_steps_before_contact": int,
        "xense_adaptive_grasp_min_travel": float,
        "xense_adaptive_grasp_require_both_contacts": bool,
        "xense_usb_adaptive_grasp_require_both_contacts": bool,
        "xense_half_cylinder_adaptive_grasp_require_both_contacts": bool,
        "xense_insert_half_cylinder_adaptive_grasp_require_both_contacts": bool,
        "xense_cube_adaptive_grasp_require_both_contacts": bool,
        "xense_cup_adaptive_grasp_require_both_contacts": bool,
        "xense_pour_cup_adaptive_grasp_require_both_contacts": bool,
        "xense_drawer_adaptive_grasp_require_both_contacts": bool,
        "xense_gear_adaptive_grasp_require_both_contacts": bool,
        "xense_usb_adaptive_grasp_depth_threshold": float,
        "xense_half_cylinder_adaptive_grasp_depth_threshold": float,
        "xense_insert_half_cylinder_adaptive_grasp_depth_threshold": float,
        "xense_cube_adaptive_grasp_depth_threshold": float,
        "xense_cup_adaptive_grasp_depth_threshold": float,
        "xense_pour_cup_adaptive_grasp_depth_threshold": float,
        "xense_drawer_adaptive_grasp_depth_threshold": float,
        "xense_gear_adaptive_grasp_depth_threshold": float,
    }
    for key, value_type in xense_tuning_types.items():
        if key in task_config:
            setattr(env_cfg, key, value_type(task_config[key]))
    env_cfg.scene.num_envs = 1
    
    if is_force_task:
        env_cfg.save_dir = Path(os.environ["OPENVITAC_FORCE_OUTPUT"])
        prepare_force_task_config(env_cfg, task_config, task_config_file, seed=start_seed)
    init_start = time.perf_counter()
    task:'BaseTask' = task_module.Task(env_cfg, mode='collect')
    init_cost = time.perf_counter() - init_start
    
    log_path = task.save_root / f"{time.strftime(r'%Y-%m-%d_%H:%M:%S')}.log"
    log(f"Task Name: {task_file_name}")
    log(f"Config Name: {task_config_file.stem}")
    log(f"Task Config: \n{json.dumps(task_config, ensure_ascii=False, indent=4)}\n{'-' * 20}\n")
    log(f"Env Config: \n{env_cfg}\n{'-' * 20}\n")
    log(f"Init cost {init_cost:.2f} seconds, devices: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    run(
        task,
        episode_num=episode_num,
        use_seed=task_config.get("use_seed", True),
        start_seed=start_seed,
        max_seed=max_seed,
    )

if __name__ == "__main__":
    main()
