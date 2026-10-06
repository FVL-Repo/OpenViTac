import os
import sys
import time
import json
import yaml
import argparse
import importlib
import traceback
import io
from tqdm import tqdm
from queue import Empty
from pathlib import Path
from typing import Literal, TYPE_CHECKING
from multiprocessing import Process, Queue, Manager, Event, current_process

if TYPE_CHECKING:
    from envs._base_task import BaseTask, BaseTaskCfg


def get_config(file, default_root: Path, type: Literal['yaml', 'json']):
    if type == 'yaml':
        file = Path(file) if (file.endswith('.yml') or file.endswith('.yaml')) else (default_root / f'{file}.yml')
        with open(file, 'r') as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        return config, file
    else:
        file = Path(file) if file.endswith('.json') else (default_root / f'{file}.json')
        with open(file, 'r') as f:
            config = json.load(f)
        return config, file


def split_devices(cuda_visible_devices: str, workers: int):
    devices = [d.strip() for d in cuda_visible_devices.split(',') if d.strip() != '']
    if not devices:
        return [['']] * workers
    assignment = []
    for i in range(workers):
        assignment.append([devices[i % len(devices)]])
    return assignment


def worker_run(task_config, task_file_name, base_save_dir: Path, seed_q: Queue,
               progress, stop_event: Event, log_file: Path, device_list,
               status_dict, result_q: Queue):
    # Per-process env: assign CUDA devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(device_list) if device_list else ''

    # Redirect stdout/stderr to unified log
    log_fd = os.open(str(log_file), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    sys.path.insert(0, '.')

    # Create Isaac app inside worker
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(parser)
    app_args = parser.parse_args([])
    app_args.enable_cameras = True
    app_args.num_envs = 1
    if task_config.get('render_frequency', 1) == 0:
        app_args.livestream = 2

    app_launcher = AppLauncher(app_args)
    simulation_app = app_launcher.app

    try:
        task_module = importlib.import_module(f"envs.{task_file_name}")

        env_cfg: 'BaseTaskCfg' = task_module.TaskCfg()
        worker_id = current_process().name.split('-')[-1]
        env_cfg.save_dir = base_save_dir
        if "background" in task_config:
            env_cfg.background = str(task_config["background"])
        env_cfg.tactile_sensor_type = task_config.get('sensor_type', 'gsmini')
        env_cfg.dense_gelpad = bool(task_config.get('dense_gelpad', env_cfg.dense_gelpad))
        env_cfg.force_field_grid = tuple(task_config.get('force_field_grid', env_cfg.force_field_grid))
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
        # Device routing by CUDA env

        init_start = time.perf_counter()
        task: 'BaseTask' = task_module.Task(env_cfg, mode='collect')

        # Override _step_callback to update status_dict
        original_step_callback = task._step_callback
        def step_callback(status):
            nonlocal status_dict
            status_dict['steps'] = status['step_count']
            status_dict['atom'] = f"{status['atom_id']}:{status['atom_tag']}"
            return original_step_callback(status)
        task._step_callback = step_callback

        init_cost = time.perf_counter() - init_start
        print(f"[Worker {worker_id}] Task init in {init_cost:.2f}s; save_dir={base_save_dir}")

        mean_steps = 0.0
        status_dict['state'] = 'idle'
        status_dict['current_seed'] = None
        status_dict['last_result'] = '-'
        status_dict['last_cost'] = None
        status_dict['steps'] = None
        status_dict['save_count'] = None
        status_dict['attempts'] = 0
        status_dict['succ'] = 0
        status_dict['errors'] = 0
        while not stop_event.is_set():
            try:
                seed = seed_q.get(timeout=1.0)
            except Exception:
                continue
            if seed is None:
                break

            try:
                status_dict['state'] = 'running'
                status_dict['current_seed'] = seed
                start_t = time.perf_counter()
                task.reset(seed=seed)
                task.play_once()
                cost_t = time.perf_counter() - start_t
            except Exception:
                progress['errors'] += 1
                progress['attempts'] += 1
                status_dict['attempts'] += 1
                status_dict['errors'] += 1
                status_dict['last_result'] = 'error'
                status_dict['last_cost'] = None
                status_dict['steps'] = None
                status_dict['save_count'] = None
                status_dict['current_seed'] = None
                status_dict['state'] = 'idle'
                print(f"[Worker {worker_id}] Seed {seed} error:\n{traceback.format_exc()}")
                result_q.put({
                    'worker': worker_id,
                    'seed': seed,
                    'result': 'error',
                    'cost': None,
                    'steps': None,
                    'save_count': None,
                    'plan_success': False,
                    'check_success': False,
                    'traceback': traceback.format_exc(),
                })
                task.clean_cache(mean_steps=mean_steps, result='error')
                continue

            if task.plan_success and task.check_success() and not task.check_early_stop():
                task.save_to_hdf5()
                progress['succ'] += 1
                progress['done'] += 1  # done counts successful episodes
                progress['attempts'] += 1
                status_dict['attempts'] += 1
                status_dict['succ'] += 1
                status_dict['last_result'] = 'success'
                status_dict['last_cost'] = cost_t
                status_dict['steps'] = task.step_count
                status_dict['save_count'] = task.save_count
                status_dict['current_seed'] = None
                status_dict['state'] = 'idle'
                mean_steps = ((progress['succ'] - 1) * mean_steps + task.step_count) / progress['succ'] if progress['succ'] > 1 else task.step_count
                task.clean_cache(mean_steps=mean_steps)
                print(f"[Worker {worker_id}] Seed {seed} success in {cost_t:.2f}s; steps {task.step_count}, save frames {task.save_count}")
                result_q.put({
                    'worker': worker_id,
                    'seed': seed,
                    'result': 'success',
                    'cost': cost_t,
                    'steps': task.step_count,
                    'save_count': task.save_count,
                    'plan_success': True,
                    'check_success': True,
                    'traceback': None,
                })
            else:
                progress['attempts'] += 1
                status_dict['attempts'] += 1
                status_dict['last_result'] = 'fail'
                status_dict['last_cost'] = cost_t
                status_dict['steps'] = task.step_count
                status_dict['save_count'] = task.save_count
                status_dict['current_seed'] = None
                status_dict['state'] = 'idle'
                task.clean_cache(mean_steps=mean_steps, result='fail')
                print(f"[Worker {worker_id}] Seed {seed} fail in {cost_t:.2f}s; plan {task.plan_success}, check {task.check_success()}")
                result_q.put({
                    'worker': worker_id,
                    'seed': seed,
                    'result': 'fail',
                    'cost': cost_t,
                    'steps': task.step_count,
                    'save_count': task.save_count,
                    'plan_success': task.plan_success,
                    'check_success': task.check_success(),
                    'traceback': None,
                })
        try:
            task.close()
        finally:
            simulation_app.close()
            os.close(log_fd)
    except Exception:
        print(f"[Worker {current_process().name}] Exception during worker run:\n{traceback.format_exc()}")
        try:
            simulation_app.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Collect data (multiprocessing)")
    parser.add_argument("task", type=str, help="Task file name")
    parser.add_argument("config", type=str, help="Config file name")
    parser.add_argument("--workers", type=int, default=2, help="Number of worker processes")
    parser.add_argument("--episodes", type=int, default=None, help="Total successful episodes to collect (override config episode_num)")
    parser.add_argument("--gpu", type=str, default=os.environ.get('CUDA_VISIBLE_DEVICES', ''),
                        help="CUDA_VISIBLE_DEVICES list to split among workers")
    parser.add_argument(
        "--background",
        type=str,
        default=None,
        help="Override env_cfg.background. Use base0/base1/... or an .exr filename under assets/scene.",
    )
    args = parser.parse_args()

    task_config, task_config_file = get_config(
        args.config,
        default_root=Path(__file__).parent.parent / 'task_config',
        type='yaml'
    )
    if args.background is not None:
        task_config["background"] = args.background
    target_episodes = args.episodes if args.episodes is not None else task_config.get("episode_num", 10)

    # Base save dir with timestamp; per-worker subfolders appended inside worker
    curr_time = time.strftime(r'%Y-%m-%d_%H:%M:%S')
    base_save_dir = Path(task_config.get("save_dir", "./data")) / args.task / task_config_file.stem
    base_save_dir.mkdir(parents=True, exist_ok=True)
    out_log = base_save_dir / f'{curr_time}.out'
    clean_log = base_save_dir / f'{curr_time}.log'
    out_log.touch()
    clean_log.touch()

    manager = Manager()
    seed_q = Queue()
    progress = manager.dict(done=0, succ=0, errors=0, attempts=0)
    stop_event = Event()
    result_q = Queue()

    def write_out(msg: str):
        with open(out_log, 'a') as f:
            f.write(msg + '\n')

    def write_clean(msg: str):
        stamped = f"[{time.strftime(r'%Y-%m-%d %H:%M:%S')}] {msg}"
        with open(clean_log, 'a') as f:
            f.write(stamped + '\n')

    # Start seed logic: use config start_seed if provided, else 0
    start_seed = task_config.get("start_seed", 0)
    if start_seed is None:
        start_seed = 0
    next_seed = start_seed

    assignments = split_devices(args.gpu, args.workers)

    # Initial run parameters into clean log
    write_clean("Run parameters:")
    write_out("Run parameters:")
    params_json = json.dumps({
        'task': args.task,
        'config_file': str(task_config_file),
        'target_episodes': target_episodes,
        'workers': args.workers,
        'cuda_assignments': assignments,
        'save_dir': str(base_save_dir),
    }, ensure_ascii=False, indent=4)
    write_clean(params_json)
    write_out(params_json)
    write_clean("Task config:")
    write_out("Task config:")
    task_json = json.dumps(task_config, ensure_ascii=False, indent=4)
    write_clean(task_json)
    write_out(task_json)

    workers = []
    worker_status = []
    for w in range(args.workers):
        status_proxy = manager.dict()
        worker_status.append(status_proxy)
        p = Process(
            target=worker_run,
            name=f"Worker-{w+1}",
            args=(task_config, args.task, base_save_dir, seed_q, progress, stop_event, out_log, assignments[w], status_proxy, result_q)
        )
        p.start()
        workers.append(p)

    pbar_io = io.StringIO()
    pbar = tqdm(total=target_episodes, file=pbar_io, ncols=100)

    try:
        last_update = 0
        last_render = 0
        last_block = ""
        for _ in range(args.workers):
            seed_q.put(next_seed)
            next_seed += 1

        while any(p.is_alive() for p in workers):
            # Drain results queue and log clean summaries
            while True:
                try:
                    event = result_q.get_nowait()
                except Empty:
                    break
                else:
                    prefix = f"Worker-{str(event['worker']):2} Seed {event['seed']}"
                    if event['result'] == 'success':
                        write_clean(f"{prefix} success in {event['cost']:.2f}s; steps {event['steps']}, save frames {event['save_count']}")
                    elif event['result'] == 'fail':
                        write_clean(f"{prefix} fail in {event['cost']:.2f}s; plan {event['plan_success']}, check {event['check_success']}")
                    else:
                        write_clean(f"{prefix} error; see out.log for traceback")
                    write_out(f"{prefix} result={event['result']} cost={event['cost']} steps={event['steps']} saves={event['save_count']} plan={event['plan_success']} check={event['check_success']}")

                    if done < target_episodes:
                        for _ in range(args.workers):
                            seed_q.put(next_seed)
                            next_seed += 1
                    else:
                        for _ in range(args.workers):
                            seed_q.put(None)
                        stop_event.set()

            done = progress.get('done', 0)
            pbar.n = done
            pbar.refresh()

            now = time.time()
            if now - last_update > 0.2:
                last_update = now
                succ = progress.get('succ', 0)
                errors = progress.get('errors', 0)
                attempts = progress.get('attempts', 0)
                rate = (succ / attempts * 100) if attempts > 0 else 0
                
                bar_str = pbar_io.getvalue()
                pbar_io.seek(0)
                pbar_io.truncate(0)

                status_lines = [
                    f"[Task: {args.task} (Config: {task_config_file.stem})]",
                    f"[Status] {done}/{target_episodes} collected | succ {succ} | attempts {attempts} | errors {errors} | success_rate {rate:.2f}%"
                ]
                for idx, st in enumerate(worker_status, start=1):
                    seed_str = str(st.get('current_seed', '-')) if st.get('current_seed') is not None else '-'
                    cost_val = st.get('last_cost')
                    cost_str = f"{cost_val:.2f}" if cost_val is not None else '-'
                    step_str = str(st.get('steps', '-')) if st.get('steps') is not None else '-'
                    save_str = str(st.get('save_count', '-')) if st.get('save_count') is not None else '-'
                    atom_str = str(st.get('atom', '-'))
                    status_lines.append(
                        f"  Worker-{idx:2d}: {st.get('state', 'init'):<7} seed={seed_str:<6} "
                        f"last={st.get('last_result', '-'):<7} cost={cost_str:<7} steps={step_str:<6} saves={save_str:<6} atom={atom_str}"
                    )
                block = '\n'.join(status_lines)
                if block != last_block or (now - last_render > 2.0):
                    # Clear screen then print the latest block to keep console clean
                    print("\033[2J\033[H" + bar_str + "\n" + block, end='', flush=True)
                    last_block = block
                    last_render = now
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_event.set()
        for _ in range(args.workers):
            seed_q.put(None)
    finally:
        pbar.close()
        pbar_io.close()
        for p in workers:
            p.join()
        succ = progress.get('succ', 0)
        errors = progress.get('errors', 0)
        attempts = progress.get('attempts', 0)
        rate = (succ / attempts * 100) if attempts > 0 else 0
        print("[Status] All workers finished.")
        final_msg = f"[Final] {succ}/{target_episodes} successful episodes; attempts {attempts}; errors {errors}; success_rate {rate:.2f}%; out_log={out_log}; clean_log={clean_log}"
        print(final_msg)
        write_clean(final_msg)
        write_out(final_msg)


if __name__ == "__main__":
    main()
