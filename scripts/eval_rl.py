from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent))

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate BC, ACT, or direct RL policies."
    )
    parser.add_argument("task_name")
    parser.add_argument("task_config")
    parser.add_argument("bc_checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--algorithm",
        choices=("bc", "act", "sac_init", "sac", "ppo"),
        default="bc",
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--act-checkpoint", type=Path)
    parser.add_argument(
        "--act-config",
        type=Path,
        default=Path("policy/ACT/train_config.yml"),
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--control-mode", choices=("direct",), default="direct")
    parser.add_argument("--reward-mode", choices=("sparse_success", "task"), default="sparse_success")
    parser.add_argument("--handoff-mode", choices=("auto", "none", "insert_usb_collect"), default="auto")
    parser.add_argument(
        "--insert-usb-handoff-distribution",
        choices=(
            "legacy",
            "coarse_preinsert",
            "direct",
            "precontact",
            "diverse_v1",
            "diverse_mild",
            "diverse_tiny",
            "curriculum_v1",
        ),
        default="legacy",
    )
    parser.add_argument(
        "--insert-usb-curriculum-success-thresholds",
        type=int,
        nargs=3,
        default=(50, 100, 150),
        metavar=("STAGE1", "STAGE2", "STAGE3"),
        help=(
            "Successful archive counts that advance curriculum_v1 through "
            "larger xy/z-offset-only stages."
        ),
    )
    parser.add_argument("--insert-usb-xy-quit-threshold", type=float)
    parser.add_argument(
        "--insert-usb-coarse-z-jitter",
        type=float,
        default=0.002,
        help="Coarse-preinsert Z jitter half-range in meters; use 0 to disable.",
    )
    parser.add_argument(
        "--insert-usb-fixed-target-slot",
        action="store_true",
        help=(
            "Keep the target USB slot at its nominal XY position while still "
            "consuming the normal reset random draw."
        ),
    )
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--bc-action-gain", type=float, default=1.0)
    parser.add_argument(
        "--zero-qpos",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Feed a zero qpos vector to the evaluated BC policy.",
    )
    parser.add_argument("--step-limit", type=int, default=80)
    parser.add_argument(
        "--hybrid-motion-plan-z-threshold",
        type=float,
        help=(
            "Switch from the evaluated policy to scripted Insert USB alignment "
            "and insertion once abs(Z error) is at or below this value in meters."
        ),
    )
    parser.add_argument(
        "--hybrid-motion-plan-retreat-clearance",
        type=float,
        help=(
            "Before scripted alignment, retreat vertically to this clearance "
            "above the slot opening, in meters."
        ),
    )
    parser.add_argument("--force-control", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-traces", action="store_true")
    if any(argument in ("-h", "--help") for argument in sys.argv[1:]):
        parser.print_help()
        raise SystemExit(0)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if (
        args.hybrid_motion_plan_z_threshold is not None
        and args.hybrid_motion_plan_z_threshold <= 0.0
    ):
        parser.error("--hybrid-motion-plan-z-threshold must be positive")
    if (
        args.hybrid_motion_plan_retreat_clearance is not None
        and args.hybrid_motion_plan_retreat_clearance < 0.0
    ):
        parser.error(
            "--hybrid-motion-plan-retreat-clearance must be non-negative"
        )
    if (
        args.hybrid_motion_plan_retreat_clearance is not None
        and args.hybrid_motion_plan_z_threshold is None
    ):
        parser.error(
            "--hybrid-motion-plan-retreat-clearance requires "
            "--hybrid-motion-plan-z-threshold"
        )
    args.enable_cameras = True
    args.num_envs = 1
    return args


args = parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from stable_baselines3 import PPO, SAC

from policy.ACT.act_policy import ACT
from policy.RL.gym_env import TactileControlEnv
from policy.RL.sb3_policy import BCGaussianSACActor, BCGaussianSACPolicy
from policy.RL.task_factory import create_task


def main() -> None:
    _ = BCGaussianSACPolicy
    if args.algorithm == "ppo":
        raise NotImplementedError(
            "PPO is not wired to the BC GaussianActor yet. "
            "Evaluate the BC baseline, SAC-init actor, or a SAC model."
        )

    output_dir = args.output_dir.expanduser().resolve() / args.algorithm
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if args.device is not None else "cuda:0"
    task = create_task(
        args.task_name,
        args.task_config,
        save_dir=output_dir / "environment",
        video_frequency=2,
        step_limit=args.step_limit,
        save_pre_move=True,
        insert_usb_fixed_target_slot=args.insert_usb_fixed_target_slot,
        device=device,
    )
    env_control_mode = "bc" if args.algorithm == "bc" else args.control_mode
    environment = TactileControlEnv(
        task,
        args.bc_checkpoint,
        image_size=args.image_size,
        action_repeat=args.action_repeat,
        bc_action_gain=args.bc_action_gain,
        control_mode=env_control_mode,
        reward_mode=args.reward_mode,
        handoff_mode=args.handoff_mode,
        insert_usb_handoff_distribution=args.insert_usb_handoff_distribution,
        insert_usb_coarse_z_jitter=args.insert_usb_coarse_z_jitter,
        zero_qpos=args.zero_qpos,
        insert_usb_curriculum_success_thresholds=tuple(
            args.insert_usb_curriculum_success_thresholds
        ),
        insert_usb_xy_quit_threshold=args.insert_usb_xy_quit_threshold,
        force_control=args.force_control,
        seed=args.start_seed,
        device=device,
    )

    model = None
    actor = None
    act_model = None
    if args.algorithm == "sac_init":
        actor = BCGaussianSACActor(
            observation_space=environment.observation_space,
            action_space=environment.action_space,
            bc_checkpoint=args.bc_checkpoint,
            freeze_encoder=args.freeze_encoder,
            normalize_images=True,
        ).to(device)
        actor.set_training_mode(False)
    elif args.algorithm == "act":
        if args.act_checkpoint is None:
            raise ValueError("--act-checkpoint is required for ACT evaluation")
        act_checkpoint = args.act_checkpoint.expanduser().resolve()
        act_config_path = args.act_config.expanduser().resolve()
        with act_config_path.open("r", encoding="utf-8") as config_file:
            act_config = yaml.load(config_file, Loader=yaml.FullLoader) or {}
        act_config.update(
            {
                "ckpt_dir": str(act_checkpoint.parent),
                "checkpoint_name": act_checkpoint.name,
                "device": device,
                "task_name": f"sim-{args.task_name}-{Path(args.task_config).stem}",
                "seed": args.start_seed,
                "num_epochs": 1,
            }
        )
        act_model = ACT(act_config)
    elif args.algorithm != "bc":
        if args.model_path is None:
            raise ValueError("--model-path is required for SAC/PPO evaluation")
        model_class = SAC if args.algorithm == "sac" else PPO
        model = model_class.load(args.model_path, env=environment, device=device)

    results = []
    for episode_index in range(args.episodes):
        seed = args.start_seed + episode_index
        observation, reset_info = environment.reset(seed=seed)
        initial_hybrid_pose_snapshot = None
        if args.hybrid_motion_plan_z_threshold is not None:
            initial_hybrid_pose_snapshot = (
                environment.capture_insert_usb_pose_snapshot()
            )
        if act_model is not None:
            act_model.reset()
        terminated = False
        truncated = False
        episode_reward = 0.0
        last_info: dict = {}
        hybrid_motion_plan = None
        trace = []
        while not terminated and not truncated:
            act_action = None
            if act_model is not None:
                raw_observation = task._get_observations()
                act_observation = _encode_act_observation(
                    raw_observation,
                    act_model,
                    tactile_key="rgb_marker",
                )
                act_action = act_model.get_action(act_observation).reshape(-1)
                (
                    _raw_observation,
                    task_reward,
                    terminated,
                    truncated,
                    last_info,
                ) = task.env_step(
                    act_action,
                    action_type="qpos",
                    force=args.force_control,
                    action_repeat=args.action_repeat,
                )
                success = bool(last_info.get("success", False))
                reward = 1.0 if success else 0.0
                if args.reward_mode == "task":
                    reward = float(task_reward)
                diagnostics = None
                if hasattr(task, "_get_success_diagnostics"):
                    diagnostics = task._get_success_diagnostics()
                terminal_reason = getattr(task, "terminal_reason", None)
                last_info.update(
                    {
                        "act_action": act_action,
                        "success_diagnostics": diagnostics,
                        "terminal_reason": terminal_reason,
                    }
                )
                observation = environment.encode_observation(_raw_observation)
            elif actor is not None:
                policy_action, _ = actor.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, last_info = environment.step(
                    policy_action
                )
            elif model is None:
                policy_action = np.zeros(environment.action_space.shape, dtype=np.float32)
                observation, reward, terminated, truncated, last_info = environment.step(
                    policy_action
                )
            else:
                policy_action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, last_info = environment.step(
                    policy_action
                )
            diagnostics = last_info.get("success_diagnostics") or {}
            z_error = diagnostics.get("abs_z_error")
            terminal_reason = last_info.get("terminal_reason")
            can_complete_after_step_limit = terminal_reason == "step_limit"
            if (
                args.hybrid_motion_plan_z_threshold is not None
                and hybrid_motion_plan is None
                and not bool(last_info.get("success", False))
                and (not terminated)
                and (not truncated or can_complete_after_step_limit)
                and z_error is not None
                and float(z_error) <= args.hybrid_motion_plan_z_threshold
            ):
                switch_step_info = last_info
                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    last_info,
                ) = environment.complete_insert_usb_with_motion_plan(
                    retreat_clearance=(
                        args.hybrid_motion_plan_retreat_clearance
                    ),
                    initial_pose_snapshot=initial_hybrid_pose_snapshot,
                )
                for key in (
                    "bc_action",
                    "scaled_bc_action",
                    "policy_action",
                    "normalized_action",
                    "target_qpos",
                ):
                    if key in switch_step_info:
                        last_info[key] = switch_step_info[key]
                hybrid_motion_plan = last_info["hybrid_motion_plan"]
            episode_reward += float(reward)
            if args.save_traces:
                diagnostics = last_info.get("success_diagnostics") or {}
                trace_step = {
                    "step": len(trace) + 1,
                    "reward": float(reward),
                    "success": bool(last_info.get("success", False)),
                    "qpos": observation["qpos"].tolist(),
                    "xy_error": diagnostics.get("xy_error"),
                    "abs_z_error": diagnostics.get("abs_z_error"),
                    "tilt_angle_deg": diagnostics.get("tilt_angle_deg"),
                    "rl_xy_out_of_slot": bool(
                        last_info.get("rl_xy_out_of_slot", False)
                    ),
                    "terminal_reason": last_info.get("terminal_reason"),
                }
                if act_action is not None:
                    trace_step["act_action"] = np.asarray(act_action).tolist()
                elif "bc_action" in last_info:
                    trace_step.update(
                        {
                            "bc_action": np.asarray(last_info["bc_action"]).tolist(),
                            "scaled_bc_action": np.asarray(
                                last_info["scaled_bc_action"]
                            ).tolist(),
                            "policy_action": np.asarray(
                                last_info["policy_action"]
                            ).tolist(),
                            "normalized_action": np.asarray(
                                last_info["normalized_action"]
                            ).tolist(),
                            "target_qpos": np.asarray(
                                last_info["target_qpos"]
                            ).tolist(),
                        }
                    )
                trace.append(trace_step)

        success = bool(last_info.get("success", False))
        task.clean_cache(result="success" if success else "failed")
        diagnostics = last_info.get("success_diagnostics") or {}
        result = {
            "episode": episode_index,
            "seed": seed,
            "success": success,
            "reward": episode_reward,
            "actions": int(task.take_action_cnt),
            "initial_info": reset_info,
            "final_diagnostics": diagnostics,
            "rl_xy_out_of_slot": bool(last_info.get("rl_xy_out_of_slot", False)),
            "terminal_reason": last_info.get("terminal_reason"),
            "hybrid_motion_plan": hybrid_motion_plan,
        }
        if args.save_traces:
            result["trace"] = trace
        printed_result = {key: value for key, value in result.items() if key != "trace"}
        print(json.dumps(printed_result, sort_keys=True))
        results.append(result)

    success_count = sum(result["success"] for result in results)
    summary = {
        "algorithm": args.algorithm,
        "episodes": len(results),
        "successes": success_count,
        "success_rate": success_count / max(len(results), 1),
        "mean_reward": float(np.mean([result["reward"] for result in results])),
        "bc_checkpoint": str(args.bc_checkpoint.expanduser().resolve()),
        "model_path": (
            str(args.model_path.expanduser().resolve())
            if args.model_path is not None
            else None
        ),
        "act_checkpoint": (
            str(args.act_checkpoint.expanduser().resolve())
            if args.act_checkpoint is not None
            else None
        ),
        "act_config": (
            str(args.act_config.expanduser().resolve())
            if args.algorithm == "act"
            else None
        ),
        "deterministic": True,
        "control_mode": env_control_mode,
        "reward_mode": args.reward_mode,
        "handoff_mode": args.handoff_mode,
        "insert_usb_handoff_distribution": args.insert_usb_handoff_distribution,
        "insert_usb_coarse_z_jitter": args.insert_usb_coarse_z_jitter,
        "insert_usb_fixed_target_slot": (
            args.insert_usb_fixed_target_slot
        ),
        "insert_usb_curriculum_success_thresholds": list(
            args.insert_usb_curriculum_success_thresholds
        ),
        "insert_usb_xy_quit_threshold": args.insert_usb_xy_quit_threshold,
        "action_repeat": args.action_repeat,
        "bc_action_gain": args.bc_action_gain,
        "zero_qpos": args.zero_qpos,
        "step_limit": args.step_limit,
        "hybrid_motion_plan_z_threshold": (
            args.hybrid_motion_plan_z_threshold
        ),
        "hybrid_motion_plan_retreat_clearance": (
            args.hybrid_motion_plan_retreat_clearance
        ),
        "hybrid_motion_plan_triggers": sum(
            result["hybrid_motion_plan"] is not None for result in results
        ),
        "hybrid_motion_plan_successes": sum(
            result["success"] and result["hybrid_motion_plan"] is not None
            for result in results
        ),
        "freeze_encoder": args.freeze_encoder,
        "results": results,
    }
    with (output_dir / "evaluation.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    environment.close()


def _encode_act_observation(
    raw_observation: dict,
    act_model: ACT,
    *,
    tactile_key: str,
) -> dict:
    import torch
    from torchvision import transforms

    camera_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    tactile_transform = transforms.Resize((256, 256))

    def image_tensor(value, *, camera: bool):
        tensor = torch.as_tensor(value).permute(2, 0, 1).float() / 255.0
        return camera_transform(tensor) if camera else tactile_transform(tensor)

    def tactile_sensor(side: str) -> dict:
        tactile = raw_observation["tactile"]
        for key in (f"{side}_tactile", f"{side}_gsmini"):
            if key in tactile:
                return tactile[key]
        raise KeyError(f"Missing {side} tactile observation: {list(tactile)}")

    joint = raw_observation["embodiment"]["joint"]
    if torch.is_tensor(joint):
        joint = joint.detach().cpu().numpy()
    encoded = {"qpos": np.asarray(joint[: act_model.state_dim], dtype=np.float32)}
    for name in act_model.camera_names:
        if name == "cam_high":
            value = raw_observation["observation"]["head"]["rgb"]
        elif name == "cam_wrist":
            value = raw_observation["observation"]["wrist"]["rgb"]
        else:
            raise KeyError(f"Unknown ACT camera name: {name}")
        encoded[name] = image_tensor(value, camera=True)
    for name in act_model.tactile_names:
        if name == "tac_left":
            sensor = tactile_sensor("left")
        elif name == "tac_right":
            sensor = tactile_sensor("right")
        else:
            raise KeyError(f"Unknown ACT tactile name: {name}")
        selected_key = tactile_key if tactile_key in sensor else "rgb_marker"
        encoded[name] = image_tensor(sensor[selected_key], camera=False)
    return encoded


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
