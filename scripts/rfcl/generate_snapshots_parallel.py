"""Generate selected Insert USB RFCL snapshot profiles across multiple GPUs."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Sequence

try:
    from ._bootstrap import REPOSITORY_ROOT, add_repository_root
except ImportError:
    from _bootstrap import REPOSITORY_ROOT, add_repository_root

add_repository_root()

from scripts.rfcl.internal.snapshot_merge import merge_snapshot_shards
from policy.RL.rfcl_snapshot import SNAPSHOT_MANIFEST_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-indices", type=int, nargs="+", required=True)
    parser.add_argument("--devices", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--merged-output", type=Path)
    parser.add_argument("--base-root", type=Path)
    parser.add_argument("--seed-base", type=int, default=40_000)
    parser.add_argument("--max-attempts-per-profile", type=int, default=12)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--action-mode", default="target_pos_vel_force")
    parser.add_argument("--step-limit", type=int, default=800)
    parser.add_argument("--task-config", default="gelsight")
    return parser.parse_args()


def partition_profile_indices(
    profile_indices: Sequence[int], worker_count: int
) -> list[list[int]]:
    if int(worker_count) <= 0:
        raise ValueError("worker_count must be positive")
    indices = [int(index) for index in profile_indices]
    if len(set(indices)) != len(indices):
        raise ValueError("profile indices must not contain duplicates")
    if any(not 1 <= index <= 40 for index in indices):
        raise ValueError("profile indices must be in [1, 40]")
    groups = [[] for _ in range(min(int(worker_count), len(indices)))]
    for offset, index in enumerate(indices):
        groups[offset % len(groups)].append(index)
    return groups


def build_worker_command(
    *,
    python_executable: str,
    generator: Path,
    output: Path,
    profile_indices: Sequence[int],
    seed_base: int,
    max_attempts_per_profile: int,
    stride: int,
    action_mode: str,
    step_limit: int,
    task_config: str,
) -> list[str]:
    return [
        python_executable,
        str(generator),
        "--demo-plan",
        "insert_usb_balanced40",
        "--profile-indices",
        *[str(index) for index in profile_indices],
        "--seed-base",
        str(seed_base),
        "--max-attempts-per-profile",
        str(max_attempts_per_profile),
        "--stride",
        str(stride),
        "--action-mode",
        str(action_mode),
        "--step-limit",
        str(step_limit),
        "--task-config",
        str(task_config),
        "--output",
        str(output),
        "--device",
        "cuda:0",
        "--headless",
    ]


def main() -> None:
    args = parse_args()
    groups = partition_profile_indices(args.profile_indices, len(args.devices))
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Parallel output is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    logs_root = args.output / "logs"
    logs_root.mkdir()
    generator = Path(__file__).with_name("generate_snapshots.py")
    processes: list[tuple[subprocess.Popen[bytes], object, Path]] = []

    def stop_children(_signum: int, _frame: object) -> None:
        for process, _handle, _root in processes:
            if process.poll() is None:
                process.terminate()

    signal.signal(signal.SIGINT, stop_children)
    signal.signal(signal.SIGTERM, stop_children)
    try:
        for worker_id, (device, indices) in enumerate(zip(args.devices, groups)):
            shard_root = args.output / "shards" / f"worker_{worker_id:02d}"
            command = build_worker_command(
                python_executable=sys.executable,
                generator=generator,
                output=shard_root,
                profile_indices=indices,
                seed_base=args.seed_base,
                max_attempts_per_profile=args.max_attempts_per_profile,
                stride=args.stride,
                action_mode=args.action_mode,
                step_limit=args.step_limit,
                task_config=args.task_config,
            )
            log_path = logs_root / f"worker_{worker_id:02d}.log"
            handle = log_path.open("wb")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(device).removeprefix("cuda:")
            process = subprocess.Popen(
                command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                cwd=REPOSITORY_ROOT,
                env=environment,
            )
            processes.append((process, handle, shard_root))
            print(
                f"[parallel-rfcl-snapshot] worker={worker_id} device={device} "
                f"profiles={indices} pid={process.pid}",
                flush=True,
            )

        failures = []
        for worker_id, (process, handle, shard_root) in enumerate(processes):
            return_code = process.wait()
            handle.close()
            if return_code != 0:
                failures.append((worker_id, return_code, shard_root))
        if failures:
            raise RuntimeError(f"Snapshot workers failed: {failures}")

        if args.merged_output is not None:
            inputs = [shard_root for _process, _handle, shard_root in processes]
            expected_demos = len(args.profile_indices)
            if args.base_root is not None:
                inputs.insert(0, args.base_root)
                base_manifest = json.loads(
                    (args.base_root / SNAPSHOT_MANIFEST_NAME).read_text(
                        encoding="utf-8"
                    )
                )
                expected_demos += len(base_manifest.get("demos", ()))
            merged = merge_snapshot_shards(
                inputs,
                args.merged_output,
                expected_demos=expected_demos,
            )
            print(
                f"[parallel-rfcl-snapshot] merged={args.merged_output} "
                f"demos={len(merged['demos'])}",
                flush=True,
            )
    finally:
        for process, handle, _root in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
            if not handle.closed:
                handle.close()


if __name__ == "__main__":
    main()
