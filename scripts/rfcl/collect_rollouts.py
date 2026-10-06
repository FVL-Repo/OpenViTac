"""Launch isolated frozen-policy RFCL rollout workers."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

try:
    from ._bootstrap import REPOSITORY_ROOT, add_repository_root
except ImportError:
    from _bootstrap import REPOSITORY_ROOT, add_repository_root

add_repository_root()

from policy.RL.rfcl_collection import balanced_quotas, resolve_snapshot_identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--successes", type=int, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--task-config", default=None)
    parser.add_argument("--max-attempts", type=int, default=10000)
    parser.add_argument("--minimum-steps", type=int, default=20)
    parser.add_argument("--frontier-window", type=int, default=8)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--exploration-noise",
        type=float,
        nargs="+",
        default=[0.0],
        help="Noise levels assigned to workers in round-robin order.",
    )
    parser.add_argument("--base-seed", type=int, default=10000)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-args", nargs=argparse.REMAINDER, default=[])
    return parser.parse_args()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def resolve_worker_device(device: str) -> tuple[str, str | None]:
    value = str(device)
    if value.startswith("cuda:"):
        physical_index = value.split(":", 1)[1]
        if not physical_index.isdigit():
            raise ValueError(f"Invalid CUDA worker device {value!r}")
        return "cuda:0", physical_index
    if value in ("cuda", "cpu"):
        return value, None
    raise ValueError(f"Unsupported worker device {value!r}")


def build_worker_command(
    *,
    python_executable: str,
    worker_script: Path,
    checkpoint: Path,
    snapshot_root: Path,
    output: Path,
    successes: int,
    max_attempts: int,
    minimum_steps: int,
    frontier_window: int,
    deterministic: bool,
    exploration_noise: float,
    worker_id: int,
    worker_seed: int,
    device: str,
    resume: bool,
    task_name: str | None = None,
    task_config: str | None = None,
    worker_args: Sequence[str] = (),
) -> list[str]:
    command = [
        python_executable,
        str(worker_script),
        "--checkpoint",
        str(checkpoint),
        "--snapshot-root",
        str(snapshot_root),
        "--output",
        str(output),
        "--successes",
        str(int(successes)),
        "--max-attempts",
        str(int(max_attempts)),
        "--minimum-steps",
        str(int(minimum_steps)),
        "--frontier-window",
        str(int(frontier_window)),
        "--exploration-noise",
        str(float(exploration_noise)),
        "--worker-id",
        str(int(worker_id)),
        "--worker-seed",
        str(int(worker_seed)),
        "--device",
        str(device),
        "--headless",
        "--deterministic" if deterministic else "--no-deterministic",
        "--resume" if resume else "--no-resume",
    ]
    if task_name is not None:
        command.extend(("--task-name", str(task_name)))
    if task_config is not None:
        command.extend(("--task-config", str(task_config)))
    command.extend(str(value) for value in worker_args)
    return command


def consolidate_trajectories(worker_roots: Sequence[Path], output: Path) -> int:
    destination_root = output / "success_trajectories"
    destination_root.mkdir(parents=True, exist_ok=True)
    sources: dict[str, Path] = {}
    for worker_root in worker_roots:
        for source in sorted((worker_root / "success_trajectories").glob("*.npz")):
            previous = sources.get(source.name)
            if previous is not None and previous.resolve() != source.resolve():
                raise RuntimeError(f"Duplicate trajectory UUID: {source.name}")
            sources[source.name] = source
    for name, source in sources.items():
        destination = destination_root / name
        if destination.exists() or destination.is_symlink():
            if destination.resolve() == source.resolve():
                continue
            raise FileExistsError(f"Refusing to replace {destination}")
        destination.symlink_to(os.path.relpath(source, start=destination.parent))
    return len(sources)


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.snapshot_root.is_dir():
        raise FileNotFoundError(args.snapshot_root)
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if not args.devices:
        raise ValueError("--devices cannot be empty")
    if any(noise < 0.0 for noise in args.exploration_noise):
        raise ValueError("--exploration-noise values cannot be negative")
    args.task_name, args.task_config = resolve_snapshot_identity(
        args.snapshot_root,
        task_name=args.task_name,
        task_config=args.task_config,
    )

    quotas = balanced_quotas(args.successes, args.workers)
    repository_root = REPOSITORY_ROOT
    worker_script = repository_root / "scripts/rfcl/internal/collect_rollout_worker.py"
    worker_root = args.output / "parallel_workers"
    log_root = args.output / "parallel_logs"
    worker_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    workers: list[dict[str, Any]] = []
    for worker_id, quota in enumerate(quotas):
        output = worker_root / f"worker_{worker_id:03d}"
        physical_device = str(args.devices[worker_id % len(args.devices)])
        process_device, visible_device = resolve_worker_device(physical_device)
        noise = float(
            args.exploration_noise[worker_id % len(args.exploration_noise)]
        )
        command = build_worker_command(
            python_executable=sys.executable,
            worker_script=worker_script,
            checkpoint=args.checkpoint.resolve(),
            snapshot_root=args.snapshot_root.resolve(),
            output=output.resolve(),
            successes=quota,
            max_attempts=args.max_attempts,
            minimum_steps=args.minimum_steps,
            frontier_window=args.frontier_window,
            deterministic=args.deterministic,
            exploration_noise=noise,
            worker_id=worker_id,
            worker_seed=args.base_seed + worker_id,
            device=process_device,
            resume=args.resume,
            task_name=args.task_name,
            task_config=args.task_config,
            worker_args=args.worker_args,
        )
        workers.append(
            {
                "worker": worker_id,
                "device": physical_device,
                "process_device": process_device,
                "visible_device": visible_device,
                "exploration_noise": noise,
                "success_quota": quota,
                "output": str(output.resolve()),
                "log": str((log_root / f"worker_{worker_id:03d}.log").resolve()),
                "command": command,
            }
        )

    manifest_path = args.output / "parallel_manifest.json"
    manifest: dict[str, Any] = {
        "schema": "rfcl_parallel_rollout_v2",
        "created_at": datetime.now().astimezone().isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "snapshot_root": str(args.snapshot_root.resolve()),
        "requested_successes": args.successes,
        "workers": workers,
        "status": "dry_run" if args.dry_run else "running",
    }
    write_json_atomic(manifest_path, manifest)
    for worker in workers:
        print(
            f"[rfcl-parallel-rollout] worker={worker['worker']} "
            f"device={worker['device']} quota={worker['success_quota']} "
            f"noise={worker['exploration_noise']} command="
            f"{shlex.join(worker['command'])}",
            flush=True,
        )
    if args.dry_run:
        return

    processes: list[tuple[dict[str, Any], subprocess.Popen, Any]] = []
    environment = os.environ.copy()
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment["PYTHONUNBUFFERED"] = "1"
    try:
        for worker in workers:
            log_handle = Path(worker["log"]).open("a", encoding="utf-8")
            worker_environment = environment.copy()
            if worker["visible_device"] is not None:
                worker_environment["CUDA_VISIBLE_DEVICES"] = str(
                    worker["visible_device"]
                )
            process = subprocess.Popen(
                worker["command"],
                cwd=repository_root,
                env=worker_environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            processes.append((worker, process, log_handle))
        pending = {process.pid for _, process, _ in processes}
        while pending:
            for worker, process, _ in processes:
                if process.pid not in pending:
                    continue
                return_code = process.poll()
                if return_code is None:
                    continue
                worker["return_code"] = int(return_code)
                pending.remove(process.pid)
                print(
                    f"[rfcl-parallel-rollout] worker={worker['worker']} "
                    f"finished return_code={return_code}",
                    flush=True,
                )
            if pending:
                time.sleep(1.0)
    except BaseException:
        for _, process, _ in processes:
            if process.poll() is None:
                process.terminate()
        for _, process, _ in processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
        raise
    finally:
        for _, _, log_handle in processes:
            log_handle.close()

    failed_workers = [
        int(worker["worker"])
        for worker in workers
        if int(worker.get("return_code", -1)) != 0
    ]
    consolidated = consolidate_trajectories(
        [Path(worker["output"]) for worker in workers],
        args.output,
    )
    manifest.update(
        {
            "finished_at": datetime.now().astimezone().isoformat(),
            "failed_workers": failed_workers,
            "consolidated_trajectories": consolidated,
            "status": "failed" if failed_workers else "complete",
        }
    )
    write_json_atomic(manifest_path, manifest)
    if failed_workers:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
