"""Run RGB/tactile RFCL trajectory re-recording in isolated worker processes."""

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
from typing import Callable, Sequence

import numpy as np

try:
    from ._bootstrap import REPOSITORY_ROOT, add_repository_root
except ImportError:
    from _bootstrap import REPOSITORY_ROOT, add_repository_root

add_repository_root()

from policy.RL.rfcl_collection import resolve_snapshot_identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--selection-file", type=Path)
    selection.add_argument(
        "--trajectory-dir",
        type=Path,
        help="Directory containing the selected successful trajectory NPZ files.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--task-config", default=None)
    parser.add_argument("--step-limit", type=int, default=200)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["cuda:0"],
        help="Devices assigned to workers in round-robin order.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create shards and print worker commands without launching Isaac.",
    )
    parser.add_argument(
        "--worker-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional AppLauncher arguments passed to every worker.",
    )
    return parser.parse_args()


def load_selection_entries(path: Path) -> list[Path]:
    if not path.is_file():
        raise FileNotFoundError(f"Selection file does not exist: {path}")
    entries: list[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        source = Path(value)
        if not source.is_file():
            candidate = path.parent / source
            if candidate.is_file():
                source = candidate
            else:
                raise FileNotFoundError(
                    f"Selected trajectory does not exist: {value}"
                )
        entries.append(source.resolve())
    if not entries:
        raise ValueError(f"Selection is empty: {path}")
    output_names = [f"{entry.stem}.hdf5" for entry in entries]
    if len(output_names) != len(set(output_names)):
        raise ValueError("Selected trajectories do not have unique output names")
    return entries


def load_trajectory_directory(path: Path) -> list[Path]:
    if not path.is_dir():
        raise FileNotFoundError(f"Trajectory directory does not exist: {path}")
    entries = sorted(source.resolve() for source in path.glob("*.npz"))
    if not entries:
        raise ValueError(f"Trajectory directory is empty: {path}")
    output_names = [f"{entry.stem}.hdf5" for entry in entries]
    if len(output_names) != len(set(output_names)):
        raise ValueError("Trajectory files do not have unique output names")
    return entries


def partition_entries(
    entries: Sequence[Path],
    worker_count: int,
    *,
    shard_sort_key: Callable[[Path], tuple] | None = None,
) -> list[list[Path]]:
    if int(worker_count) <= 0:
        raise ValueError("worker_count must be positive")
    if not entries:
        raise ValueError("entries cannot be empty")
    shard_count = min(int(worker_count), len(entries))
    shards = [[] for _ in range(shard_count)]
    for entry_index, entry in enumerate(entries):
        shards[entry_index % shard_count].append(Path(entry))
    if shard_sort_key is not None:
        for shard in shards:
            shard.sort(key=shard_sort_key)
    return shards


def trajectory_replay_sort_key(path: Path) -> tuple[int, int, str]:
    with np.load(path, allow_pickle=False) as archive:
        if "trajectory_uuid" in archive.files:
            identity = str(np.asarray(archive["trajectory_uuid"]).item())
        else:
            identity = path.name
        return (
            int(np.asarray(archive["demo_index"]).item()),
            int(np.asarray(archive["state_index"]).item()),
            identity,
        )


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
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


def write_shard(path: Path, entries: Sequence[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{entry.resolve()}\n" for entry in entries),
        encoding="utf-8",
    )


def build_worker_command(
    *,
    python_executable: str,
    worker_script: Path,
    snapshot_root: Path,
    selection_file: Path,
    output: Path,
    task_name: str,
    task_config: str,
    step_limit: int,
    max_retries: int,
    device: str,
    resume: bool,
    worker_args: Sequence[str] = (),
) -> list[str]:
    command = [
        python_executable,
        str(worker_script),
        "--snapshot-root",
        str(snapshot_root),
        "--selection-file",
        str(selection_file),
        "--output",
        str(output),
        "--task-name",
        task_name,
        "--task-config",
        task_config,
        "--step-limit",
        str(int(step_limit)),
        "--max-retries",
        str(int(max_retries)),
        "--device",
        device,
        "--headless",
        "--resume" if resume else "--no-resume",
    ]
    command.extend(str(value) for value in worker_args)
    return command


def consolidate_hdf5(worker_roots: Sequence[Path], output: Path) -> int:
    hdf5_root = output / "hdf5"
    hdf5_root.mkdir(parents=True, exist_ok=True)
    sources: dict[str, Path] = {}
    for worker_root in worker_roots:
        for source in sorted((worker_root / "hdf5").glob("*.hdf5")):
            previous = sources.get(source.name)
            if previous is not None and previous.resolve() != source.resolve():
                raise RuntimeError(
                    f"Duplicate worker output {source.name}: {previous} and {source}"
                )
            sources[source.name] = source
    for name, source in sources.items():
        destination = hdf5_root / name
        if destination.exists() or destination.is_symlink():
            if destination.resolve() == source.resolve():
                continue
            raise FileExistsError(
                f"Refusing to replace consolidated output: {destination}"
            )
        relative_source = os.path.relpath(source, start=destination.parent)
        destination.symlink_to(relative_source)
    return len(sources)


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.step_limit <= 0:
        raise ValueError("--step-limit must be positive")
    if args.max_retries <= 0:
        raise ValueError("--max-retries must be positive")
    if not args.devices:
        raise ValueError("--devices cannot be empty")
    args.task_name, args.task_config = resolve_snapshot_identity(
        args.snapshot_root,
        task_name=args.task_name,
        task_config=args.task_config,
    )

    repository_root = REPOSITORY_ROOT
    worker_script = repository_root / "scripts/rfcl/internal/rerecord_worker.py"
    entries = (
        load_selection_entries(args.selection_file)
        if args.selection_file is not None
        else load_trajectory_directory(args.trajectory_dir)
    )
    shards = partition_entries(
        entries,
        args.workers,
        shard_sort_key=trajectory_replay_sort_key,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    shard_root = args.output / "parallel_shards"
    worker_root = args.output / "parallel_workers"
    log_root = args.output / "parallel_logs"
    worker_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    workers: list[dict[str, object]] = []
    for worker_index, shard in enumerate(shards):
        shard_path = shard_root / f"worker_{worker_index:03d}.txt"
        output_path = worker_root / f"worker_{worker_index:03d}"
        log_path = log_root / f"worker_{worker_index:03d}.log"
        physical_device = str(args.devices[worker_index % len(args.devices)])
        process_device, visible_device = resolve_worker_device(physical_device)
        write_shard(shard_path, shard)
        command = build_worker_command(
            python_executable=sys.executable,
            worker_script=worker_script,
            snapshot_root=args.snapshot_root.resolve(),
            selection_file=shard_path.resolve(),
            output=output_path.resolve(),
            task_name=args.task_name,
            task_config=args.task_config,
            step_limit=args.step_limit,
            max_retries=args.max_retries,
            device=process_device,
            resume=args.resume,
            worker_args=args.worker_args,
        )
        workers.append(
            {
                "worker": worker_index,
                "device": physical_device,
                "process_device": process_device,
                "visible_device": visible_device,
                "items": len(shard),
                "selection_file": str(shard_path.resolve()),
                "output": str(output_path.resolve()),
                "log": str(log_path.resolve()),
                "command": command,
            }
        )

    manifest_path = args.output / "parallel_manifest.json"
    manifest: dict[str, object] = {
        "schema": "rfcl_parallel_rerecord_v1",
        "created_at": datetime.now().astimezone().isoformat(),
        "snapshot_root": str(args.snapshot_root.resolve()),
        "selection_file": (
            str(args.selection_file.resolve())
            if args.selection_file is not None
            else None
        ),
        "trajectory_dir": (
            str(args.trajectory_dir.resolve())
            if args.trajectory_dir is not None
            else None
        ),
        "selected_trajectories": [str(entry) for entry in entries],
        "task_name": args.task_name,
        "task_config": args.task_config,
        "selected": len(entries),
        "workers": workers,
        "status": "dry_run" if args.dry_run else "running",
    }
    write_json_atomic(manifest_path, manifest)
    for worker in workers:
        print(
            f"[rfcl-parallel] worker={worker['worker']} device={worker['device']} "
            f"items={worker['items']} command="
            f"{shlex.join(worker['command'])}",
            flush=True,
        )
    if args.dry_run:
        return

    processes: list[tuple[dict[str, object], subprocess.Popen, object]] = []
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
                pending.remove(process.pid)
                worker["return_code"] = int(return_code)
                print(
                    f"[rfcl-parallel] worker={worker['worker']} "
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
    consolidated = consolidate_hdf5(
        [Path(worker["output"]) for worker in workers],
        args.output,
    )
    manifest.update(
        {
            "finished_at": datetime.now().astimezone().isoformat(),
            "status": "failed" if failed_workers else "complete",
            "failed_workers": failed_workers,
            "consolidated_hdf5": consolidated,
        }
    )
    write_json_atomic(manifest_path, manifest)
    print(
        f"[rfcl-parallel] status={manifest['status']} "
        f"hdf5={consolidated}/{len(entries)}",
        flush=True,
    )
    if failed_workers:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
