"""Summarize one distributed RFCL training output directory."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tail", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = args.output / "metrics.jsonl"
    manifest_path = args.output / "distributed_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    if metrics_path.is_file():
        rows = [
            json.loads(line)
            for line in metrics_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    tail = rows[-max(1, int(args.tail)) :]
    latest = rows[-1] if rows else {}
    success_count = sum(bool(row.get("success")) for row in tail)
    worker_counts = Counter(int(row["worker_id"]) for row in tail)
    worker_successes = Counter(
        int(row["worker_id"]) for row in tail if bool(row.get("success"))
    )
    payload = {
        "output": str(args.output.resolve()),
        "session_id": manifest.get("session_id"),
        "learner_device": manifest.get("learner_device"),
        "worker_count": manifest.get("worker_count"),
        "worker_devices": manifest.get("worker_devices"),
        "completed_episodes": latest.get("completed_episodes", 0),
        "total_steps": latest.get("total_steps", 0),
        "frontiers": latest.get("frontiers"),
        "curriculum_progress": latest.get("curriculum_progress"),
        "qualifying_success_trajectories": latest.get(
            "qualifying_success_trajectories", 0
        ),
        "recent_window": len(tail),
        "recent_successes": success_count,
        "recent_success_rate": (
            float(success_count) / float(len(tail)) if tail else None
        ),
        "worker_recent_episodes": dict(sorted(worker_counts.items())),
        "worker_recent_successes": dict(sorted(worker_successes.items())),
        "latest_checkpoint": (
            str((args.output / "latest.pt").resolve())
            if (args.output / "latest.pt").exists()
            else None
        ),
        "elapsed_s": latest.get("elapsed_s", 0.0),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
