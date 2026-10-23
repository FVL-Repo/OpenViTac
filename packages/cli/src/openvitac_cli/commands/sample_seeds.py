"""Sample task randomization without launching the simulator.

Only the pure task specs are imported: ``Task.setup(seed)`` runs the task's
randomizer and never touches Kit, the scene, or the GPU.  Each sampled seed
becomes one JSONL row, so manifests can be reviewed, filtered with ``jq``, and
collected seed-by-seed afterwards::

    openvitac sample-seeds --task insert_usb --count 100 --out seeds.jsonl
    jq -r 'select(.randomization["dr.slot_xy_half_m"] > 0.01) | .seed' \\
        seeds.jsonl | while read -r seed; do
        openvitac collect insert_usb --seed "$seed" --expert
    done

Seeds are sampled in ``[start, start + count)`` order.  ``randomization`` holds
the scalar ``EpisodeRandomization.to_metadata()`` values; expert-only parameters
are excluded because they are runtime inputs, not manifest labels.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import cast

from openvitac_tasks import TASKS, EpisodeRandomization, Task, TaskName


def _manifest_metadata(randomization: EpisodeRandomization) -> dict[str, float | str]:
    """Scalar metadata for the manifest, without expert-only parameters."""
    expert_keys = {f"expert.{name}" for name in randomization.expert_params}
    sampled: dict[str, float | str] = {}
    for key, value in randomization.to_metadata().items():
        if key in expert_keys:
            continue
        sampled[key] = value
    return dict(sorted(sampled.items()))


def _rows(task: str, spec: Task, seeds: range) -> Iterator[dict[str, object]]:
    for seed in seeds:
        randomization = spec.setup(seed)
        yield {
            "seed": seed,
            "task": task,
            "instruction": randomization.instruction,
            "randomization": _manifest_metadata(randomization),
        }


def _write_jsonl(out: Path, rows: Iterator[dict[str, object]]) -> None:
    """Write all rows to a temporary sibling, then atomically replace ``out``.

    A failure while sampling (for example, a task randomizer raising) leaves no
    partial manifest and does not clobber a previous run's file.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{out.name}.", suffix=".tmp", dir=out.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False))
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, out)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise


def sample_seeds(
    task: str,
    out: Path,
    start: int = 0,
    count: int = 100,
) -> None:
    """Sample task randomization for a contiguous range of seeds.

    Writes one UTF-8 JSON object per line with ``seed``, ``task``,
    ``instruction``, and the scalar ``randomization`` metadata.  No simulator
    is launched, so this is safe to run on any machine.  Filter the manifest
    with ``jq`` and then call ``collect --seed <n>`` once per selected seed:

        openvitac sample-seeds --task insert_usb --count 100 --out seeds.jsonl
        jq -r 'select(.randomization["dr.slot_xy_half_m"] > 0.01) | .seed' \\
            seeds.jsonl | while read -r seed; do
            openvitac collect insert_usb --seed "$seed" --expert
        done

    Parameters
    ----------
    task: Registered task name (``openvitac tasks`` lists them).
    out: Destination JSONL path; written atomically.
    start: First seed, inclusive.
    count: Number of consecutive seeds; must be positive.
    """
    if start < 0:
        raise ValueError(f"--start must be non-negative, got {start}")
    if count <= 0:
        raise ValueError(f"--count must be positive, got {count}")
    spec = TASKS[cast(TaskName, task)]
    seeds = range(start, start + count)
    _write_jsonl(out, _rows(task, spec, seeds))
    print(f"wrote {count} seeds to {out}")
