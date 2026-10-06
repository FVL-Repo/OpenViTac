# RFCL Commands

This directory contains the supported RFCL training and data-collection CLI.
Run commands from the repository root.

## Workflow

1. `generate_snapshots.py` creates Motion Plan snapshot demonstrations.
2. `generate_snapshots_parallel.py` distributes snapshot generation across GPUs.
3. `train.py` trains one shared policy with multiple rollout workers.
4. `status.py` reports distributed training progress.
5. `collect_rollouts.py` freezes a checkpoint and collects successful trajectories.
6. `rerecord.py` replays successful trajectories with RGB and tactile recording.
7. `summarize_rerecord.py` validates and summarizes re-recorded HDF5 files.
8. `record_motion_plan_prefixes.py` records full Motion Plan prefixes.
9. `concat_full_trajectories.py` joins prefixes and RFCL suffixes.
10. `export_preview_videos.py` exports four-panel preview videos.
11. `analyze_diversity.py` analyzes trajectory coverage and diversity.

Modules under `internal/` are worker implementations launched by the public
commands. They are not stable user-facing entry points.
