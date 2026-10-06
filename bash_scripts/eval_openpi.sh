#!/usr/bin/env bash
# OpenViTac openpi rollout evaluation driver.
#
# Replaces the former one-script-per-experiment eval_openpi_*.sh family.
# All 29 calls preserved; each combination is one line in the CASES list.
#
# Usage:
#   bash bash_scripts/eval_openpi.sh                          # run all cases
#   bash bash_scripts/eval_openpi.sh weight_classify           # filter by task substring
#   bash bash_scripts/eval_openpi.sh weight_classify heavy     # + by label substring
#   OPENVITAC_OPENPI_HOST=10.0.0.5 OPENPI_PORT=8000 bash bash_scripts/eval_openpi.sh
#
# Columns: task | sensor | deploy_config | extra args (space-separated)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OPENPI_HOST="${OPENVITAC_OPENPI_HOST:-127.0.0.1}"
FILTER_TASK="${1:-}"
FILTER_LABEL="${2:-}"

CASES=(
  # --- hardness_classify (vision / vision_tactile) ---
  "hardness_classify gelsight openpi/eef_delta/deploy_vision --hardness_label hard --start_seed 0 --max_seed 49 --total_num 50 --openpi_port 8000"
  "hardness_classify gelsight openpi/eef_delta/deploy_vision --hardness_label soft --start_seed 0 --max_seed 49 --total_num 50 --openpi_port 8000"
  "hardness_classify gelsight openpi/eef_delta/deploy_vision_tactile --hardness_label hard --start_seed 0 --max_seed 49 --total_num 50 --openpi_port 8000"
  "hardness_classify gelsight openpi/eef_delta/deploy_vision_tactile --hardness_label soft --start_seed 0 --max_seed 49 --total_num 50 --openpi_port 8000"
  # --- roughness_classify ---
  "roughness_classify gelsight openpi/eef_delta/deploy_vision --roughness_label rough --start_seed 0 --max_seed 49 --total_num 50 --openpi_port 8000"
  "roughness_classify gelsight openpi/eef_delta/deploy_vision --roughness_label smooth --start_seed 0 --max_seed 49 --total_num 50 --openpi_port 8000"
  "roughness_classify gelsight openpi/eef_delta/deploy_vision_tactile --roughness_label rough --start_seed 0 --max_seed 49 --total_num 50 --openpi_port 8000"
  "roughness_classify gelsight openpi/eef_delta/deploy_vision_tactile --roughness_label smooth --start_seed 0 --max_seed 49 --total_num 50 --openpi_port 8000"
  # --- weight_classify ---
  "weight_classify gelsight openpi/eef_delta/deploy_vision --weight_label light --start_seed 0 --max_seed 49 --total_num 50 --openpi_port 8000"
  "weight_classify gelsight openpi/eef_delta/deploy_vision --weight_label heavy --start_seed 0 --max_seed 49 --total_num 50 --openpi_port 8000"
  "weight_classify gelsight openpi/eef_delta/deploy_vision_tactile --weight_label light --start_seed 0 --max_seed 49 --total_num 50 --openpi_port 8000"
  "weight_classify gelsight openpi/eef_delta/deploy_vision_tactile --weight_label heavy --start_seed 0 --max_seed 49 --total_num 50 --openpi_port 8000"
  # --- roughness_regrasp ---
  "roughness_regrasp gelsight openpi/eef_delta/deploy_vision --rough_block_side left --initial_grasp_side random --start_seed 0 --max_seed 49 --total_num 50 --openpi_port 8000"
  "roughness_regrasp gelsight openpi/eef_delta/deploy_vision --rough_block_side right --initial_grasp_side random --start_seed 0 --max_seed 49 --total_num 50 --openpi_port 8000"
  "roughness_regrasp gelsight openpi/eef_delta/deploy_vision_tactile --rough_block_side left --initial_grasp_side random --start_seed 0 --max_seed 49 --total_num 50 --openpi_port 8000"
  "roughness_regrasp gelsight openpi/eef_delta/deploy_vision_tactile --rough_block_side right --initial_grasp_side random --start_seed 0 --max_seed 49 --total_num 50 --openpi_port 8000"
  # --- insert_block 12 cases (target x base_pose_indices x seed range) ---
  "insert_block gelsight openpi/eef_delta/deploy_vision_tactile --target_block cube --block_base_pose_indices 0,1,4 --start_seed 0 --max_seed 1000 --total_num 1 --openpi_port 8000"
  "insert_block gelsight openpi/eef_delta/deploy_vision_tactile --target_block half_cylinder --block_base_pose_indices 0,1,4 --start_seed 1000 --max_seed 2000 --total_num 1 --openpi_port 8000"
  "insert_block gelsight openpi/eef_delta/deploy_vision_tactile --target_block hexagon --block_base_pose_indices 0,1,4 --start_seed 2000 --max_seed 3000 --total_num 1 --openpi_port 8000"
  "insert_block gelsight openpi/eef_delta/deploy_vision_tactile --target_block cube --block_base_pose_indices 1,3,2 --start_seed 0 --max_seed 1000 --total_num 1 --openpi_port 8000"
  "insert_block gelsight openpi/eef_delta/deploy_vision_tactile --target_block half_cylinder --block_base_pose_indices 1,3,2 --start_seed 1000 --max_seed 2000 --total_num 1 --openpi_port 8000"
  "insert_block gelsight openpi/eef_delta/deploy_vision_tactile --target_block hexagon --block_base_pose_indices 1,3,2 --start_seed 2000 --max_seed 3000 --total_num 1 --openpi_port 8000"
  "insert_block gelsight openpi/eef_delta/deploy_vision_tactile --target_block cube --block_base_pose_indices 2,4,0 --start_seed 0 --max_seed 1000 --total_num 1 --openpi_port 8000"
  "insert_block gelsight openpi/eef_delta/deploy_vision_tactile --target_block half_cylinder --block_base_pose_indices 2,4,0 --start_seed 1000 --max_seed 2000 --total_num 1 --openpi_port 8000"
  "insert_block gelsight openpi/eef_delta/deploy_vision_tactile --target_block hexagon --block_base_pose_indices 2,4,0 --start_seed 2000 --max_seed 3000 --total_num 1 --openpi_port 8000"
  "insert_block gelsight openpi/eef_delta/deploy_vision_tactile --target_block cube --block_base_pose_indices 3,2,1 --start_seed 0 --max_seed 1000 --total_num 1 --openpi_port 8000"
  "insert_block gelsight openpi/eef_delta/deploy_vision_tactile --target_block half_cylinder --block_base_pose_indices 3,2,1 --start_seed 1000 --max_seed 2000 --total_num 1 --openpi_port 8000"
  "insert_block gelsight openpi/eef_delta/deploy_vision_tactile --target_block hexagon --block_base_pose_indices 3,2,1 --start_seed 2000 --max_seed 3000 --total_num 1 --openpi_port 8000"
)

run=0
for case in "${CASES[@]}"; do
  read -r task sensor deploy extra <<<"$case"
  if [[ -n "$FILTER_TASK" && "$task" != *"$FILTER_TASK"* ]]; then continue; fi
  if [[ -n "$FILTER_LABEL" && "$extra" != *"$FILTER_LABEL"* ]]; then continue; fi
  # strip --openpi_port from extra; host/port are injected uniformly
  extra="${extra//--openpi_port 8000/}"
  extra="${extra//--openpi_port 8001/}"
  echo ">>> [$(date +%H:%M:%S)] $task $sensor $deploy $extra"
  OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
    "$task" \
    "$sensor" \
    "$deploy" \
    --tactile_sensor gelsight \
    --openpi_host "$OPENPI_HOST" \
    --openpi_port "${OPENPI_PORT:-8000}" \
    $extra
  run=$((run + 1))
done
echo "done: $run case(s)"
