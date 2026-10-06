#!/usr/bin/env bash
set -e

cd 

CUDA_VISIBLE_DEVICES=5 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  roughness_regrasp \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_tactile_step_smooth.yml \
  --rough_block_side right \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9001

CUDA_VISIBLE_DEVICES=5 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  roughness_regrasp \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_tactile_step_smooth.yml \
  --rough_block_side left \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9001
