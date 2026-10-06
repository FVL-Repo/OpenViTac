```bash

ssh -N -L 8010:127.0.0.1:8010 -p <port> root@<ftp1-train-server>
ssh -N -L 8018:127.0.0.1:8018 -p <port> root@<ftp1-train-server>
ssh -N -L 8020:127.0.0.1:8020 -p <port> root@<ftp1-train-server>
ssh -N -L 8022:127.0.0.1:8022 -p <port> root@<ftp1-train-server>

ssh -N -L 8000:127.0.0.1:8000 -p <port> root@<ftp2-train-server>
ssh -N -L 8002:127.0.0.1:8002 -p <port> root@<ftp2-train-server>
ssh -N -L 8004:127.0.0.1:8004 -p <port> root@<ftp2-train-server>
ssh -N -L 8006:127.0.0.1:8006 -p <port> root@<ftp2-train-server>
ssh -N -L 8024:127.0.0.1:8024 -p <port> root@<ftp2-train-server>
ssh -N -L 8026:127.0.0.1:8026 -p <port> root@<ftp2-train-server>

cd /dataconda activate OpenViTac

CUDA_VISIBLE_DEVICES=3 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  turn_gear_pair \
  task_config/gelsight.yml \
  policy/internvla_a1_5/deploy_turn_gear_pair.yml \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 100

CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  can_empty_select \
  task_config/gelsight.yml \
  policy/internvla_a1_5/deploy_can_empty_select.yml \
  --empty_can coke \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=2 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  can_empty_select \
  task_config/gelsight.yml \
  policy/internvla_a1_5/deploy_can_empty_select.yml \
  --empty_can fanta \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=2 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  can_empty_select \
  task_config/gelsight.yml \
  policy/internvla_a1_5/deploy_can_empty_select.yml \
  --empty_can 7up \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=3 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  can_empty_select \
  task_config/gelsight.yml \
  policy/internvla_a1_5/deploy_can_empty_select.yml \
  --empty_can pepsi \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_USB \
  task_config/gelsight.yml \
  policy/internvla_a1_5/deploy_insert_USB.yml \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 100

CUDA_VISIBLE_DEVICES=3 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  hardness_classify \
  task_config/gelsight.yml \
  policy/internvla_a1_5/deploy_hardness_classify.yml \
  --hardness_label soft \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=2 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  hardness_classify \
  task_config/gelsight.yml \
  policy/internvla_a1_5/deploy_hardness_classify.yml \
  --hardness_label hard \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=3 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  weight_classify \
  task_config/gelsight.yml \
  policy/internvla_a1_5/deploy_weight_classify.yml \
  --weight_label light\
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=2 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  weight_classify \
  task_config/gelsight.yml \
  policy/internvla_a1_5/deploy_weight_classify.yml \
  --weight_label heavy\
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=2 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  roughness_regrasp \
  task_config/gelsight.yml \
  policy/internvla_a1_5/deploy_roughness_regrasp_classify.yml \
  --rough_block_side right \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=3 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  roughness_regrasp \
  task_config/gelsight.yml \
  policy/internvla_a1_5/deploy_roughness_regrasp_classify.yml \
  --rough_block_side left \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=2 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  roughness_classify \
  task_config/gelsight.yml \
  policy/internvla_a1_5/deploy_roughness_classify.yml \
  --roughness_label rough \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=3 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  roughness_classify \
  task_config/gelsight.yml \
  policy/internvla_a1_5/deploy_roughness_classify.yml \
  --roughness_label smooth \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=7 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  pull_drawer \
  task_config/gelsight.yml \
  policy/internvla_a1_5/deploy_pull_drawer.yml \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 100 \
  --background base4

CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  task_config/gelsight.yml \
  policy/internvla_a1_5/deploy_insert_block.yml \
  --target_block cube \
  --block_base_pose_indices 0,1,4 \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  task_config/gelsight.yml \
  policy/internvla_a1_5/deploy_insert_block.yml \
  --target_block cube \
  --block_base_pose_indices 1,3,2 \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=2 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  task_config/gelsight.yml \
  policy/internvla_a1_5/deploy_insert_block.yml \
  --target_block half_cylinder \
  --block_base_pose_indices 0,1,4 \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=3 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  task_config/gelsight.yml \
  policy/internvla_a1_5/deploy_insert_block.yml \
  --target_block half_cylinder \
  --block_base_pose_indices 1,3,2 \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

  CUDA_VISIBLE_DEVICES=4 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  task_config/gelsight.yml \
  policy/internvla_a1_5/deploy_insert_block.yml \
  --target_block hexagon \
  --block_base_pose_indices 0,1,4 \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=5 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  task_config/gelsight.yml \
  policy/internvla_a1_5/deploy_insert_block.yml \
  --target_block hexagon \
  --block_base_pose_indices 1,3,2 \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50
```