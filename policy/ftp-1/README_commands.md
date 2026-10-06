```bash

ssh -N -L 8000:127.0.0.1:8000 -p <ftp1-train-server-port> root@<ftp1-train-server>

ssh -N -L 8008:127.0.0.1:8008 -p <port> root@<ftp1-train-server>
ssh -N -L 8010:127.0.0.1:8010 -p <port> root@<ftp1-train-server>
ssh -N -L 8012:127.0.0.1:8012 -p <port> root@<ftp1-train-server>
ssh -N -L 8014:127.0.0.1:8014 -p <port> root@<ftp1-train-server>

ssh -N -L 8016:127.0.0.1:8016 -p <port> root@<ftp1-train-server>
ssh -N -L 8018:127.0.0.1:8018 -p <port> root@<ftp1-train-server>
ssh -N -L 8020:127.0.0.1:8020 -p <port> root@<ftp1-train-server>
ssh -N -L 8022:127.0.0.1:8022 -p <port> root@<ftp1-train-server>

cd /dataconda activate OpenViTac

CUDA_VISIBLE_DEVICES=7 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  turn_gear_pair \
  task_config/gelsight.yml \
  policy/ftp-1/deploy_turn_gear_pair.yml \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 100

CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  can_empty_select \
  task_config/gelsight.yml \
  policy/ftp-1/deploy_can_empty_select.yml \
  --empty_can coke \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  can_empty_select \
  task_config/gelsight.yml \
  policy/ftp-1/deploy_can_empty_select.yml \
  --empty_can fanta \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=6 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  can_empty_select \
  task_config/gelsight.yml \
  policy/ftp-1/deploy_can_empty_select.yml \
  --empty_can 7up \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  can_empty_select \
  task_config/gelsight.yml \
  policy/ftp-1/deploy_can_empty_select.yml \
  --empty_can pepsi \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=6 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  pull_drawer \
  task_config/gelsight.yml \
  policy/ftp-1/deploy_pull_drawer.yml \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 100 \
  --background base4

CUDA_VISIBLE_DEVICES=7 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  roughness_regrasp \
  task_config/gelsight.yml \
  policy/ftp-1/deploy_roughness_regrasp.yml \
  --rough_block_side right \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=6 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  roughness_regrasp \
  task_config/gelsight.yml \
  policy/ftp-1/deploy_roughness_regrasp.yml \
  --rough_block_side left \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=7 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_USB \
  task_config/gelsight.yml \
  policy/ftp-1/deploy_insert_USB.yml \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 100

CUDA_VISIBLE_DEVICES=3 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  roughness_classify \
  task_config/gelsight.yml \
  policy/ftp-1/deploy_roughness_classify.yml \
  --roughness_label rough \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=3 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  roughness_classify \
  task_config/gelsight.yml \
  policy/ftp-1/deploy_roughness_classify.yml \
  --roughness_label smooth \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  grasp_in_clutter \
  task_config/gelsight.yml \
  policy/ftp-1/deploy_grasp_in_clutter.yml \
  --target_block block_blue_half_cylinder \
  --block_base_pose_indices 0,1,2,3,4,5,6 \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  grasp_in_clutter \
  task_config/gelsight.yml \
  policy/ftp-1/deploy_grasp_in_clutter.yml \
  --target_block block_blue_half_cylinder \
  --block_base_pose_indices 2,3,4,5,6,0,1 \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  grasp_in_clutter \
  task_config/gelsight.yml \
  policy/ftp-1/deploy_grasp_in_clutter.yml \
  --target_block block_yellow_cylinder \
  --block_base_pose_indices 0,1,2,3,4,5,6 \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=4 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  grasp_in_clutter \
  task_config/gelsight.yml \
  policy/ftp-1/deploy_grasp_in_clutter.yml \
  --target_block block_yellow_cylinder \
  --block_base_pose_indices 2,3,4,5,6,0,1 \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=6 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  hardness_classify \
  task_config/gelsight.yml \
  policy/ftp-1/deploy_hardness_classify.yml \
  --hardness_label soft \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=7 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  hardness_classify \
  task_config/gelsight.yml \
  policy/ftp-1/deploy_hardness_classify.yml \
  --hardness_label hard \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=5 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  weight_classify \
  task_config/gelsight.yml \
  policy/ftp-1/deploy_weight_classify.yml \
  --weight_label light\
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=4 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  weight_classify \
  task_config/gelsight.yml \
  policy/ftp-1/deploy_weight_classify.yml \
  --weight_label heavy\
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=4 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  move_cup \
  task_config/gelsight.yml \
  policy/ftp-1/deploy_move_cup.yml \
  --target_cup blue \
  --reference_cup yellow \
  --placement_side left \
  --cup_base_pose_indices 0,1,2 \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=6 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  move_cup \
  task_config/gelsight.yml \
  policy/ftp-1/deploy_move_cup.yml \
  --target_cup blue \
  --reference_cup yellow \
  --placement_side left \
  --cup_base_pose_indices 1,2,0 \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50
  
  CUDA_VISIBLE_DEVICES=5 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  move_cup \
  task_config/gelsight.yml \
  policy/ftp-1/deploy_move_cup.yml \
  --target_cup green \
  --reference_cup blue \
  --placement_side right \
  --cup_base_pose_indices 0,1,2 \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

  CUDA_VISIBLE_DEVICES=5 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  move_cup \
  task_config/gelsight.yml \
  policy/ftp-1/deploy_move_cup.yml \
  --target_cup green \
  --reference_cup blue \
  --placement_side right \
  --cup_base_pose_indices 1,2,0 \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

  CUDA_VISIBLE_DEVICES=4 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  place_cube_on_colored_area \
  task_config/gelsight.yml \
  policy/ftp-1/deploy_place_cube_on_colored_area.yml \
  --target_area yellow \
  --frame_order blue_left \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50 \
  --background base4

  CUDA_VISIBLE_DEVICES=5 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  place_cube_on_colored_area \
  task_config/gelsight.yml \
  policy/ftp-1/deploy_place_cube_on_colored_area.yml \
  --target_area yellow \
  --frame_order yellow_left \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50 \
  --background base4

CUDA_VISIBLE_DEVICES=6 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  place_cube_on_colored_area \
  task_config/gelsight.yml \
  policy/ftp-1/deploy_place_cube_on_colored_area.yml \
  --target_area blue \
  --frame_order yellow_left \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50 \
  --background base4

CUDA_VISIBLE_DEVICES=7 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  place_cube_on_colored_area \
  task_config/gelsight.yml \
  policy/ftp-1/deploy_place_cube_on_colored_area.yml \
  --target_area blue \
  --frame_order blue_left \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50 \
  --background base4

CUDA_VISIBLE_DEVICES=5 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  pour_ball_to_cup \
  task_config/gelsight.yml \
  policy/ftp-1/deploy_pour_ball_to_cup.yml \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 100 \
  --background base4

CUDA_VISIBLE_DEVICES=4 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  task_config/gelsight.yml \
  policy/ftp-1/deploy_insert_block.yml \
  --target_block cube \
  --block_base_pose_indices 0,1,4 \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=5 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  task_config/gelsight.yml \
  policy/ftp-1/deploy_insert_block.yml \
  --target_block cube \
  --block_base_pose_indices 1,3,2 \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=6 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  task_config/gelsight.yml \
  policy/ftp-1/deploy_insert_block.yml \
  --target_block half_cylinder \
  --block_base_pose_indices 0,1,4 \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=7 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  task_config/gelsight.yml \
  policy/ftp-1/deploy_insert_block.yml \
  --target_block half_cylinder \
  --block_base_pose_indices 1,3,2 \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

  CUDA_VISIBLE_DEVICES=4 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  task_config/gelsight.yml \
  policy/ftp-1/deploy_insert_block.yml \
  --target_block hexagon \
  --block_base_pose_indices 0,1,4 \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=5 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  task_config/gelsight.yml \
  policy/ftp-1/deploy_insert_block.yml \
  --target_block hexagon \
  --block_base_pose_indices 1,3,2 \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50
  ```