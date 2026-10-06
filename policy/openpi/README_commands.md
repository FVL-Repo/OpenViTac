ssh -N -L 9000:127.0.0.1:9000 -p <ftp1-train-server-port> root@<ftp1-train-server>
ssh -N -L 9001:127.0.0.1:9001 -p <ftp1-train-server-port> root@<ftp1-train-server>
ssh -N -L 9002:127.0.0.1:9002 -p <ftp1-train-server-port> root@<ftp1-train-server>
ssh -N -L 9003:127.0.0.1:9003 -p <ftp1-train-server-port> root@<ftp1-train-server>

cd 
conda activate OpenViTac

CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  turn_gear_pair \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_step_smooth.yml \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 100 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=6 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  turn_gear_pair \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_tactile_step_smooth.yml \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 100 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=4 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  pull_drawer \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_step_smooth.yml \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 100 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9002 \
  --background base4

<!-- CUDA_VISIBLE_DEVICES=4 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  pull_drawer \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_tactile_step_smooth.yml \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 100 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9002 \
  --background base4 -->

CUDA_VISIBLE_DEVICES=2 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  roughness_regrasp \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_step_smooth.yml \
  --rough_block_side right \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=3 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  roughness_regrasp \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_step_smooth.yml \
  --rough_block_side left \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

<!-- CUDA_VISIBLE_DEVICES=5 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
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
  --openpi_port 9001 -->

CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_USB \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_step_smooth.yml \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 100 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_USB \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_tactile_step_smooth.yml \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 100 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=3 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  roughness_classify \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_step_smooth.yml \
  --roughness_label rough \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=3 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  roughness_classify \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_tactile_step_smooth.yml \
  --roughness_label rough \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=3 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  roughness_classify \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_step_smooth.yml \
  --roughness_label smooth \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=3 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  roughness_classify \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_tactile_step_smooth.yml \
  --roughness_label smooth \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  grasp_in_clutter \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_step_smooth.yml \
  --target_block block_blue_half_cylinder \
  --block_base_pose_indices 0,1,2,3,4,5,6 \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  grasp_in_clutter \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_tactile_step_smooth.yml \
  --target_block block_blue_half_cylinder \
  --block_base_pose_indices 0,1,2,3,4,5,6 \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  grasp_in_clutter \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_step_smooth.yml \
  --target_block block_blue_half_cylinder \
  --block_base_pose_indices 2,3,4,5,6,0,1 \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  grasp_in_clutter \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_tactile_step_smooth.yml \
  --target_block block_blue_half_cylinder \
  --block_base_pose_indices 2,3,4,5,6,0,1 \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  grasp_in_clutter \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_step_smooth.yml \
  --target_block block_yellow_cylinder \
  --block_base_pose_indices 0,1,2,3,4,5,6 \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  grasp_in_clutter \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_tactile_step_smooth.yml \
  --target_block block_yellow_cylinder \
  --block_base_pose_indices 0,1,2,3,4,5,6 \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=4 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  grasp_in_clutter \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_step_smooth.yml \
  --target_block block_yellow_cylinder \
  --block_base_pose_indices 2,3,4,5,6,0,1 \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=4 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  grasp_in_clutter \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_tactile_step_smooth.yml \
  --target_block block_yellow_cylinder \
  --block_base_pose_indices 2,3,4,5,6,0,1 \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=5 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  hardness_classify \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_step_smooth.yml \
  --hardness_label soft \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=5 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  hardness_classify \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_tactile_step_smooth.yml \
  --hardness_label soft \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=5 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  hardness_classify \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_step_smooth.yml \
  --hardness_label hard \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=5 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  hardness_classify \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_tactile_step_smooth.yml \
  --hardness_label hard \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=5 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  weight_classify \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_step_smooth.yml \
  --weight_label light \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=5 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  weight_classify \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_tactile_step_smooth.yml \
  --weight_label light \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=5 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  weight_classify \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_step_smooth.yml \
  --weight_label heavy \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=5 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  weight_classify \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_tactile_step_smooth.yml \
  --weight_label heavy \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=5 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  move_cup \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_step_smooth.yml \
  --target_cup blue \
  --reference_cup yellow \
  --placement_side left \
  --cup_base_pose_indices 0,1,2 \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=5 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  move_cup \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_tactile_step_smooth.yml \
  --target_cup blue \
  --reference_cup yellow \
  --placement_side left \
  --cup_base_pose_indices 0,1,2 \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=5 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  move_cup \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_step_smooth.yml \
  --target_cup blue \
  --reference_cup yellow \
  --placement_side left \
  --cup_base_pose_indices 1,2,0 \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=5 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  move_cup \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_tactile_step_smooth.yml \
  --target_cup blue \
  --reference_cup yellow \
  --placement_side left \
  --cup_base_pose_indices 1,2,0 \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=5 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  move_cup \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_step_smooth.yml \
  --target_cup green \
  --reference_cup blue \
  --placement_side right \
  --cup_base_pose_indices 0,1,2 \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=5 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  move_cup \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_tactile_step_smooth.yml \
  --target_cup green \
  --reference_cup blue \
  --placement_side right \
  --cup_base_pose_indices 0,1,2 \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=5 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  move_cup \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_step_smooth.yml \
  --target_cup green \
  --reference_cup blue \
  --placement_side right \
  --cup_base_pose_indices 1,2,0 \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=5 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  move_cup \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_tactile_step_smooth.yml \
  --target_cup green \
  --reference_cup blue \
  --placement_side right \
  --cup_base_pose_indices 1,2,0 \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=4 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  place_cube_on_colored_area \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_step_smooth.yml \
  --target_area yellow \
  --frame_order blue_left \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000 \
  --background base4

CUDA_VISIBLE_DEVICES=4 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  place_cube_on_colored_area \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_tactile_step_smooth.yml \
  --target_area yellow \
  --frame_order blue_left \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000 \
  --background base4

CUDA_VISIBLE_DEVICES=5 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  place_cube_on_colored_area \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_step_smooth.yml \
  --target_area yellow \
  --frame_order yellow_left \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000 \
  --background base4

CUDA_VISIBLE_DEVICES=5 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  place_cube_on_colored_area \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_tactile_step_smooth.yml \
  --target_area yellow \
  --frame_order yellow_left \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000 \
  --background base4

CUDA_VISIBLE_DEVICES=6 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  place_cube_on_colored_area \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_step_smooth.yml \
  --target_area blue \
  --frame_order yellow_left \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000 \
  --background base4

CUDA_VISIBLE_DEVICES=6 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  place_cube_on_colored_area \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_tactile_step_smooth.yml \
  --target_area blue \
  --frame_order yellow_left \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000 \
  --background base4

CUDA_VISIBLE_DEVICES=7 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  place_cube_on_colored_area \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_step_smooth.yml \
  --target_area blue \
  --frame_order blue_left \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000 \
  --background base4

CUDA_VISIBLE_DEVICES=7 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  place_cube_on_colored_area \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_tactile_step_smooth.yml \
  --target_area blue \
  --frame_order blue_left \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000 \
  --background base4

CUDA_VISIBLE_DEVICES=4 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  pour_ball_to_cup \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_step_smooth.yml \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 100 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000 \
  --background base4

<!-- CUDA_VISIBLE_DEVICES=3 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  pour_ball_to_cup \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_tactile_step_smooth.yml \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 100 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9003 \
  --background base4 -->

CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_step_smooth.yml \
  --target_block cube \
  --block_base_pose_indices 0,1,4 \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_tactile_step_smooth.yml \
  --target_block cube \
  --block_base_pose_indices 0,1,4 \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_step_smooth.yml \
  --target_block cube \
  --block_base_pose_indices 1,3,2 \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_tactile_step_smooth.yml \
  --target_block cube \
  --block_base_pose_indices 1,3,2 \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=6 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_step_smooth.yml \
  --target_block half_cylinder \
  --block_base_pose_indices 0,1,4 \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=6 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_tactile_step_smooth.yml \
  --target_block half_cylinder \
  --block_base_pose_indices 0,1,4 \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=6 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_step_smooth.yml \
  --target_block half_cylinder \
  --block_base_pose_indices 1,3,2 \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=6 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_tactile_step_smooth.yml \
  --target_block half_cylinder \
  --block_base_pose_indices 1,3,2 \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=6 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_step_smooth.yml \
  --target_block hexagon \
  --block_base_pose_indices 0,1,4 \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=6 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_tactile_step_smooth.yml \
  --target_block hexagon \
  --block_base_pose_indices 0,1,4 \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=6 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_step_smooth.yml \
  --target_block hexagon \
  --block_base_pose_indices 1,3,2 \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000

CUDA_VISIBLE_DEVICES=6 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  task_config/gelsight.yml \
  policy/openpi/abs_joint/deploy_vision_tactile_step_smooth.yml \
  --target_block hexagon \
  --block_base_pose_indices 1,3,2 \
  --start_seed 10000 \
  --max_seed -1 \
  --total_num 50 \
  --openpi_host 127.0.0.1 \
  --openpi_port 9000
