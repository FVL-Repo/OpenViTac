# N0-VTLA OpenViTac deployment

The remote server is one policy instance per port. It keeps an episode-local tactile baseline, so do not share one server port between parallel evaluators.

On the remote machine:

```bash
cd /notebook/baselines/N0-VTLA
source /root/miniconda3/bin/activate vtla
# This machine already has the DINOv2 dependency cached. HF_HUB_OFFLINE avoids
# unnecessary Hub HEAD requests when its outbound network is unavailable.
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python scripts/serve_openvitac_sim_single_arm_ws.py \
  --checkpoint-dir checkpoints/sim_single_arm_tactile/openvitac_sim_single_arm_insert_USB/20000 \
  --port 8002 \
  --default-prompt "Pick up the USB plug from the blue slot and insert it into the red USB slot."

HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=4 python scripts/serve_openvitac_sim_single_arm_ws.py \
  --checkpoint-dir checkpoints/sim_single_arm_tactile/openvitac_sim_single_arm_insert_block_v1/20000 \
  --port 8014

HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python scripts/serve_openvitac_sim_single_arm_ws.py \
  --checkpoint-dir checkpoints/sim_single_arm_tactile/openvitac_sim_single_arm_roughness_regrasp/20000 \
  --port 8000

HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=7 python scripts/serve_openvitac_sim_single_arm_ws.py \
  --checkpoint-dir checkpoints/sim_single_arm_tactile/openvitac_sim_single_arm_roughness_regrasp/20000 \
  --port 8002

HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python scripts/serve_openvitac_sim_single_arm_ws.py \
  --checkpoint-dir checkpoints/sim_single_arm_tactile/openvitac_sim_single_arm_hardness_classify/20000 \
  --port 8000

HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=7 python scripts/serve_openvitac_sim_single_arm_ws.py \
  --checkpoint-dir checkpoints/sim_single_arm_tactile/openvitac_sim_single_arm_hardness_classify/20000 \
  --port 8002

HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python scripts/serve_openvitac_sim_single_arm_ws.py \
  --checkpoint-dir checkpoints/sim_single_arm_tactile/openvitac_sim_single_arm_weight_classify/20000 \
  --port 8004

HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=6 python scripts/serve_openvitac_sim_single_arm_ws.py \
  --checkpoint-dir checkpoints/sim_single_arm_tactile/openvitac_sim_single_arm_weight_classify/20000 \
  --port 8006

HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python scripts/serve_openvitac_sim_single_arm_ws.py \
  --checkpoint-dir checkpoints/sim_single_arm_tactile/openvitac_sim_single_arm_turn_gear_pair_3cm/20000 \
  --port 8000

HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python scripts/serve_openvitac_sim_single_arm_ws.py \
  --checkpoint-dir checkpoints/sim_single_arm_tactile/openvitac_sim_single_arm_roughness_classify/20000 \
  --port 8002

HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=6 python scripts/serve_openvitac_sim_single_arm_ws.py \
  --checkpoint-dir checkpoints/sim_single_arm_tactile/openvitac_sim_single_arm_roughness_classify/20000 \
  --port 8004
```

The server discovers the matching normalization-stat asset from
`CHECKPOINT_DIR/assets/*/norm_stats.json`. For this checkpoint it selects
`sim_single_arm_insert_USB`; pass `--asset-id ...` only for an ambiguous,
nonstandard checkpoint layout.

On the local machine, forward the port and run the standard evaluator:

```bash
ssh -N -L 8004:127.0.0.1:8004 -p <port> root@<ftp1-train-server>
ssh -N -L 8014:127.0.0.1:8014 -p <port> root@<ftp1-train-server>

cd /dataconda activate OpenViTac
CUDA_VISIBLE_DEVICES=3 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_USB task_config/gelsight.yml policy/n0-vtla/deploy_insert_USB.yml \
  --start_seed 10000 --max_seed 10099 --total_num 100

CUDA_VISIBLE_DEVICES=6 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  task_config/gelsight.yml \
  policy/n0-vtla/deploy_insert_block.yml \
  --target_block cube \
  --block_base_pose_indices 0,1,4 \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=7 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  task_config/gelsight.yml \
  policy/n0-vtla/deploy_insert_block.yml \
  --target_block cube \
  --block_base_pose_indices 1,3,2 \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=2 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  task_config/gelsight.yml \
  policy/n0-vtla/deploy_insert_block.yml \
  --target_block half_cylinder \
  --block_base_pose_indices 0,1,4 \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=3 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  task_config/gelsight.yml \
  policy/n0-vtla/deploy_insert_block.yml \
  --target_block half_cylinder \
  --block_base_pose_indices 1,3,2 \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

  CUDA_VISIBLE_DEVICES=4 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  task_config/gelsight.yml \
  policy/n0-vtla/deploy_insert_block.yml \
  --target_block hexagon \
  --block_base_pose_indices 0,1,4 \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=5 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_block \
  task_config/gelsight.yml \
  policy/n0-vtla/deploy_insert_block.yml \
  --target_block hexagon \
  --block_base_pose_indices 1,3,2 \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=4 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  roughness_regrasp \
  task_config/gelsight.yml \
  policy/n0-vtla/deploy_roughness_regrasp.yml \
  --rough_block_side right \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=5 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  roughness_regrasp \
  task_config/gelsight.yml \
  policy/n0-vtla/deploy_roughness_regrasp.yml \
  --rough_block_side left \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  hardness_classify \
  task_config/gelsight.yml \
  policy/n0-vtla/deploy_hardness_classify.yml \
  --hardness_label soft \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  hardness_classify \
  task_config/gelsight.yml \
  policy/n0-vtla/deploy_hardness_classify.yml \
  --hardness_label hard \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=2 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  weight_classify \
  task_config/gelsight.yml \
  policy/n0-vtla/deploy_weight_classify.yml \
  --weight_label light\
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=3 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  weight_classify \
  task_config/gelsight.yml \
  policy/n0-vtla/deploy_weight_classify.yml \
  --weight_label heavy\
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  turn_gear_pair \
  task_config/gelsight.yml \
  policy/n0-vtla/deploy_turn_gear_pair.yml \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 100

CUDA_VISIBLE_DEVICES=2 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  roughness_classify \
  task_config/gelsight.yml \
  policy/n0-vtla/deploy_roughness_classify.yml \
  --roughness_label rough \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50

CUDA_VISIBLE_DEVICES=3 OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  roughness_classify \
  task_config/gelsight.yml \
  policy/n0-vtla/deploy_roughness_classify.yml \
  --roughness_label smooth \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 50
```

`open_loop_horizon` controls how many of the 50 predicted actions are executed before the next request. The default is 5. The training converter's default gripper representation is the mean of the two Franka finger positions, and the client keeps that convention.
