# FTP-1 训练与部署说明

本文档记录当前 FTP-1 OpenViTac 任务的最新训练、norm stats、serve 部署、client 执行流程。重点说明动作空间语义、action chunk、serve/client 传输内容，以及重新训练 13 个任务时需要保持一致的配置。

## 代码位置

本地仿真 client 位于本仓库：

```text
policy/ftp-1/deploy_policy.py
policy/ftp-1/client.py
policy/ftp-1/transforms.py
policy/ftp-1/deploy_<task>.yml
```

远端 FTP-1 训练和 serve 仓库：

```text
ssh <ftp1-train-server>
/notebook/ftp1-policy
```

远端脚本命名规则：

```text
scripts_exp_zarr/openvitac/process/process_<task>.sh
scripts_exp_zarr/openvitac/compute/compute_norm_stats_<task>.sh
scripts_exp_zarr/openvitac/train/train_<task>_ftp1.sh
scripts_exp_zarr/openvitac/serve/serve_<task>_ftp1.sh
scripts_exp_zarr/openvitac/dataset_<task>.json
```

## 当前任务

```text
grasp_in_clutter
move_cup
pull_drawer
turn_gear_pair
hardness_classify
insert_block_v1
insert_block_v2
place_cube_on_colored_area
roughness_classify
weight_classify
insert_USB
pour_ball_to_cup
roughness_regrasp
```

原始数据统一位于：

```text
/data/sim/<task>
```

数据量约定：

```text
insert_block_v2: 300 episodes
其他 12 个任务: 100 episodes
总计: 1500 episodes
```

## 训练数据处理

parse 阶段把 OpenViTac hdf5 转成 FTP-1 使用的 zarr。普通任务的新 process 脚本会递归查找：

```text
/data/sim/<task>/**/*.hdf5
```

然后软链接到标准 staging 结构：

```text
/tmp/ftp1_openvitac_<task>_staging/<task>/demo/hdf5/<idx>.hdf5
```

这样即使原始数据有多层目录，或者不同子目录里有重复的 `0.hdf5`、`1.hdf5`，也不会互相覆盖。

`roughness_regrasp` 仍使用专用 parser：

```text
data_processing.parse_data_module.parse_data_openvitac_roughness_regrasp
```

它的当前原始数据实际位于：

```text
roughness_regrasp/GelSight_100/rough_left/grasp_left/*.hdf5
roughness_regrasp/GelSight_100/rough_left/grasp_right/*.hdf5
roughness_regrasp/GelSight_100/rough_right/grasp_left/*.hdf5
roughness_regrasp/GelSight_100/rough_right/grasp_right/*.hdf5
```

所以 `process_roughness_regrasp.sh` 会先在任务根目录统计 100 条 hdf5，再自动把传给 parser 的 `base_dir` 切到：

```text
/data/sim/roughness_regrasp/GelSight_100
```

parse 输出默认保存到：

```text
/run/ti/ftp1-training-data/ftp1_<task>_zarr
```

## 动作空间

FTP-1 模型内部默认 state/action 维度是 120：

```text
0:9      right wrist pose9d
9:16     right arm joints, 7D
16:48    right hand joints, 32-slot canonical hand vector
48:57    left wrist pose9d
57:64    left arm joints, 7D
64:96    left hand joints, 32-slot canonical hand vector
96:105   head/ego pose9d
105:120  supplementary/reserved joints
```

OpenViTac 单臂仿真任务最终部署只用 8 维：

```text
right arm 7D:       state/action[9:16]
right gripper qpos: state/action[16 + 28] = state/action[44]
```

parse 阶段写入 zarr 的是 absolute joint trajectory：

```text
right_arm_joints = joint[:, :7]
right_hand_joints = joint[:, 7:8]
right_hand_joints_idx = 28
```

relative action 不是在 parse 阶段提前写死的，而是在训练 dataset 取样时根据 `action_joint_rep` 动态构造。

## 训练动作表示

当前需要的训练动作空间是：

```text
action_joint_rep=mix
```

它对最终 8 维部署动作的语义是：

```text
action8[:, :7] = future_arm_qpos - current_arm_qpos
action8[:, 7]  = absolute_gripper_qpos
```

也就是：

```text
arm joints: relative
gripper: absolute qpos
```

模型输入的 proprioception/state 仍使用 absolute joint：

```text
proprioception_joint_rep=abs
```

所以训练时必须保证 compute norm stats 和 train 使用完全一致的表示：

```bash
PROPRIOCEPTION_POSE_REP=relative
ACTION_POSE_REP=relative
PROPRIOCEPTION_JOINT_REP=abs
ACTION_JOINT_REP=mix
NORM_TYPE=zscore
NORM_IMAGE_TACTILE_MODE=channel_wise
STATE_INPUT_MODE=adarms
```

## norm stats

计算 norm stats 的脚本：

```bash
bash scripts_exp_zarr/openvitac/compute/compute_norm_stats_<task>.sh
```

 compute 脚本显式传入：

```bash
--proprioception_pose_rep="${proprioception_pose_rep}"
--action_pose_rep="${action_pose_rep}"
--proprioception_joint_rep="${proprioception_joint_rep}"
--action_joint_rep="${action_joint_rep}"
```

其中默认：

```bash
action_joint_rep="${ACTION_JOINT_REP:-mix}"
```

这点非常关键。之前旧 checkpoint 里出现过：

```text
train_config.json: action_joint_rep=mix
normalization/norm_params_snapshot.json: action_joint_rep=relative
```

这个不一致会让夹爪维度的 normalization 按 relative gripper delta 统计，而训练目标却是 absolute gripper qpos。新脚本已经修正。

norm stats 输出到：

```text
assets/ftp1/<task>/
```

其中 `norm_params_snapshot.json` 会记录用于计算 stats 的关键配置。训练时会用它和当前 train config 对比。

## 训练

训练脚本：

```bash
bash scripts_exp_zarr/openvitac/train/train_<task>_ftp1.sh
```

 train 脚本默认：

```bash
EXP_NAME=<task>_mixnorm_train
CHECKPOINT_BASE_DIR=<repo>/checkpoints/finetune/
CUDA_VISIBLE_DEVICES=4,5,6,7
NUM_TRAIN_STEPS=40000
BATCH_SIZE=64
ACTION_JOINT_REP=mix
PROPRIOCEPTION_JOINT_REP=abs
```

checkpoint 默认写到新路径：

```text
checkpoints/finetune/ftp1/<task>_mixnorm_train/
```

新 train 脚本已经移除：

```bash
--no-check_norm_params_snapshot
```

因此训练会默认检查：

```text
assets/ftp1/<task>/norm_params_snapshot.json
```

如果 compute 和 train 的动作表示不一致，训练应该直接报错。不要再用 `--no-check_norm_params_snapshot` 绕过这个检查。

## 推荐训练流程

登录远端：

```bash
ssh <ftp1-train-server>
cd /notebook/ftp1-policy
```

单任务，例如 `insert_USB`：

```bash
bash scripts_exp_zarr/openvitac/process/process_insert_USB.sh
bash scripts_exp_zarr/openvitac/compute/compute_norm_stats_insert_USB.sh
bash scripts_exp_zarr/openvitac/train/train_insert_USB_ftp1.sh
```

批量运行：

```bash
tasks=(
  grasp_in_clutter
  move_cup
  pull_drawer
  turn_gear_pair
  hardness_classify
  insert_block_v1
  insert_block_v2
  place_cube_on_colored_area
  roughness_classify
  weight_classify
  insert_USB
  pour_ball_to_cup
  roughness_regrasp
)

for task in "${tasks[@]}"; do
  bash "scripts_exp_zarr/openvitac/process/process_${task}.sh"
  bash "scripts_exp_zarr/openvitac/compute/compute_norm_stats_${task}.sh"
  bash "scripts_exp_zarr/openvitac/train/train_${task}_ftp1.sh"
done
```

覆盖默认 GPU 或步数：

```bash
CUDA_VISIBLE_DEVICES=0,1 NUM_TRAIN_STEPS=20000 \
bash scripts_exp_zarr/openvitac/train/train_move_cup_ftp1.sh
```

## 模型输入输出

训练 sample 主要包含：

```python
{
    "image": dict,
    "image_mask": dict,
    "tactile": dict,
    "tactile_function_area": dict,
    "tactile_sensor": dict,
    "tactile_type": dict,
    "state": np.ndarray,       # usually (1, 120)
    "state_mask": np.ndarray,
    "actions": np.ndarray,     # (32, 120)
    "action_mask": np.ndarray,
    "prompt": str,
    "domain_name": str,
}
```

FTP-1 原生一次推理输出：

```text
raw_chunk.shape = (32, 120)
```

serve 会从 120 维中抽取右臂和右夹爪，形成：

```text
action8_model_chunk.shape = (32, 8)
```

然后根据 checkpoint 的 `train_config.json` 中的 `action_joint_rep` 转成 client 需要的 absolute qpos8。

当前新训练应为 `mix`，serve 转换规则是：

```python
resolved[:, :7] = qpos8_base[:7] + action8_model_chunk[:, :7]
resolved[:, 7] = action8_model_chunk[:, 7]
```

因此 serve 返回给 client 的不是 relative/mix 原始模型输出，而是：

```text
absolute_qpos8
```

## 远端 serve

serve 脚本：

```bash
bash scripts_exp_zarr/openvitac/serve/serve_<task>_ftp1.sh
```

例如：

```bash
cd /notebook/ftp1-policy
bash scripts_exp_zarr/openvitac/serve/serve_insert_USB_ftp1.sh
```

默认使用：

```bash
CHECKPOINT_DIR=checkpoints/finetune/ftp1/<task>_mixnorm_train/39999
DOMAIN_NAME=<task>
ACTION_REP=auto
NUM_INFERENCE_STEPS=10
IMAGE_SIZE=224
HOST=0.0.0.0
```

`ACTION_REP=auto` 表示 serve 会读取：

```text
<checkpoint>/train_config.json
```

自动判断 `action_joint_rep`。如果 checkpoint 是当前新训练，应读到：

```text
mix
```

serve 默认端口：

```text
grasp_in_clutter:              8000
move_cup:                      8001
pull_drawer:                   8002
turn_gear_pair:                8003
hardness_classify:             8004
insert_block_v1:               8005
insert_block_v2:               8006
place_cube_on_colored_area:    8007
roughness_classify:            8008
weight_classify:               8009
insert_USB:                    8010
pour_ball_to_cup:              8011
roughness_regrasp:             8012
```

可以覆盖端口和 GPU：

```bash
PORT=8004 CUDA_VISIBLE_DEVICES=2 \
bash scripts_exp_zarr/openvitac/serve/serve_insert_USB_ftp1.sh
```

## client 部署配置

本地仿真通过 `scripts/eval_policy.py` 加载：

```text
policy/ftp-1/deploy_<task>.yml
```

示例：

```bash
CUDA_VISIBLE_DEVICES=2 OMNI_KIT_ACCEPT_EULA=yes \
python scripts/eval_policy.py \
  insert_USB \
  task_config/gelsight.yml \
  policy/ftp-1/deploy_insert_USB.yml \
  --start_seed 10000 \
  --max_seed 10999 \
  --total_num 100
```

deploy yml 中的关键配置：

```yaml
ftp_1:
  host: "127.0.0.1"
  port: 8004
  prompt_from_task_instruction: true
  action_schema: "absolute_qpos8"
  action_dim: 8
  execute_from_index: 1
  open_loop_horizon: 5
  temporal_ensemble: true
  temporal_ensemble_horizon: 20
  temporal_ensemble_k: 0.01
  tactile_image_key: rgb_marker
```

注意：本地某些 `deploy_<task>.yml` 的端口沿用旧实验，不一定等于新远端 serve 脚本的默认端口。部署时必须保证 client 的 `ftp_1.port` 和远端 serve 实际监听端口一致。可以选择：

```text
1. 修改本地 deploy yml 的 port
2. 启动远端 serve 时用 PORT=<client_port> 覆盖
```

如果 serve 在远端机器，本地 client 通常通过 SSH 端口转发连接：

```bash
ssh -N -L <local_port>:127.0.0.1:<remote_port> <ftp1-train-server>
```

此时本地 deploy yml 写：

```yaml
host: "127.0.0.1"
port: <local_port>
```

## client 发送给 serve 的内容

本地 `deploy_policy.py` 调用 `ftp_obs_from_openvitacorge`，把仿真 observation 转成 websocket/msgpack 请求：

```python
{
    "prompt": str,
    "qpos": np.ndarray,                       # (8,), float32
    "camera_ego_rgb": np.ndarray,             # (H, W, 3), uint8
    "right_wrist_camera_rgb": np.ndarray,     # (H, W, 3), uint8
    "right_tactile_data_gripper": np.ndarray, # (2, H, W, 3), uint8
}
```

字段来源：

```text
qpos = observation["embodiment"]["joint"][:8]
camera_ego_rgb = observation["observation"]["head"]["rgb"]
right_wrist_camera_rgb = observation["observation"]["wrist"]["rgb"]
right_tactile_data_gripper = stack(left_tactile, right_tactile)
prompt = task.instruction if prompt_from_task_instruction=true else yml prompt
```

图像会被整理成 HWC、3 通道、uint8。触觉图默认使用 `rgb_marker`，也可以通过 `tactile_image_key` 配置。

## serve 接收后做什么

远端 `serve_ftp1_sim_policy.py` 收到请求后：

```text
1. 读取 qpos8，作为当前 absolute state
2. 构造 FTP-1 120D state
3. 把 head/wrist RGB resize 到 image_size，默认 224
4. 把 tactile 整理成 (1, 2, 224, 224, 3)
5. 调用 FTP1InferenceWrapper.infer(...)
6. 得到 raw_chunk: (32, 120)，已经 denormalize 回 raw scale
7. 抽取 right arm [9:16] 和 gripper slot 44，得到 (32, 8)
8. 根据 action_joint_rep 转成 absolute_qpos8
9. 返回 msgpack response
```

返回格式：

```python
{
    "actions": np.ndarray,              # (32, 8), float32, absolute qpos8
    "action_schema": "absolute_qpos8",
    "action_horizon": 32,
    "execute_from_index": 1,
    "chunk_zero_semantics": "placeholder_current_timestep",
    "server_timing": dict,
}
```

其中 `actions` 已经是 absolute qpos8。client 不再做 relative 到 absolute 的转换。

## client 收到后如何执行

client 收到 response 后会检查：

```text
action_schema == "absolute_qpos8"
actions.shape == (T, 8)
execute_from_index >= 0
actions 中没有 NaN/Inf
```

然后根据是否启用 temporal ensemble 分两种执行方式。

不启用 `temporal_ensemble` 时：

```python
executable = actions[execute_from_index:]
action = executable[chunk_step]
```

如果 `execute_from_index=1`，就是丢弃 `actions[0]`，从 `actions[1]` 开始执行。

启用 `temporal_ensemble` 时：

```text
policy step 0: 请求 chunk A，执行 A[1]
policy step 1: 执行 A[2]
policy step 2: 执行 A[3]
policy step 3: 执行 A[4]
policy step 4: 执行 A[5]
policy step 5: 请求 chunk B，融合 A[6] 和 B[1]
```

`open_loop_horizon=5` 表示每 5 个 policy step 请求一次新 chunk。它不改变模型输出长度，模型一次仍输出 32 步。

`temporal_ensemble_horizon=20` 表示旧 chunk 在最多 20 个相对步长内可参与融合。`temporal_ensemble_k=0.01` 控制指数权重：

```python
weights = exp(-k * arange(num_candidates))
weights = weights / weights.sum()
```

最后 client 会调用：

```python
sanitize_qpos8_action(action, task)
task.take_action(torch_action, action_type="qpos")
```

`sanitize_qpos8_action` 只做基本检查和夹爪裁剪：

```text
action shape 必须是 8
不能有 NaN/Inf
gripper qpos clip 到 [0, task._robot_manager.gripper_max_qpos]
```

仿真执行时：

```text
action[:7] -> robot arm qpos target
action[7]  -> gripper qpos target
```

## execute_from_index 的含义

`execute_from_index=1` 的含义是：

```text
actions[0] 不执行
从 actions[1] 开始执行
```

原因来自远端训练 dataset 的 action chunk 构造。`src/openpi/dataset_zarr.py` 中 `action_idx_slice` 从当前样本 index 开始：

```python
slice_end = min(end_idx, idx + (action_horizon - 1) * action_down_sample_steps + 1)
action_idx_slice = np.arange(idx, slice_end, action_down_sample_steps)
```

当前 state 的 base 也是当前 `idx`。因此在 `action_joint_rep=mix` 下：

```text
actions[0][:7] = qpos[idx][:7] - qpos[idx][:7] = 0
actions[0][7]  = gripper_qpos[idx]
```

serve 再把它转成 absolute qpos8 后：

```text
returned actions[0] == 当前 qpos8
```

所以 `actions[0]` 是当前时刻对齐项，不是下一步控制目标。index 从 0 开始，`execute_from_index=1` 才表示从第一个未来动作开始执行。

结合当前默认：

```yaml
execute_from_index: 1
open_loop_horizon: 5
temporal_ensemble: true
```

首次请求后，前 5 个 policy step 使用的是：

```text
actions[1], actions[2], actions[3], actions[4], actions[5]
```

之后每 5 步重新请求新 chunk，并按 temporal ensemble 融合新旧 chunk 在当前时刻的预测。

## 部署前检查清单

训练前：

```text
process 已生成 /run/ti/ftp1-training-data/ftp1_<task>_zarr
dataset_<task>.json 顶层是 {"datasets": [...]}
compute norm stats 已按 ACTION_JOINT_REP=mix 跑完
assets/ftp1/<task>/norm_params_snapshot.json 中 action_joint_rep=mix
train 脚本没有 --no-check_norm_params_snapshot
```

serve 前：

```text
CHECKPOINT_DIR 指向新的 <task>_mixnorm_train/<step>
checkpoint/train_config.json 中 action_joint_rep=mix
serve DOMAIN_NAME=<task>
serve PORT 与 client deploy yml 的 ftp_1.port 一致
ACTION_REP=auto
```

client 前：

```text
SSH 端口转发已经建立，或 client 可以直接访问 serve
deploy yml 的 host/port 正确
action_schema=absolute_qpos8
action_dim=8
execute_from_index=1
tactile_image_key 与仿真 observation 中的触觉图 key 对齐
```
