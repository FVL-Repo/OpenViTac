# OpenViTac OpenPI Client 接口说明

本文档说明 `policy/openpi` 这个 client 在 OpenViTac 仿真测评时会发送什么给 OpenPI server，以及期望从 server 接收什么。

## 使用入口

OpenViTac 的 `scripts/eval_policy.py` 会根据 deploy yaml 里的：

```yaml
policy_name: openpi
```

导入：

```python
importlib.import_module("policy.openpi")
```

实际 policy 类是：

```python
policy.openpi.Policy
```

默认配置在：

```text
policy/openpi/abs_joint/deploy.yml
```

为了区分不同 OpenPI server 输入模态，也提供了两个更明确的配置：

```text
policy/openpi/abs_joint/deploy_vision.yml          # 只发送 head + wrist 两路视觉图像
policy/openpi/abs_joint/deploy_vision_tactile.yml  # 发送 head + wrist + left/right tactile 四路图像
policy/openpi/abs_joint/deploy_swap_cup_order_vision.yml  # swap_cup_order 两路视觉版本
policy/openpi/abs_joint/deploy_swap_cup_order_vision_tactile.yml  # swap_cup_order 视触觉版本
policy/openpi/eef_delta/deploy_eef_delta_vision.yml  # EEF delta 两路视觉版本
policy/openpi/eef_delta/deploy_eef_delta_vision_tactile.yml  # EEF delta 视触觉版本
```

EEF delta 的控制语义、调用链和涉及代码见：

```text
policy/openpi/eef_delta/README.md
```

当前默认任务接口是 `abs_joint`，即 state 和 action 都是 8 维：

```text
[7 个 Franka arm joints, 1 个 gripper qpos]
```

也支持 `delta_eef`：

```text
state  = [ee_pos(3), ee_quat_xyzw(4), gripper_qpos(1)] # shape [8]
action = [delta_xyz(3), delta_rotvec(3), gripper_abs_qpos(1)] # shape [7]
```

注意：`delta_eef` 的 action 最后一维是绝对 gripper qpos，不是 delta。client 会在执行前用当前 gripper qpos 转成 OpenViTac 底层 `delta_ee_rotvec_ik` 接口需要的 delta gripper。原有 `delta_ee` Euler 接口保持不变。

## Client 连接方式

client 使用 `policy/openpi/client.py` 里的 `_OpenPiWebsocketClient`，协议和 `openpi-client` 的 `WebsocketClientPolicy` 保持一致，并复用：

```python
openpi_client.msgpack_numpy
```

连接配置来自 `deploy.yml`：

```yaml
openpi:
  host: ""
  port: 8000
  api_key:
  websocket_ping_interval:
  websocket_ping_timeout:
  websocket_open_timeout: 10
  websocket_close_timeout: 10
```

安装 client：

```bash
cd 
bash policy/openpi/install_client.sh
```

## 发送给 Server 的 Observation

每次需要重新查询动作时，client 会调用：

```python
openpi_obs_from_openvitac(observation, prompt, image_size, send_tactile, image_color_order)
```

发送给 server 的 dict 字段如下。

| Key | dtype | shape | 来源 | 说明 |
| --- | --- | --- | --- | --- |
| `observation/state` | `np.float32` | `[8]` | `observation["embodiment"]["joint"][:8]` | 7 个 arm joint + 第一个 gripper qpos |
| `observation/image` | `np.uint8` | `[image_size, image_size, 3]` | `observation["observation"]["head"]["rgb"]` | head camera |
| `observation/wrist_image` | `np.uint8` | `[image_size, image_size, 3]` | `observation["observation"]["wrist"]["rgb"]` | wrist camera |
| `observation/left_tactile_image` | `np.uint8` | `[image_size, image_size, 3]` | `observation["tactile"]["left_tactile"]["rgb_marker"]` | 左触觉图，`send_tactile=true` 时发送 |
| `observation/right_tactile_image` | `np.uint8` | `[image_size, image_size, 3]` | `observation["tactile"]["right_tactile"]["rgb_marker"]` | 右触觉图，`send_tactile=true` 时发送 |
| `prompt` | `str` | scalar | `openpi.prompt` | 语言指令 |

默认：

```yaml
openpi:
  prompt: "pick and insert the HDMI"
  # true 时优先发送 task.instruction；没有 task instruction 时才回退到上面的固定 prompt。
  # 对 insert_block/move_cup/place_cube_on_colored_area/grasp_in_clutter
  # 这类语义任务应保持 true，以便每个 case 向 server 发送不同 prompt。
  prompt_from_task_instruction: true
  image_size: 224
  send_tactile: true
  # 可选；不写时按 rgb_marker, gel_particle, force_field_img,
  # marker_force_img, rgb 的顺序自动选择。
  tactile_image_key:
  image_color_order: "rgb"
```

也就是说默认会发送 4 张图，都是 `224x224x3 uint8`。
在 OpenViTac 中，`gelsight` 和 `xense` 通常使用 `rgb_marker`，`neote`
通常使用 `gel_particle`，`neote_force_field` 通常使用 `force_field_img`。

## RGB/BGR 约定

在线仿真 observation 的 head/wrist 图来自 IsaacLab `TiledCamera.data.output["rgb"]`，触觉图来自 TacEx sensor 的 `marker_rgb`。当前 client 默认按 RGB 发送，不做 BGR->RGB。

如果发现训练 server 实际按照 BGR 数据训练，可以临时切换：

```yaml
openpi:
  image_color_order: "bgr"
```

这会在发送前交换 R/B 通道。

确认实际发送图片的方法：

```yaml
openpi:
  debug_dump_first_n_obs: 1
  debug_dump_dir: "debug/openpi_obs"
  debug_dump_bgr_interpretation: true
```

运行一次 eval 后会保存：

```text
debug/openpi_obs/*_as_rgb.png
debug/openpi_obs/*_as_bgr.png
```

`*_as_rgb.png` 是“按 RGB 解释”的图；`*_as_bgr.png` 是“同一数组交换 R/B 后”的对照图。哪张肉眼颜色正常，就说明当前数组语义更接近哪种通道顺序。

也可以离线检查 HDF5 训练图像：

```bash
cd 
conda activate OpenViTac
python policy/openpi/dump_hdf5_images.py data/insert_HDMI/demo/hdf5/0.hdf5 --frame 0
```

输出默认在：

```text
debug/openpi_hdf5_images/
```

## 期望 Server 返回的 Action

server 对一次 `infer(obs)` 的返回必须是 dict，并包含：

```python
{
    "actions": np.ndarray  # abs_joint/relative_joint: [T,8], delta_eef: [T,7]
}
```

## WebSocket 超时参数

OpenPI client 使用 websocket 和 server 通信。Isaac Sim 里如果一次 server 推理时间比较长，默认 websocket keepalive 可能会报：

```text
keepalive ping timeout
```

当前 deploy 默认关闭 ping keepalive：

```yaml
openpi:
  websocket_ping_interval:
  websocket_ping_timeout:
```

空值会被解析成 Python `None`。如果希望开启 keepalive，可以设置成秒数：

```yaml
openpi:
  websocket_ping_interval: 60
  websocket_ping_timeout: 120
```

相关参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `websocket_ping_interval` | 空值 | 每隔多少秒发一次 ping；空值表示关闭自动 ping |
| `websocket_ping_timeout` | 空值 | ping 后等待 pong 的超时；空值表示不因 pong 超时断开 |
| `websocket_open_timeout` | `10` | 建立连接超时秒数 |
| `websocket_close_timeout` | `10` | 关闭连接超时秒数 |

注意：Isaac Sim 运行后可能优先加载 Omniverse 预捆绑的 `websockets`。部分版本的 `websockets.sync.client.connect()` 不支持 `ping_interval/ping_timeout` 参数，client 会自动跳过不支持的参数，避免启动时报 `unexpected keyword argument`。

约束：

| 字段 | 要求 |
| --- | --- |
| `actions.ndim` | 必须是 2 |
| `actions.shape` | `abs_joint`/`relative_joint` 为 `[T,8]`；`delta_eef` 为 `[T,7]` |
| `T` | 必须大于等于 `open_loop_horizon` |
| 值域 | 不能包含 `NaN` 或 `Inf` |
| 语义 | 由 `control_mode` 决定 |

`abs_joint` 的 8 维 action 语义：

```text
actions[t] = [
  panda_joint1,
  panda_joint2,
  panda_joint3,
  panda_joint4,
  panda_joint5,
  panda_joint6,
  panda_joint7,
  gripper_qpos,
]
```

`relative_joint` 的 8 维 action 语义：

```text
actions[t] = [
  target_panda_joint1 - observation/state[0],
  target_panda_joint2 - observation/state[1],
  target_panda_joint3 - observation/state[2],
  target_panda_joint4 - observation/state[3],
  target_panda_joint5 - observation/state[4],
  target_panda_joint6 - observation/state[5],
  target_panda_joint7 - observation/state[6],
  gripper_abs_qpos,
]
```

这里的 relative 是“相对本次 infer 的 `observation/state`”，不是相邻两帧之间的 delta。client 会用本次 infer 时发送的 `observation/state` 作为 base，将整个 chunk 转成 absolute qpos8 后再缓存、temporal ensemble 和执行：

```text
target_qpos[:7] = observation/state[:7] + actions[t][:7]
target_qpos[7] = actions[t][7]
```

`delta_eef` 的 7 维 action 语义：

```text
actions[t] = [
  delta_x,
  delta_y,
  delta_z,
  delta_rotvec_x,
  delta_rotvec_y,
  delta_rotvec_z,
  gripper_abs_qpos,
]
```

其中 `delta_xyz` 单位是米，`delta_rotvec` 单位是弧度，`gripper_abs_qpos` 单位是米。旋转约定为：

```text
delta_rotvec = log(R_target * R_current^-1)
R_target = exp(delta_rotvec) * R_current
```

client 会对单步 action 做检查。`abs_joint` 使用：

```python
sanitize_abs_joint_action(action, task)
```

行为：

- 检查 action 是 8 维。
- 检查所有值 finite。
- 将最后一维 gripper qpos clip 到 `[0, task._robot_manager.gripper_max_qpos]`。
- 转成 `torch.float32`，放到 `task.device`。

然后执行：

```python
task.take_action(action, action_type="qpos")
```

`delta_eef` 使用：

```python
sanitize_delta_eef_action(action, task)
```

行为：

- 检查 action 是 7 维。
- 检查所有值 finite。
- 可选裁剪 `delta_xyz` 和 `delta_rotvec`。
- 将最后一维 `gripper_abs_qpos` clip 到 `[0, gripper_max_qpos]`。
- 用当前 gripper qpos 转成 `delta_gripper_qpos`。

然后执行：

```python
task.take_action(action, action_type="delta_ee_rotvec_ik")
```

## Action Chunk 消费

默认：

```yaml
openpi:
  open_loop_horizon: 20
```

此时 client 会向 server 请求一个 action chunk，要求 server 返回至少 `[20, 8]`。
然后 client 会连续执行前 20 个动作，消费完再请求下一次。

如果想模仿 ACT temporal aggregation 的 chunk 平滑，可以设置：

```yaml
openpi:
  open_loop_horizon: 5
  temporal_ensemble: true
  chunk_first_n: 20
  ensemble_K: 0.01
```

此时 client 每 `open_loop_horizon` 个 env step 重新 query server 一次。每次返回的 chunk 会保留下来：

```text
当前 step s query 得到 actions[0:20]
actions[0] 预测 step s
actions[1] 预测 step s+1
...
actions[19] 预测 step s+19
```

对于当前 step，如果多个历史 chunk 都预测过这个 step，client 会做指数加权平均：

```text
weights = exp(-ensemble_K * arange(num_predictions))
```

然后执行平滑后的 action。该逻辑同时支持 7D delta EEF 和 8D abs joint。

简要区别：

| 模式 | server query 频率 | 执行动作 | 适合场景 |
| --- | --- | --- | --- |
| `temporal_ensemble: false` 或未配置 | 每 `open_loop_horizon` 步 query 一次 | 直接顺序执行 chunk | 快、稳定，默认先用 |
| `temporal_ensemble: true` | 每 `open_loop_horizon` 步 query 一次 | 对重叠 chunk 做 ACT 风格平滑 | 想降低 action 抖动时使用 |

## EEF Delta 约定

当前 EEF deploy 使用：

```text
state:  [ee_pos(3), ee_quat_xyzw(4), gripper_qpos(1)] # shape [8]
action: [delta_xyz(3), delta_rotvec(3), gripper_qpos(1)]  # shape [7]
```

注意最后一维 `gripper_qpos` 是绝对夹爪 qpos，不是 delta。

OpenViTac 为 OpenPI rotvec action 新增的底层接口：

```python
task.take_action(action, action_type="delta_ee_rotvec_ik")
```

原有 `delta_ee` 仍保留为 Euler 增量接口，不用于 OpenPI `delta_eef` 配置。`delta_ee_rotvec` 也保留为 rotvec + planner 路径；当前 OpenPI EEF 默认使用 `delta_ee_rotvec_ik`，即 Differential IK servo 路径。

相关 deploy 参数：

```yaml
openpi:
  control_mode: "delta_eef"
  eef_action_type: "delta_ee_rotvec_ik"
  eef_servo_force: true
```

`eef_action_type: "delta_ee_rotvec_ik"` 表示每个 OpenPI action 只通过 Differential IK 计算一次 joint target，不走 cuRobo planner。`eef_servo_force: true` 表示将该 joint target 直接写入 DOF 位置，和现有 abs-joint eval 的直接 joint 写入方式一致。

当前实现中会把 `action[6]` 当作 `delta_gripper_qpos`：

```python
gripper_pos = self._robot_manager.get_gripper_qpos()
gripper_next_pos = gripper_pos + action[6]
```

所以如果 OpenPI server 返回的是绝对 gripper qpos，policy 层需要在调用 `take_action(..., action_type="delta_ee_rotvec_ik")` 前转换一次：

```python
abs_gripper = clip(server_action[6], 0.0, task._robot_manager.gripper_max_qpos)
delta_gripper = abs_gripper - task._robot_manager.get_gripper_qpos()
delta_ee_action = [
    delta_x,
    delta_y,
    delta_z,
    delta_rotvec_x,
    delta_rotvec_y,
    delta_rotvec_z,
    delta_gripper,
]
```

或者在 OpenViTac 底层新增一个更明确的 action type，例如 `delta_ee_abs_gripper`，直接接受：

```text
[delta_xyz(3), delta_rotvec(3), absolute_gripper_qpos(1)]
```

## 典型测评命令

推荐直接用 `scripts/eval_policy.py`，这样可以显式指定 seed 范围和测评轮数：

```bash
cd 
conda activate OpenViTac
OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py insert_USB demo openpi/abs_joint/deploy --start_seed 0 --max_seed 5 --total_num 5
```

如果 server 是两路 vision 版本，使用：

```bash
OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py insert_USB demo openpi/abs_joint/deploy_vision --start_seed 0 --max_seed 200 --total_num 100
```

结果会保存到：

```text
eval_result/openpi/insert_USB/deploy_vision/<当前时间>/
```

如果 server 是 vision+tactile 版本，使用：

```bash
OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py insert_USB demo openpi/abs_joint/deploy_vision_tactile --start_seed 0 --max_seed 200 --total_num 100
```

结果会保存到：

```text
eval_result/openpi/insert_USB/deploy_vision_tactile/<当前时间>/
```

如果 server 是 EEF delta 的 vision+tactile 版本，使用：

```bash
OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py insert_USB demo openpi/eef_delta/deploy_eef_delta_vision_tactile --start_seed 0 --max_seed 200 --total_num 100
```

结果会保存到：

```text
eval_result/openpi/insert_USB/deploy_eef_delta_vision_tactile/<当前时间>/
```

如果要测 `swap_cup_order` 的两路 vision 版本，使用：

```bash
OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py swap_cup_order demo openpi/abs_joint/deploy_swap_cup_order_vision --start_seed 0 --max_seed 200 --total_num 100
```

结果会保存到：

```text
eval_result/openpi/swap_cup_order/deploy_swap_cup_order_vision/<当前时间>/
```

如果要测 `swap_cup_order` 的 vision+tactile 版本，使用：

```bash
OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py swap_cup_order demo openpi/abs_joint/deploy_swap_cup_order_vision_tactile --start_seed 0 --max_seed 200 --total_num 100
```

结果会保存到：

```text
eval_result/openpi/swap_cup_order/deploy_swap_cup_order_vision_tactile/<当前时间>/
```

这条命令含义：

| 参数 | 示例值 | 说明 |
| --- | --- | --- |
| `insert_USB` | `insert_USB` | task 名字，对应 `envs/insert_USB.py` |
| `demo` | `demo` | task config，对应 `task_config/demo.yml` |
| `openpi/abs_joint/deploy` | `openpi/abs_joint/deploy` | deploy config，对应 `policy/openpi/abs_joint/deploy.yml` |
| `--start_seed` | `0` | 从 seed 0 开始测 |
| `--max_seed` | `5` | 最多尝试到 seed 5 |
| `--total_num` | `5` | 最多实际评测 5 个 episode |

seed 逻辑：

- 如果不传 `--start_seed`，默认不是从 0 开始，而是从 `1000000 * (1 + deploy.yml 里的 seed)` 开始。
- 当前 deploy 默认 `seed: 0`，所以不传 `--start_seed` 时会从 `1000000` 开始。
- 如果想明确测 seed 0 到 5，就传 `--start_seed 0 --max_seed 5`。
- `--total_num 5` 表示最多实际测 5 轮；如果某些 seed 被 `seeds.json` 标成不可用，会跳过。

输出位置：

```text
eval_result/openpi/insert_USB/deploy/<当前时间>/
```

其中：

```text
log.log
video/<seed>_success.mp4
video/<seed>_failed.mp4
video/<seed>_error.mp4
metadata.json
```

视频是否保存由 `task_config/demo.yml` 控制。当前 demo 配置里：

```yaml
video_frequency: 2
save_frequency: 2
```

所以会保存 eval 视频。

如果想 dump 第一帧实际发给 server 的 OpenPI observation 图像，deploy 里保持：

```yaml
openpi:
  debug_dump_first_n_obs: 1
  debug_dump_dir: "debug/openpi_obs"
```

输出在：

```text
debug/openpi_obs/
```

## Smooth + Horizon 20 示例

如果要开启 ACT 风格 temporal ensemble，并且使用 20 步窗口，可以直接使用
`policy/openpi/abs_joint/deploy_vision_tactile_step_smooth.yml`，或设置：

```yaml
openpi:
  host: ""
  port: 8000
  api_key:
  websocket_ping_interval:
  websocket_ping_timeout:
  websocket_open_timeout: 10
  websocket_close_timeout: 10
  prompt: "pick and insert the USB"
  action_dim: 8
  image_size: 224
  send_tactile: true
  image_color_order: "rgb"
  debug_dump_first_n_obs: 1
  debug_dump_dir: "debug/openpi_obs"
  debug_dump_bgr_interpretation: true
  open_loop_horizon: 5
  temporal_ensemble: true
  chunk_first_n: 20
  ensemble_K: 0.01
```

然后运行同一条测评命令：

```bash
cd 
conda activate OpenViTac
OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py insert_USB demo openpi/abs_joint/deploy --start_seed 0 --max_seed 5 --total_num 5
```

开启 smoothing 后，server 每 `open_loop_horizon` 个 env step 会被 query 一次，并且每次最好返回至少 `chunk_first_n` 个 action，保证相邻 chunk 有足够重叠用于平滑。

注意：运行前需要 OpenPI server 已经监听 `deploy.yml` 中的 `host:port`。
