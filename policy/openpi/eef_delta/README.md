# OpenPI Delta EEF 部署说明

本目录的 `deploy_eef_delta_vision.yml` 和
`deploy_eef_delta_vision_tactile.yml` 用于 OpenPI 的 `delta_eef` 部署。

当前默认控制方式不是 `move()` / cuRobo 轨迹规划，而是：

```text
OpenPI delta EEF action
-> Differential IK
-> 7D arm joint position target
-> 直接写入 Franka DOF position
-> 仿真推进 1 个 step
```

因此它和 OpenViTac 原有 `abs_joint` eval 一样，最终都通过 joint position 直接驱动机器人；区别只在于 action 的来源：

```text
abs_joint:
OpenPI 直接返回 [7 arm joint qpos, gripper qpos]

delta_eef:
OpenPI 返回 EEF delta，OpenViTac 先用 IK 转成 7 arm joint qpos
```

## 1. Server 输入和输出

发送给 OpenPI server 的 state：

```text
state = [
  ee_x, ee_y, ee_z,
  ee_qx, ee_qy, ee_qz, ee_qw,
  gripper_qpos,
]  # float32, shape [8]
```

server 每次返回 action chunk：

```text
actions.shape = [T, 7]

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

单位和坐标约定：

```text
delta_xyz: meter，机器人 base/world 坐标系
delta_rotvec: rad
delta_rotvec = log(R_target * R_current^-1)
R_target = exp(delta_rotvec) * R_current
gripper_abs_qpos: meter，绝对夹爪开度，不是 delta
```

## 2. 当前调用链

### A. 组装 observation

文件：

```text
policy/openpi/transforms.py
```

函数：

```python
state8_eef_quat_xyzw_from_openvitac_observation(observation)
openpi_obs_from_openvitac(...)
```

`embodiment/ee` 原始格式是：

```text
[x, y, z, qw, qx, qy, qz]
```

部署时转换为 `ee_pos(3) + quat_xyzw(4) + gripper_qpos(1)`，并连同 head/wrist 图像发送给 server；视触觉配置额外发送左右触觉图像。

### B. 校验 action 和处理绝对 gripper

文件：

```text
policy/openpi/transforms.py
```

函数：

```python
sanitize_delta_eef_action(action, task)
```

这一步做三件事：

```text
1. 检查 server action 是有限的 7D 数值。
2. 保持 delta_xyz 和 delta_rotvec 的数值语义。
3. 将 gripper_abs_qpos 转成 delta_gripper_qpos：
   delta_gripper = gripper_abs_qpos - current_gripper_qpos
```

OpenViTac 底层 servo 接口接收的是这个 7D 格式：

```text
[delta_xyz(3), delta_rotvec(3), delta_gripper_qpos(1)]
```

### C. 选择 EEF IK action type

文件：

```text
policy/openpi/deploy_policy.py
```

代码块：

```python
elif self.control_mode == "delta_eef":
    torch_action = sanitize_delta_eef_action(action, task)
    task.take_action(
        torch_action,
        action_type="delta_ee_rotvec_ik",
        force=True,
    )
```

默认参数也在该文件中：

```python
eef_action_type = "delta_ee_rotvec_ik"
eef_servo_force = True
```

因此新的 `delta_eef` 配置即使不显式写这些字段，也默认走 IK + 直接 joint 写入。

### D. Differential IK 转 joint position

文件：

```text
envs/_base_task.py
```

代码块：

```python
elif action_type == "delta_ee_rotvec_ik":
    self._robot_manager.servo_delta_ee_rotvec(action, force=force)
    self._step()
```

文件：

```text
envs/robot/robot.py
```

函数：

```python
RobotManager.servo_delta_ee_rotvec(action, force=True)
```

该函数依次：

```text
1. 读取当前 panda_hand 在 robot base 坐标系的 pose。
2. 用 delta_xyz 和 delta_rotvec 构造 target EEF pose。
3. 读取 panda_hand Jacobian 和当前 7D arm joint qpos。
4. 用 IsaacLab DifferentialIKController.compute(...) 算出 7D joint target。
5. 调 set_arm(joint_pos_des, force=True)。
6. 调 set_gripper(target_gripper, force=True)。
```

`force=True` 时，`RobotManager.set_arm()` 和 `set_gripper()` 除了设置 position target，也会调用 PhysX 的 DOF position 写入接口。因此这一路不会调用 `self.move()`，也不会执行 cuRobo 规划出的长 dense trajectory。

## 3. 默认部署配置

当前 EEF YAML 的关键字段：

```yaml
openpi:
  control_mode: "delta_eef"
  eef_action_type: "delta_ee_rotvec_ik"
  eef_servo_force: true
  state_dim: 8
  action_dim: 7
  open_loop_horizon: 20
```

一个 OpenPI action 对应：

```text
一次 IK 求解
一次直接 joint 写入
一次仿真 step
```

`open_loop_horizon: 20` 仅控制多久向 server 请求一次 action chunk，不会把 20 个 delta 预先相加。每个 delta action 仍逐个执行。

## 4. 与旧 EEF 路径的区别

保留的旧接口：

```text
delta_ee:
Euler 增量 + self.move() + planner

delta_ee_rotvec:
rotvec 增量 + self.move() + planner
```

当前 OpenPI `delta_eef` 默认接口：

```text
delta_ee_rotvec_ik:
rotvec 增量 + Differential IK + 直接 joint position 写入
```

不要把 `eef_action_type` 改成 `delta_ee_rotvec`，除非明确要恢复“目标 EEF pose -> cuRobo 规划 -> 长轨迹执行”的旧行为。

## 5. 运行命令

vision：

```bash
cd 
conda activate OpenViTac
OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_USB demo openpi/eef_delta/deploy_eef_delta_vision \
  --start_seed 0 --max_seed 0 --total_num 1
```

vision + tactile：

```bash
cd 
conda activate OpenViTac
OMNI_KIT_ACCEPT_EULA=yes python scripts/eval_policy.py \
  insert_USB demo openpi/eef_delta/deploy_eef_delta_vision_tactile \
  --start_seed 0 --max_seed 0 --total_num 1
```
