import yaml
import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.utils import configclass

from ..utils.transforms import *
from ..utils.atom import GRASP_DIRECTION_DIC
from .robot_cfg import RobotCfg
from .curobo_planner import CuroboPlanner, CuroboPlannerCfg
from .._global import *

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from curobo.wrap.reacher.motion_gen import MotionGenResult
    from .._base_task import BaseTask

class RobotManager:
    def __init__(self, robot_cfg:RobotCfg, task:'BaseTask', planner_time_dilation_factor:float=1.0):
        self.cfg = robot_cfg
        self.task = task
        self.device = task.device
        self.sensor_type = task.cfg.tactile_sensor_type
        if self.sensor_type == 'xensews':
            self.robot_type = 'franka_robotiq'
        elif self.sensor_type in ['gsmini', 'neote']:
            self.robot_type = 'franka_panda'
        else:
            raise NotImplementedError(f"Sensor type {self.sensor_type} not implemented.")

        self.robot = Articulation(self.cfg.robot)
        self.task.scene.articulations['robot'] = self.robot
        self.planner_time_dilation_factor = planner_time_dilation_factor

        self.gripper_max_qpos = 0.039
        self.last_arm_velocity = None
        self.last_gripper_velocity = None
        self._ik_controller = None

        if self.robot_type == 'franka_panda':
            self.hand_name = 'panda_hand'
            self._arm_joint_names = [
                'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
                'panda_joint5', 'panda_joint6', 'panda_joint7'
            ]
            self._gripper_joint_names = [
                'panda_finger_joint1', 'panda_finger_joint2'
            ]
            self.gripper_max_qpos = self.cfg.gripper_max_qpos
            self.gripper_open_qpos = self.cfg.gripper_max_qpos
            self.gripper_close_qpos = 0.0
            self.gripper_plan_step = 0.0005
            self.gripper_velocity_limit = 0.0001
            self.yaml_path = str(EMBODIMENTS_ROOT / 'franka' / 'curobo.yml')
            offset = self.cfg.gripper_offset
        elif self.robot_type == 'franka_robotiq':
            self.hand_name = 'panda_link8'
            self._arm_joint_names = [
                'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
                'panda_joint5', 'panda_joint6', 'panda_joint7'
            ]
            # Official Robotiq 2F-85 USD uses finger_joint as the single active
            # master joint; the right side and closed-chain joints are handled by USD.
            self._gripper_joint_names = ['finger_joint']
            self._gripper_mimic_multipliers = torch.tensor([1.0], device=self.device)
            self.gripper_max_qpos = self.cfg.gripper_max_qpos
            explicit_open = getattr(self.cfg, "gripper_open_qpos", None)
            explicit_close = getattr(self.cfg, "gripper_close_qpos", None)
            if explicit_open is not None and explicit_close is not None:
                self.gripper_open_qpos = float(explicit_open)
                self.gripper_close_qpos = float(explicit_close)
            else:
                self.gripper_open_qpos = 0.0
                self.gripper_close_qpos = self.cfg.gripper_max_qpos
            self.gripper_plan_step = 0.02
            self.gripper_velocity_limit = 1.0
            self.yaml_path = str(EMBODIMENTS_ROOT / 'franka' / 'curobo_panda_link8.yml')
            offset = self.cfg.gripper_offset
        else:
            raise NotImplementedError(f"Robot type {self.robot_type} not implemented.")
 
        # offset from end-effector to gripper center frame
        self._offset = Pose(p=[0, 0, -offset], q=[1, 0, 0, 0])
        self._offset_pos = torch.tensor([0.0, 0.0, offset], device=self.device).repeat(self.task.num_envs, 1)
        self._offset_rot = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.task.num_envs, 1)

    def setup(self):
        """设置机器人属性"""
        body_ids, body_names = self.robot.find_bodies(self.hand_name)
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]
        self._jacobi_body_idx = self._body_idx - 1

        joint_names = self.robot.joint_names
        self.joint_name_to_id = {name: i for i, name in enumerate(joint_names)}

        self._arm_ids = torch.tensor([
            self.joint_name_to_id[n] for n in self._arm_joint_names
        ], device=self.device)
        self._gripper_ids = torch.tensor([
            self.joint_name_to_id[n] for n in self._gripper_joint_names
        ], device=self.device)
        if self.robot_type != 'franka_robotiq':
            self._gripper_mimic_multipliers = torch.ones(len(self._gripper_ids), device=self.device)
        self.origin_pose = self.get_gripper_center_pose()
        self._all_ids = torch.cat([self._arm_ids, self._gripper_ids], dim=0)
 
        self.root_pose = Pose.from_list(self.robot.data.root_link_pos_w[0])
        planner_cfg = CuroboPlannerCfg(
            dt=self.task.cfg.sim.dt,
            all_joints_name=self.robot.joint_names,
            active_joints_name=self._arm_joint_names,
            robot_prime_path=self.cfg.robot.prim_path,
            yaml_path=self.yaml_path
        )
        self.planner = CuroboPlanner(
            task=self.task,
            cfg=planner_cfg,
            robot_origin_pose=self.root_pose,
        )
        self._setup_ik_controller()

    def _setup_ik_controller(self):
        """Initialize Differential IK controller for single-step EEF servo."""

        ik_controller_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="svd",
        )
        self._ik_controller = DifferentialIKController(
            cfg=ik_controller_cfg,
            num_envs=self.task.num_envs,
            device=self.device,
        )

    @property
    def jacobian_w(self) -> torch.Tensor:
        """Geometric Jacobian of the hand body in world frame."""

        return self.robot.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, :]

    @property
    def jacobian_b(self) -> torch.Tensor:
        """Geometric Jacobian of the hand body in robot base frame."""

        jacobian = self.jacobian_w.clone()
        base_rot = self.robot.data.root_link_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        return jacobian

    def get_ee_pose_tensor(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return current hand frame pose in robot base frame."""

        ee_pos_w = self.robot.data.body_link_pos_w[:, self._body_idx]
        ee_quat_w = self.robot.data.body_link_quat_w[:, self._body_idx]
        root_pos_w = self.robot.data.root_link_pos_w
        root_quat_w = self.robot.data.root_link_quat_w
        return math_utils.subtract_frame_transforms(
            root_pos_w,
            root_quat_w,
            ee_pos_w,
            ee_quat_w,
        )

    def servo_delta_ee_rotvec(self, action: torch.Tensor, force: bool = True):
        """Execute one OpenPI EEF delta action through Differential IK.

        Action semantics: [delta_xyz(3), delta_rotvec(3), delta_gripper_qpos(1)].
        The rotvec is base-frame and satisfies R_target = exp(rotvec) * R_current.
        """

        if self._ik_controller is None:
            self._setup_ik_controller()

        action = action.to(device=self.device, dtype=torch.float32).reshape(1, -1)
        if action.shape[-1] != 7:
            raise ValueError(f"delta_ee_rotvec_ik action must be 7D, got shape={tuple(action.shape)}")

        ee_pos_b, ee_quat_b = self.get_ee_pose_tensor()
        delta_rotvec = action[:, 3:6]
        delta_angle = torch.linalg.vector_norm(delta_rotvec, dim=1, keepdim=True)
        delta_axis = delta_rotvec / torch.clamp(delta_angle, min=1.0e-6)
        delta_quat = math_utils.quat_from_angle_axis(delta_angle.squeeze(-1), delta_axis)
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.task.num_envs, 1)
        delta_quat = torch.where(delta_angle > 1.0e-6, delta_quat, identity_quat)

        target_pos_b = ee_pos_b + action[:, :3]
        target_quat_b = math_utils.quat_mul(delta_quat, ee_quat_b)
        ik_command = torch.cat([target_pos_b, target_quat_b], dim=-1)
        self._ik_controller.set_command(ik_command)

        jacobian = self.jacobian_b[:, :, self._arm_ids]
        joint_pos = self.robot.data.joint_pos[:, self._arm_ids]
        joint_pos_des = self._ik_controller.compute(
            ee_pos_b,
            ee_quat_b,
            jacobian,
            joint_pos,
        )

        target_gripper_qpos = self.robot.data.joint_pos[:, self._gripper_ids][0, 0] + action[0, 6]
        target_gripper_qpos = torch.clamp(target_gripper_qpos, 0.0, self.gripper_max_qpos)
        target_gripper = target_gripper_qpos.repeat(len(self._gripper_ids))

        self.set_arm(joint_pos_des[0], force=force)
        self.set_gripper(target_gripper, force=force)
        return True

    def compute_delta_ee_rotvec_qpos_target(
        self,
        action: torch.Tensor,
        ee_pos_b: torch.Tensor | None = None,
        ee_quat_b: torch.Tensor | None = None,
        joint_pos: torch.Tensor | None = None,
        gripper_qpos: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute one delta EEF IK target without writing it to the simulator.

        This mirrors ``servo_delta_ee_rotvec`` but returns the virtual next
        arm qpos, gripper qpos, EEF position, and EEF quaternion.  It is used
        to roll out a predicted delta-EFF chunk into absolute qpos targets
        before temporal ensembling.
        """

        if self._ik_controller is None:
            self._setup_ik_controller()

        action = action.to(device=self.device, dtype=torch.float32).reshape(1, -1)
        if action.shape[-1] != 7:
            raise ValueError(f"delta_ee_rotvec_ik action must be 7D, got shape={tuple(action.shape)}")

        if ee_pos_b is None or ee_quat_b is None:
            ee_pos_b, ee_quat_b = self.get_ee_pose_tensor()
        else:
            ee_pos_b = ee_pos_b.to(device=self.device, dtype=torch.float32).reshape(self.task.num_envs, 3)
            ee_quat_b = ee_quat_b.to(device=self.device, dtype=torch.float32).reshape(self.task.num_envs, 4)

        if joint_pos is None:
            joint_pos = self.robot.data.joint_pos[:, self._arm_ids]
        else:
            joint_pos = joint_pos.to(device=self.device, dtype=torch.float32).reshape(self.task.num_envs, -1)
            if joint_pos.shape[-1] != len(self._arm_ids):
                raise ValueError(f"joint_pos must have {len(self._arm_ids)} arm dims, got shape={tuple(joint_pos.shape)}")

        delta_rotvec = action[:, 3:6]
        delta_angle = torch.linalg.vector_norm(delta_rotvec, dim=1, keepdim=True)
        delta_axis = delta_rotvec / torch.clamp(delta_angle, min=1.0e-6)
        delta_quat = math_utils.quat_from_angle_axis(delta_angle.squeeze(-1), delta_axis)
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.task.num_envs, 1)
        delta_quat = torch.where(delta_angle > 1.0e-6, delta_quat, identity_quat)

        target_pos_b = ee_pos_b + action[:, :3]
        target_quat_b = math_utils.quat_mul(delta_quat, ee_quat_b)
        self._ik_controller.set_command(torch.cat([target_pos_b, target_quat_b], dim=-1))

        jacobian = self.jacobian_b[:, :, self._arm_ids]
        joint_pos_des = self._ik_controller.compute(
            ee_pos_b,
            ee_quat_b,
            jacobian,
            joint_pos,
        )

        if gripper_qpos is None:
            current_gripper = self.robot.data.joint_pos[:, self._gripper_ids][0, 0]
        else:
            current_gripper = torch.as_tensor(gripper_qpos, dtype=torch.float32, device=self.device).reshape(-1)[0]
        target_gripper_qpos = torch.clamp(current_gripper + action[0, 6], 0.0, self.gripper_max_qpos)
        return joint_pos_des, target_gripper_qpos.reshape(1), target_pos_b, target_quat_b
    
    def ee_to_gripper_center(self, ee_pose:Pose) -> Pose:
        """将夹爪中心位姿转换为末端执行器目标位姿"""
        return ee_pose.add_offset(self._offset.inv())

    def gripper_center_to_ee(self, gripper_center_pose:Pose) -> Pose:
        """将夹爪中心位姿转换为末端执行器目标位姿"""
        return gripper_center_pose.add_offset(self._offset)
    
    def get_gripper_center_pose(self, env_ids:slice=None) -> Pose:
        """获取当前夹爪中心位姿"""
        return self.ee_to_gripper_center(self.get_ee_pose())
    
    def get_inhand_pose(self, actor:'Actor') -> Pose:
        return actor.get_pose().rebase(self.get_gripper_center_pose())
    
    def get_ee_pose(self, env_ids:slice=None) -> Pose:
        """获取当前末端执行器目标位姿（target_pose）"""
        if env_ids is None:
            env_ids = [0]
        ee_pos_w = self.robot.data.body_link_pos_w[:, self._body_idx]
        ee_quat_w = self.robot.data.body_link_quat_w[:, self._body_idx]
        root_pos_w = self.robot.data.root_link_pos_w
        root_quat_w = self.robot.data.root_link_quat_w
        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        return Pose(ee_pose_b[0].cpu().numpy(), ee_quat_b[0].cpu().numpy())

    def get_qpos(self):
        return self.robot.data.joint_pos.clone().cpu()
    
    def get_gripper_qpos(self):
        return self.get_qpos()[0, self._gripper_ids[0]].clone().cpu().item()

    def get_gripper_target_qpos(self):
        target = self.robot.data.joint_pos_target[0, self._gripper_ids[0]]
        return target.detach().cpu().item()
    
    def get_gripper_percentage(self):
        qpos = self.get_gripper_qpos()
        denom = self.gripper_open_qpos - self.gripper_close_qpos
        if abs(denom) < 1e-8:
            return 0.0
        return (qpos - self.gripper_close_qpos) / denom

    def is_gripper_opening(self, target_qpos: float):
        current_qpos = self.get_gripper_qpos()
        return abs(target_qpos - self.gripper_open_qpos) < abs(current_qpos - self.gripper_open_qpos)

    def set_arm(self, pos:torch.Tensor, vel:torch.Tensor=None, env_ids:slice=None, force:bool=True):
        '''设置目标位姿'''
        if force and getattr(self.task.cfg, "absolute_joint_zero_velocity_targets", False):
            vel = torch.zeros_like(pos)
        self.robot.set_joint_position_target(pos, joint_ids=self._arm_ids, env_ids=env_ids)
        if vel is not None:
            self.robot.set_joint_velocity_target(vel, joint_ids=self._arm_ids, env_ids=env_ids)
        if force:
            if self.robot_type == 'franka_robotiq':
                # Track Curobo exactly for the arm while leaving the Robotiq
                # finger/mimic joints under physical PD control.
                arm_pos = pos.unsqueeze(0) if pos.ndim == 1 else pos
                if vel is None:
                    arm_vel = torch.zeros_like(arm_pos)
                else:
                    arm_vel = vel.unsqueeze(0) if vel.ndim == 1 else vel
                self.robot.write_joint_state_to_sim(
                    arm_pos,
                    arm_vel,
                    joint_ids=self._arm_ids,
                    env_ids=env_ids,
                )
            else:
                self.robot.root_physx_view.set_dof_positions(
                    self.robot._data.joint_pos_target,
                    self.robot._ALL_INDICES
                )

    def _map_gripper_command(self, value: torch.Tensor | float | int):
        value = torch.as_tensor(value, dtype=torch.float32, device=self.device).flatten()
        if value.numel() == 1:
            value = value * self._gripper_mimic_multipliers
        elif self.robot_type == 'franka_robotiq':
            # Upper layers pass the configured Robotiq master qpos; USD handles mimic joints.
            value = value[:1] * self._gripper_mimic_multipliers
        return value

    def set_gripper(self, pos:torch.Tensor, vel:torch.Tensor=None, env_ids:slice=None, force:bool=True):
        '''Set gripper target pose.'''
        pos = self._map_gripper_command(pos)
        if force and getattr(self.task.cfg, "absolute_joint_zero_velocity_targets", False):
            vel = torch.zeros_like(pos)
        self.robot.set_joint_position_target(pos, joint_ids=self._gripper_ids, env_ids=env_ids)
        if vel is not None:
            vel = self._map_gripper_command(vel)
            self.robot.set_joint_velocity_target(vel, joint_ids=self._gripper_ids, env_ids=env_ids)
        if force and self.robot_type != 'franka_robotiq':
            self.robot.root_physx_view.set_dof_positions(
                self.robot._data.joint_pos_target,
                self.robot._ALL_INDICES
            )

    def plan_arm(self, target_pose:Pose, constraint_pose=None, pre_dis=None, time_dilation_factor=None):
        result:MotionGenResult = self.planner.plan_path(
            curr_joint_pos=self.robot.data.joint_pos[0],
            curr_joint_vel=self.robot.data.joint_vel[0],
            target_ee_pose=target_pose,
            real_robot_pose=self.root_pose,
            pre_dis=pre_dis,
            constraint_pose=constraint_pose,
            time_dilation_factor=time_dilation_factor
        )
        
        if result.success.item():
            return {
                'status': 'Success',
                'num_steps': result.interpolated_plan.position.shape[0],
                'position': result.interpolated_plan.position.detach(),
                'velocity': result.interpolated_plan.velocity.detach()
            }
        else:
            return {'status': 'Fail', 'num_steps': 0, 'position': None, 'velocity': None}

    def gripper_percent2qpos(self, percentage:float):
        percentage = float(np.clip(percentage, 0.0, 1.0))
        return self.gripper_close_qpos + (
            self.gripper_open_qpos - self.gripper_close_qpos
        ) * percentage

    def plan_gripper(self, pos:float, type:Literal['percent', 'qpos'] = 'percent'):
        if type == 'percent':
            target_pos = self.gripper_percent2qpos(pos)
        else:
            target_pos = pos
        gripper_pos = self.robot.data.joint_pos[0, self._gripper_ids][0]
        num_steps = np.ceil(abs(target_pos - gripper_pos.cpu().item()) / self.gripper_plan_step).astype(int)
        position = torch.linspace(gripper_pos, target_pos, num_steps, device=self.device)
        velocity = torch.clip((position - gripper_pos)/self.task.cfg.sim.dt, -self.gripper_velocity_limit, self.gripper_velocity_limit)

        return {
            'status': 'Success',
            'num_steps': num_steps,
            'position': position.detach(),
            'velocity': velocity.detach()
        }

    def _reset_idx(self, env_ids: torch.Tensor | None=None):
        """重置环境"""
        if not hasattr(self, 'origin_pose'):
            self._setup_robot_properties()
        joint_pos = self.robot.data.default_joint_pos.clone()
        joint_vel = torch.zeros_like(joint_pos)
        
        self.planner.reset()
        if self._ik_controller is not None:
            self._ik_controller.reset()
        self.robot.set_joint_position_target(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self.robot._physics_sim_view.update_articulations_kinematic()
    
    def get_observations(self, data_type:list[str]=['joint', 'ee']) -> dict:
        obs = {}
        if 'ee' in data_type:
            obs['ee'] = self.get_ee_pose().totensor(device=self.device)
        if 'joint' in data_type:
            obs['joint'] = self.robot.data.joint_pos.squeeze(0)
        return obs
    
    def get_grasp_perfect_direction(self):
        return 'top_down'
