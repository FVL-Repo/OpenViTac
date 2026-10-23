"""Whole-robot assembly from a pure :class:`EmbodimentSpec`, plus end-effector IK.

The embodiment owns the USD, arm/gripper joints, nominal pose, and PD groups.  A
sensor binding contributes only mount joint offsets (applied by ``Scene`` before
calling :func:`assemble_robot`); it never selects the robot.  Quaternions are XYZW
(isaaclab 3.0).
"""

from __future__ import annotations

import copy
from typing import cast

import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import ActuatorBaseCfg, ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.utils.math import (
    # torchscript ScriptFunction; the stub leaves its type partially Unknown.
    matrix_from_quat,
    subtract_frame_transforms,
)
from isaaclab_physx.sim.schemas import RigidBodyPropertiesCfg

from openvitac_spec import EmbodimentSpec

from .assets import asset
from .spawn import apply_init_state, spawn_articulation


def _actuator_cfgs(embodiment: EmbodimentSpec) -> dict[str, ActuatorBaseCfg]:
    """PD groups keyed by the embodiment's declared actuator profile."""
    if embodiment.arm.actuator_profile != "franka_high_pd":
        raise NotImplementedError(
            f"embodiment {embodiment.name!r}: unknown actuator profile "
            f"{embodiment.arm.actuator_profile!r}"
        )
    actuators: dict[str, ActuatorBaseCfg] = {
        # HIGH_PD: 400/80 gains and a real panda wrist effort limit (12 N). Gravity
        # is disabled in assemble_robot; scripted Cartesian targets rely on the PD
        # loop and would droop without that.
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            joint_effort_limit=87.0,
            joint_velocity_limit=2.175,
            stiffness=400.0,
            damping=80.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            joint_effort_limit=12.0,
            joint_velocity_limit=2.61,
            stiffness=400.0,
            damping=80.0,
        ),
    }
    if embodiment.gripper.kind == "panda_parallel":
        actuators["panda_fingers"] = ImplicitActuatorCfg(
            joint_names_expr=["panda_finger.*"],
            joint_effort_limit=200.0,
            joint_velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        )
    elif embodiment.gripper.kind == "robotiq_85":
        actuators["robotiq_85"] = ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            joint_effort_limit=80.0,
            joint_velocity_limit=1.0,
            stiffness=200.0,
            damping=20.0,
        )
    else:
        raise NotImplementedError(
            f"embodiment {embodiment.name!r}: unknown gripper "
            f"{embodiment.gripper.kind!r}"
        )
    return actuators


def assemble_robot(
    embodiment: EmbodimentSpec,
    prim_path: str = "/World/Robot",
    joint_pos: dict[str, float] | None = None,
) -> RobotArm:
    """Build the whole robot for ``embodiment``.

    ``joint_pos`` is the task's nominal pose (already carrying sensor mount offsets);
    the embodiment default is used when omitted.
    """
    embodiment.validate_supported()
    joints = dict(
        embodiment.default_joint_pos if joint_pos is None else joint_pos
    )
    cfg = ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=asset(embodiment.asset.usd),
            rigid_props=RigidBodyPropertiesCfg(disable_gravity=True),
            # Finger-pad contact reporting for ArmHandle.gripper_contact.
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos=joints),
        actuators=copy.deepcopy(_actuator_cfgs(embodiment)),
    )
    return RobotArm(spawn_articulation(cfg), embodiment)


class RobotArm:
    """An assembled articulation plus a differential-IK solver for its end effector.

    Body/joint indices and the IK solver resolve lazily on first use: the
    articulation's ``.data`` only exists after the sim is reset, so they cannot be
    read at spawn time.  ``_last_target`` mirrors the actuator position targets in
    the articulation's joint order so observations publish the command in effect.
    """

    def __init__(self, robot: Articulation, embodiment: EmbodimentSpec) -> None:
        self.robot = robot
        self.embodiment = embodiment
        self._ik: DifferentialIKController | None = None
        self._ee_body_idx = -1
        self._ee_jacobi_idx = -1
        self._arm_joint_ids: list[int] = []
        self._gripper_joint_ids: list[int] = []
        self._last_target: torch.Tensor | None = None

    def _resolve(self) -> None:
        if self._ik is not None:
            return
        ee_link = self.embodiment.arm.ee_link
        ee_ids = self.robot.find_bodies(ee_link)[0]
        if len(ee_ids) != 1:
            raise ValueError(f"expected one EE body {ee_link!r}, got {ee_ids}")
        self._ee_body_idx = ee_ids[0]
        # Fixed-base arm: the root body owns no Jacobian row, so body i maps to row i-1.
        self._ee_jacobi_idx = ee_ids[0] - 1
        arm_joints = list(self.embodiment.arm.arm_joints)
        # find_joints returns plain index lists here (no env_ids given); the stub's
        # ProxyArray union is the batched-query shape.
        self._arm_joint_ids = cast(list[int], self.robot.find_joints(arm_joints)[0])
        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls"
        )
        self._ik = DifferentialIKController(
            ik_cfg, num_envs=self.robot.num_instances, device=self.robot.device
        )

    def ee_pose_base(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Current EE pose ``(pos, quat_xyzw)`` in the robot base frame."""
        self._resolve()
        ee_pose_w = self.robot.data.body_pose_w.torch[:, self._ee_body_idx]
        root_pose_w = self.robot.data.root_pose_w.torch
        return subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3],
            ee_pose_w[:, 3:7],
        )

    def _jacobian_base(self) -> torch.Tensor:
        jacobian = self.robot.data.body_link_jacobian_w.torch[
            :, self._ee_jacobi_idx, :, self._arm_joint_ids
        ].clone()
        root_quat = self.robot.data.root_pose_w.torch[:, 3:7]
        root_rot = cast(torch.Tensor, matrix_from_quat(root_quat))
        jacobian[:, :3, :] = torch.bmm(root_rot, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(root_rot, jacobian[:, 3:, :])
        assert isinstance(jacobian, torch.Tensor)
        return jacobian

    def ik_to(
        self, target_pos_b: torch.Tensor, target_quat_b: torch.Tensor
    ) -> torch.Tensor:
        """Joint positions moving the EE to (pos, quat) target (base frame, XYZW)."""
        self._resolve()
        assert self._ik is not None
        self._ik.set_command(torch.cat([target_pos_b, target_quat_b], dim=-1))
        ee_pos_b, ee_quat_b = self.ee_pose_base()
        joint_pos = self.robot.data.joint_pos.torch[:, self._arm_joint_ids]
        return self._ik.compute(ee_pos_b, ee_quat_b, self._jacobian_base(), joint_pos)

    def reset_to_init(self) -> None:
        """Drive the arm to its ``init_state`` pose and hold it (after reset)."""
        apply_init_state(self.robot)
        # apply_init_state commands every joint (arm and gripper) to the init pose;
        # that constant target is the episode's action until a new command arrives.
        self._last_target = self.robot.data.default_joint_pos.torch.clone()

    def set_arm_target(self, joint_pos_des: torch.Tensor) -> None:
        """Command the arm joints toward ``joint_pos_des`` (e.g. :meth:`ik_to`)."""
        self._resolve()
        # The actuator target_command stub leaves the member partially Unknown.
        target_command = self.robot.actuators.target_command
        target_command.set_position_index(
            value=joint_pos_des, joint_ids=self._arm_joint_ids
        )
        assert self._last_target is not None  # reset_to_init() runs after sim.reset()
        self._last_target[:, self._arm_joint_ids] = joint_pos_des

    def _resolve_gripper_joints(self) -> list[int]:
        self._resolve()
        if not self._gripper_joint_ids:
            self._gripper_joint_ids = cast(
                list[int],
                self.robot.find_joints(list(self.embodiment.gripper.joints))[0],
            )
        return self._gripper_joint_ids

    def set_gripper_target(self, qpos: float) -> None:
        """Command every driven gripper joint to ``qpos`` for the next step."""
        joint_ids = self._resolve_gripper_joints()
        value = float(qpos)
        target = torch.full(
            (self.robot.num_instances, len(joint_ids)),
            value,
            device=self.robot.device,
        )
        # The actuator target_command stub leaves the member partially Unknown.
        target_command = self.robot.actuators.target_command
        target_command.set_position_index(
            value=target, joint_ids=joint_ids
        )
        assert self._last_target is not None  # reset_to_init() runs after sim.reset()
        self._last_target[:, joint_ids] = value

    def gripper_joint_pos(self) -> float:
        """Current driven gripper position, averaged over mirrored jaws."""
        joint_ids = self._resolve_gripper_joints()
        values = self.robot.data.joint_pos.torch[0, joint_ids]
        return float(values.mean().item())

    @property
    def last_target(self) -> torch.Tensor:
        """Full joint-position target in effect, in ``joint_pos`` column order."""
        if self._last_target is None:
            raise RuntimeError("RobotArm.last_target read before reset_to_init()")
        return self._last_target
