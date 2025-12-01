# *******************************************************************************
# *                                                                             *
# *  Private and Confidential                                                   *
# *                                                                             *
# *  Unauthorized copying of this file, via any medium is strictly prohibited.  *
# *  Proprietary and confidential.                                              *
# *                                                                             *
# *  © 2024 DiffLab. All rights reserved.                                       *
# *                                                                             *
# *  Author: Yu Feng                                                            *
# *  Data: 2025/03/06                                                           *
# *  Contact: yu-feng@sjtu.edu.cn                                               *
# *  Description: Controllers for quadrotor control                             *
# *******************************************************************************
from __future__ import annotations
import torch
import torch.nn.functional as F
from .thrust_controller_diff import ThrustController
import isaaclab.utils.math as math_utils
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
if TYPE_CHECKING:
    from .controller_diff_cfg import CTBRControllerCfg, LVControllerCfg, PSControllerCfg

class ControllerBase(ABC):
    def __init__(self) -> None:
        pass
    
    @abstractmethod
    def update_state(self, states):
        raise NotImplementedError
    
    @abstractmethod
    def compute(self, now_state, cmd):
        raise NotImplementedError
    
class CTBRController(ControllerBase):
    """
    Combine
    Thrust and Body Rate Loop Controller. \n
    Given desired total thrust and body rate, compute the real thrust and tau for executing with `ThrustController` and `P-controller`. \n
    Mathmatics Description:
        `tau_des = J * beta + w x Jw = J * Kp * error_angle + w x Jw = J * Kp * (angle_rate_des - angle_rate) + w x Jw`
         `[f, tau_x, tau_y, tau_z] = M4 * [f1, f2, f3, f4]`
    reference: Quanquan Lecture 6: 动态模型和参数测量 Page 17.   
    """
    def __init__(self, cfg: CTBRControllerCfg, num_envs: int, device: str, mass: float, inertia: torch.Tensor, dt: float):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        self.mass = mass
        self.inertia = inertia
        self.inertia_inv = torch.inverse(inertia)

        self.t_BM_ = (
            cfg.arm_length
            * torch.tensor(0.5).sqrt()
            * torch.tensor([
                [1, -1, -1, 1],
                [-1, -1, 1, 1],
            ])
        )
        self._B_allocation = torch.vstack(
            [torch.ones(1, 4), self.t_BM_, cfg.kappa * torch.tensor([1, -1, 1, -1])]
        ).to(self.device)
        self._B_allocation_inv = torch.inverse(self._B_allocation)
        self._B_allocation_T = self._B_allocation.transpose(0, 1)
        self._B_allocation_inv_T = self._B_allocation_inv.transpose(0, 1)

        self.pos = torch.zeros([self.num_envs,3],device=self.device)
        self.quat = torch.zeros([self.num_envs,4],device=self.device)
        self.lin_vel_w = torch.zeros([self.num_envs,3],device=self.device)
        self.ang_vel_w = torch.zeros([self.num_envs,3],device=self.device)
        self.lin_vel_b = torch.zeros([self.num_envs,3],device=self.device)
        self.ang_vel_b = torch.zeros([self.num_envs,3],device=self.device)
        self.lin_acc_w = torch.zeros([self.num_envs,3],device=self.device)
        self.ang_acc_w = torch.zeros([self.num_envs,3],device=self.device)
        self.lin_acc_b = torch.zeros([self.num_envs,3],device=self.device)
        self.ang_acc_b = torch.zeros([self.num_envs,3],device=self.device)

        self.thrust_ctrl = ThrustController(device=self.device, num_envs=self.num_envs, params={
            "motor_tau": cfg.motor_tau,
            "dt": self.dt,
            "thrustmap": cfg.thrustmap,
            "arm_length": cfg.arm_length,
            "kappa": cfg.kappa
        })

        self.rate_gain_p = torch.tensor([[ float(cfg.rate_gain_p[0]), float(cfg.rate_gain_p[1]), float(cfg.rate_gain_p[2]) ]], device=self.device).repeat(self.num_envs, 1)
        self.rate_gain_i = torch.tensor([[ float(cfg.rate_gain_i[0]), float(cfg.rate_gain_i[1]), float(cfg.rate_gain_i[2]) ]], device=self.device).repeat(self.num_envs, 1)
        self.rate_gain_d = torch.tensor([[ float(cfg.rate_gain_d[0]), float(cfg.rate_gain_d[1]), float(cfg.rate_gain_d[2]) ]], device=self.device).repeat(self.num_envs, 1)

        self.motor_omega = cfg.motor_omega
        self.thrust_map = cfg.thrustmap
        self.thrust_max = self.thrust_map[0] * self.motor_omega[1] ** 2 + self.thrust_map[1] * self.motor_omega[1] + self.thrust_map[2]
        self.thrust_min = self.thrust_map[0] * self.motor_omega[0] ** 2 + self.thrust_map[1] * self.motor_omega[0] + self.thrust_map[2]

        self.gross_thrust_bound = [self.thrust_min * 4, self.thrust_max * 4]
        self.body_rate_bound = cfg.body_rate_bound

        # thrust delay parameters
        self.gross_thrust = torch.zeros([self.num_envs, 1], device=self.device)
        self.torque =  torch.zeros([self.num_envs, 3], device=self.device)
        self.thrust_ctrl_delay = torch.ones([self.num_envs, 1], device=self.device) * cfg.thrust_ctrl_delay
        self.torque_ctrl_delay = torch.ones([self.num_envs, 3], device=self.device) * torch.tensor(cfg.torque_ctrl_delay, device=self.device)

    def update_state(self, states):
        self.pos = states["pos"]
        self.quat = states["quat"]
        self.lin_vel_w = states["lin_vel_w"]
        self.ang_vel_w = states["ang_vel_w"]
        self.lin_vel_b = states["lin_vel_b"]
        self.ang_vel_b = states["ang_vel_b"]
        self.lin_acc_w = states["lin_acc_w"]
        self.ang_acc_w = states["ang_acc_w"]
        self.lin_acc_b = states["lin_acc_b"]
        self.ang_acc_b = states["ang_acc_b"]

    def compute(self, now_state, cmd):
        # NOTE: diff maybe problematic here
        self.update_state(now_state)
        assert self.pos.requires_grad == False, "now_state should be detached"
        cmd_thrust = cmd[:, :1]
        gross_thrust_des = cmd_thrust.clamp(float(self.gross_thrust_bound[0]),
                                                  float(self.gross_thrust_bound[1]))
        # apply thrust delay
        self.gross_thrust = (1 - torch.exp(-self.dt / self.thrust_ctrl_delay)) * gross_thrust_des + \
                            torch.exp(-self.dt / self.thrust_ctrl_delay) * self.gross_thrust
        cmd_rate = cmd[:, 1:4]
        cmd_rate = cmd_rate.clamp(float(self.body_rate_bound[0]), float(self.body_rate_bound[1]))
        err_rate = cmd_rate - self.ang_vel_b
        torque_des = (self.inertia @ (self.rate_gain_p * err_rate)[..., None]).squeeze(-1) + torch.linalg.cross(self.ang_vel_b, (self.inertia @ self.ang_vel_b[...,None]).squeeze(-1)) - self.rate_gain_d * self.ang_acc_b
        self.torque = (1 - torch.exp(-self.dt / self.torque_ctrl_delay)) * torque_des + \
                        torch.exp(-self.dt / self.torque_ctrl_delay) * self.torque
        thrust_torque_des = torch.cat((self.gross_thrust, self.torque), dim=1)
        if not self.cfg.use_motor_model:
            return None, thrust_torque_des
        # [TODO]: fix bug on thrust_controller
        thrusts_des = thrust_torque_des @ self._B_allocation_inv_T
        thrusts_des.clamp_(0.0, self.thrust_max)
        thrusts_now = self.thrust_ctrl.update(thrusts_des)
        thrust_torque_now = thrusts_now @ self._B_allocation_T
        return thrusts_now, thrust_torque_now
    
    def reset_idx(self, envs_id):
        if len(envs_id) == self.num_envs:
            self.gross_thrust = self.gross_thrust.detach()
            self.gross_thrust.zero_()
            self.torque = self.torque.detach()
            self.torque.zero_()
            return
        clone = self.gross_thrust.clone()
        clone[envs_id] = clone[envs_id].detach()
        clone[envs_id] = torch.zeros_like(clone[envs_id])
        self.gross_thrust = clone
        clone = self.torque.clone()
        clone[envs_id] = clone[envs_id].detach()
        clone[envs_id] = torch.zeros_like(clone[envs_id])
        self.torque = clone
        #
        self.thrust_ctrl.reset_idx(envs_id)
        
    def reset(self,):
        self.reset_idx(torch.arange(self.num_envs, dtype=torch.long, device=self.device))

    def detach(self, ):
        self.thrust_ctrl.detach()
        self.gross_thrust = self.gross_thrust.clone().detach()
        self.torque = self.torque.clone().detach()

class LVController(ControllerBase):
    """
    Linear Velocity and Yaw Loop Controller
    """
    def __init__(self, cfg: LVControllerCfg, num_envs: int, device: str, mass: float, inertia: torch.Tensor, dt: float):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        self.mass = mass
        self.inertia = inertia
        self.inertia_inv = torch.inverse(inertia)

        self.t_BM_ = (
            cfg.arm_length
            * torch.tensor(0.5).sqrt()
            * torch.tensor([
                [1, -1, -1, 1],
                [-1, -1, 1, 1],
            ])
        )
        self._B_allocation = torch.vstack(
            [torch.ones(1, 4), self.t_BM_, cfg.kappa * torch.tensor([1, -1, 1, -1])]
        ).to(self.device)
        self._B_allocation_inv = torch.inverse(self._B_allocation)
        self._B_allocation_inv_T = self._B_allocation_inv.transpose(0, 1)
        self._B_allocation_T = self._B_allocation.transpose(0, 1)

        self.pos = torch.zeros([self.num_envs,3],device=self.device)
        self.quat = torch.zeros([self.num_envs,4],device=self.device)
        self.lin_vel_w = torch.zeros([self.num_envs,3],device=self.device)
        self.ang_vel_w = torch.zeros([self.num_envs,3],device=self.device)
        self.lin_vel_b = torch.zeros([self.num_envs,3],device=self.device)
        self.ang_vel_b = torch.zeros([self.num_envs,3],device=self.device)

        self.thrust_ctrl = ThrustController(device=self.device, num_envs=self.num_envs, params={
            "motor_tau": cfg.motor_tau,
            "dt": self.dt,
            "thrustmap": cfg.thrustmap,
            "arm_length": cfg.arm_length,
            "kappa": cfg.kappa
        })

        self.g = torch.tensor([[0, 0, -float(cfg.g)]], device=self.device)
        self.g_norm = self.g.norm()

        self.speed_gain = torch.tensor([[ float(cfg.speed_gain[0]), float(cfg.speed_gain[1]), float(cfg.speed_gain[2]) ]], device=self.device).repeat(self.num_envs, 1)
        self.pose_gain = torch.tensor([[ float(cfg.pose_gain[0]), float(cfg.pose_gain[1]), float(cfg.pose_gain[2]) ]], device=self.device).repeat(self.num_envs, 1)
        self.rate_gain = torch.tensor([[ float(cfg.rate_gain[0]), float(cfg.rate_gain[1]), float(cfg.rate_gain[2]) ]], device=self.device).repeat(self.num_envs, 1)

        self.motor_omega = cfg.motor_omega
        self.thrust_map = cfg.thrustmap
        self.thrust_max = self.thrust_map[0] * self.motor_omega[1] ** 2 + self.thrust_map[1] * self.motor_omega[1] + self.thrust_map[2]
        self.thrust_min = self.thrust_map[0] * self.motor_omega[0] ** 2 + self.thrust_map[1] * self.motor_omega[0] + self.thrust_map[2]

        self.gross_thrust_bound = [self.thrust_min * 4, self.thrust_max * 4]
        self.body_rate_bound = cfg.body_rate_bound

        # thrust delay parameters
        self.gross_thrust = torch.zeros([self.num_envs, 1], device=self.device)
        self.thrust_ctrl_delay = torch.ones([self.num_envs, 1], device=self.device) * cfg.thrust_ctrl_delay

    def update_state(self, states):
        self.pos = states["pos"]
        self.quat = states["quat"]
        self.lin_vel_w = states["lin_vel_w"]
        self.ang_vel_w = states["ang_vel_w"]
        self.lin_vel_b = states["lin_vel_b"]
        self.ang_vel_b = states["ang_vel_b"]

    def compute(self, now_state, cmd):
        self.update_state(now_state)
        cmd_speed = cmd[:, 1:]
        cmd_yaw = cmd[:, :1]
        err_speed = cmd_speed - self.lin_vel_w
        acc_fb = torch.min(torch.norm(self.speed_gain * err_speed, dim=-1, keepdim=True),
                           torch.tensor(self.cfg.max_feedback_accel, device=self.device)) * F.normalize(err_speed, p=2, dim=1)
        des_F = self.mass * (acc_fb - self.g)
        gross_thrust_des = math_utils.quat_rotate_inverse(self.quat, des_F)[:, 2:]
        R = math_utils.matrix_from_quat(self.quat)
        R_T = R.transpose(1, 2)
        b1_des = torch.cat([
            torch.cos(cmd_yaw),
            torch.sin(cmd_yaw),
            torch.zeros_like(cmd_yaw)
        ], dim=-1)
        b3_des = F.normalize(des_F, p=2, dim=1)
        b2_des = torch.cross(b3_des, b1_des, dim=1)
        b2_des = F.normalize(b2_des, p=2, dim=1)
        R_des = torch.stack([
            b2_des.cross(b3_des, 1),
            b2_des,
            b3_des
        ], dim=-1)
        R_des_T = R_des.transpose(1, 2)
        m = 0.5 * (torch.bmm(R_des_T, R) - torch.bmm(R_T, R_des))
        pose_err = -torch.stack((-m[:, 1, 2], m[:, 0, 2], -m[:, 0, 1]), dim=1)

        bodyrate_des = self.pose_gain * pose_err

        gross_thrust_des = gross_thrust_des.clamp(float(self.gross_thrust_bound[0]),
                                                  float(self.gross_thrust_bound[1]))
        # apply thrust delay
        self.gross_thrust = (1 - torch.exp(-self.dt / self.thrust_ctrl_delay)) * gross_thrust_des + \
                            torch.exp(-self.dt / self.thrust_ctrl_delay) * self.gross_thrust
        bodyrate_des = bodyrate_des.clamp(float(self.body_rate_bound[0]), float(self.body_rate_bound[1]))
        err_rate = bodyrate_des - self.ang_vel_b
        torque_des = (self.inertia @ (self.rate_gain * err_rate)[..., None]).squeeze(-1) + torch.linalg.cross(self.ang_vel_b, (self.inertia @ self.ang_vel_b[...,None]).squeeze(-1))


        thrust_torque_des = torch.cat((self.gross_thrust, torque_des), dim=1)
        if not self.cfg.use_motor_model:
            return None, thrust_torque_des  #return `bodyrate_des`, `R_des` for PID tuning.
        thrusts_des = thrust_torque_des @ self._B_allocation_inv_T
        thrusts_des.clamp_(0.0, self.thrust_max)
        thrusts_now = self.thrust_ctrl.update(thrusts_des)
        thrust_torque_now = thrusts_now @ self._B_allocation_T
        return thrusts_now, thrust_torque_now
    
    def reset_idx(self, envs_id):
        if len(envs_id) == self.num_envs:
            self.gross_thrust = self.gross_thrust.detach()
            self.gross_thrust.zero_()
            return
        clone = self.gross_thrust.clone()
        clone[envs_id] = clone[envs_id].detach()
        clone[envs_id] = torch.zeros_like(clone[envs_id])
        self.gross_thrust = clone
        
    def reset(self,):
        self.reset_idx(torch.arange(self.num_envs, dtype=torch.long, device=self.device))

    def detach(self, ):
        self.gross_thrust = self.gross_thrust.clone().detach()

class PSController(ControllerBase):
    """
    Position and Yaw Loop Controller
    """
    def __init__(self, cfg: PSControllerCfg, num_envs: int, device: str, mass: float, inertia: torch.Tensor, dt: float):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        self.mass = mass
        self.inertia = inertia
        self.inertia_inv = torch.inverse(inertia)

        self.t_BM_ = (
            cfg.arm_length
            * torch.tensor(0.5).sqrt()
            * torch.tensor([
                [1, -1, -1, 1],
                [-1, -1, 1, 1],
            ])
        )
        self._B_allocation = torch.vstack(
            [torch.ones(1, 4), self.t_BM_, cfg.kappa * torch.tensor([1, -1, 1, -1])]
        ).to(self.device)
        self._B_allocation_inv = torch.inverse(self._B_allocation)
        self._B_allocation_inv_T = self._B_allocation_inv.transpose(0, 1)
        self._B_allocation_T = self._B_allocation.transpose(0, 1)

        self.pos = torch.zeros([self.num_envs,3],device=self.device)
        self.quat = torch.zeros([self.num_envs,4],device=self.device)
        self.lin_vel_w = torch.zeros([self.num_envs,3],device=self.device)
        self.ang_vel_w = torch.zeros([self.num_envs,3],device=self.device)
        self.lin_vel_b = torch.zeros([self.num_envs,3],device=self.device)
        self.ang_vel_b = torch.zeros([self.num_envs,3],device=self.device)

        self.thrust_ctrl = ThrustController(device=self.device, num_envs=self.num_envs, params={
            "motor_tau": cfg.motor_tau,
            "dt": self.dt,
            "thrustmap": cfg.thrustmap,
            "arm_length": cfg.arm_length,
            "kappa": cfg.kappa
        })

        self.g = torch.tensor([[0, 0, -float(cfg.g)]], device=self.device)
        self.g_norm = self.g.norm()
        self.speed_gain = torch.tensor([[ float(cfg.speed_gain[0]),float(cfg.speed_gain[1]), float(cfg.speed_gain[2]) ]], device=self.device).repeat(self.num_envs, 1)
        self.pose_gain = torch.tensor([[ float(cfg.pose_gain[0]), float(cfg.pose_gain[1]), float(cfg.pose_gain[2]) ]], device=self.device).repeat(self.num_envs, 1)
        self.rate_gain = torch.tensor([[ float(cfg.rate_gain[0]), float(cfg.rate_gain[1]), float(cfg.rate_gain[2]) ]], device=self.device).repeat(self.num_envs, 1)
        self.pos_gain = torch.tensor([[ float(cfg.pos_gain[0]), float(cfg.pos_gain[1]), float(cfg.pos_gain[2]) ]], device=self.device).repeat(self.num_envs, 1)

        self.motor_omega = cfg.motor_omega
        self.thrust_map = cfg.thrustmap
        self.thrust_max = self.thrust_map[0] * self.motor_omega[1] ** 2 + self.thrust_map[1] * self.motor_omega[1] + self.thrust_map[2]
        self.thrust_min = self.thrust_map[0] * self.motor_omega[0] ** 2 + self.thrust_map[1] * self.motor_omega[0] + self.thrust_map[2]

        self.gross_thrust_bound = [self.thrust_min * 4, self.thrust_max * 4]
        self.body_rate_bound = cfg.body_rate_bound
        # thrust delay parameters
        self.gross_thrust = torch.zeros([self.num_envs, 1], device=self.device)
        self.thrust_ctrl_delay = torch.ones([self.num_envs, 1], device=self.device) * cfg.thrust_ctrl_delay

    def update_state(self, states):
        self.pos = states["pos"]
        self.quat = states["quat"]
        self.lin_vel_w = states["lin_vel_w"]
        self.ang_vel_w = states["ang_vel_w"]
        self.lin_vel_b = states["lin_vel_b"]
        self.ang_vel_b = states["ang_vel_b"]

    def compute(self, now_state, cmd):
        self.update_state(now_state)
        cmd_pos = cmd[:, 1:]
        cmd_yaw = cmd[:, :1]

        err_pos = cmd_pos - self.pos
        vel_des = self.pos_gain * err_pos

        err_speed = vel_des - self.lin_vel_w
        acc_fb = torch.min(torch.norm(self.speed_gain * err_speed, dim=-1, keepdim=True),
                           torch.tensor(self.cfg.max_feedback_accel, device=self.device)) * F.normalize(err_speed, p=2, dim=1)
        des_F = self.mass * (acc_fb - self.g)
        gross_thrust_des = math_utils.quat_rotate_inverse(self.quat, des_F)[:, 2:]
        R = math_utils.matrix_from_quat(self.quat)
        R_T = R.transpose(1, 2)
        b1_des = torch.cat([
            torch.cos(cmd_yaw),
            torch.sin(cmd_yaw),
            torch.zeros_like(cmd_yaw)
        ], dim=-1)
        b3_des = F.normalize(des_F, p=2, dim=1)
        b2_des = torch.cross(b3_des, b1_des, dim=1)
        b2_des = F.normalize(b2_des, p=2, dim=1)
        R_des = torch.stack([
            b2_des.cross(b3_des, 1),
            b2_des,
            b3_des
        ], dim=-1)
        R_des_T = R_des.transpose(1, 2)
        m = 0.5 * (torch.bmm(R_des_T, R) - torch.bmm(R_T, R_des))
        pose_err = -torch.stack((-m[:, 1, 2], m[:, 0, 2], -m[:, 0, 1]), dim=1)

        bodyrate_des =self.pose_gain * pose_err

        gross_thrust_des = gross_thrust_des.clamp(float(self.gross_thrust_bound[0]),
                                                  float(self.gross_thrust_bound[1]))
        # apply thrust delay
        self.gross_thrust = (1 - torch.exp(-self.dt / self.thrust_ctrl_delay)) * gross_thrust_des + \
                            torch.exp(-self.dt / self.thrust_ctrl_delay) * self.gross_thrust
        bodyrate_des = bodyrate_des.clamp(float(self.body_rate_bound[0]), float(self.body_rate_bound[1]))
        err_rate = bodyrate_des - self.ang_vel_b
        torque_des = (self.inertia @ (self.rate_gain * err_rate)[..., None]).squeeze(-1) + torch.linalg.cross(self.ang_vel_b, (self.inertia @ self.ang_vel_b[...,None]).squeeze(-1))


        thrust_torque_des = torch.cat((self.gross_thrust, torque_des), dim=1)
        if not self.cfg.use_motor_model:
            return None, thrust_torque_des # return `bodyrate_des`, `R_des` for PID tuning.
        # [TODO]: fix bug on thrust_controller
        thrusts_des = thrust_torque_des @ self._B_allocation_inv_T
        thrusts_des.clamp_(0.0, self.thrust_max)
        thrusts_now = self.thrust_ctrl.update(thrusts_des)
        thrust_torque_now = thrusts_now @ self._B_allocation_T
        return thrusts_now, thrust_torque_now

    def reset_idx(self, envs_id):
        if len(envs_id) == self.num_envs:
            self.gross_thrust = self.gross_thrust.detach()
            self.gross_thrust.zero_()
            return
        clone = self.gross_thrust.clone()
        clone[envs_id] = clone[envs_id].detach()
        clone[envs_id] = torch.zeros_like(clone[envs_id])
        self.gross_thrust = clone
        
    def reset(self,):
        self.reset_idx(torch.arange(self.num_envs, dtype=torch.long, device=self.device))

    def detach(self, ):
        self.gross_thrust = self.gross_thrust.clone().detach()