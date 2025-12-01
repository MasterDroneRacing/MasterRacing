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
# *  Data: 2025/03/06     	                                                    *
# *  Contact: yu-feng@sjtu.edu.cn                                               *
# *  Description: None                                                          *
# *******************************************************************************

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .lee_controller_position_and_yaw_cfg import LeePositionAndYawControllerCfg
from isaaclab.utils.math import matrix_from_quat, quat_from_matrix, quat_rotate_inverse
import torch.nn.functional as F

class LeePositionAndYawController:
    r""" Position and yaw controller for a quadrotor based on its differential flatness.
         Given a desired position, velocity, acceleration and yaw sampled from a trajectory,
         the controller computes the desired thrust and body rates to track the trajectory.
    Reference:
    1. Mellinger D, Kumar V. Minimum snap trajectory generation and control for quadrotors[C]
       //2011 IEEE international conference on robotics and automation. IEEE, 2011: 2520-2525.
    2. Lee T, Leok M, McClamroch N H. Geometric tracking control of a quadrotor UAV on SE (3)[C]//
       49th IEEE conference on decision and control (CDC). IEEE, 2010: 5420-5425.
    """
    def __init__(self, cfg:LeePositionAndYawControllerCfg, num_envs:int, device:str):
        self.cfg = cfg
        self.mass = cfg.mass    
        self.inertial = torch.tensor(cfg.inertial, device=device)     # shape: (3, 3)
        self.max_fb_acc = cfg.max_fb_acc
        self.g = cfg.gravity_norm
        self.num_envs = num_envs
        self.device = device
        self.kp = torch.tensor(cfg.k_p, device=device)
        self.kv = torch.tensor(cfg.k_v, device=device)
        self.kq = torch.tensor(cfg.k_q, device=device)
        self.kw = torch.tensor(cfg.k_w, device=device)

    @property
    def command_dim(self,):
        return 3 + 1 + 3 + 3        # position, yaw, velocity, acceleration
    
    @property
    def action_dim(self,):
        return 4                # thrust and body rates
    
    def reset(self,):
        pass

    def compute(self, 
            cur_pos: torch.Tensor, 
            cur_quat: torch.Tensor, 
            cur_lin_vel: torch.Tensor, 
            cur_ang_vel: torch.Tensor,
            goal_pos: torch.Tensor,
            goal_yaw: torch.Tensor,
            goal_vel: torch.Tensor | None = None,
            goal_acc: torch.Tensor | None = None,):
        """ Compute control commands given current state and desired trajectory.
            Args:
                - `cur_pos`: current position, shape: (num_envs, 3)
                - `cur_quat`: current quaternion, shape: (num_envs, 4)
                - `cur_lin_vel`: current linear velocity, shape: (num_envs, 3)
                - `cur_ang_vel`: current angular velocity, shape: (num_envs, 3)
                - `goal_pos`: desired position, shape: (num_envs, 3)
                - `goal_yaw`: desired yaw angle, shape: (num_envs,)
                - `goal_vel`: desired velocity, shape: (num_envs, 3)
                - `goal_acc`: desired acceleration, shape: (num_envs, 3)

            Returns:
                `ctbr`:  thrust and body rates, shape: (num_envs, 4)
        """
        # transform current quaternion to rotation matrix
        current_R = matrix_from_quat(cur_quat)
        z_w = torch.tensor([0., 0., 1.], device=self.device)
        z_B = current_R[..., :, -1]        # z-axis in world frame, shape: (num_envs, 3)
        z_B = F.normalize(z_B, 2, -1)       # normalize z-axis
        # compute errors
        error_pos = (cur_pos - goal_pos)
        if goal_vel is not None:
            error_vel = (cur_lin_vel - goal_vel)
        else:
            error_vel = torch.zeros_like(cur_lin_vel)
        if goal_acc is not None:
            acc_feedforward = goal_acc        
        else:
            acc_feedforward = torch.zeros_like(cur_lin_vel)
        # compute feedback acceleration
        acc_fb = - self.kp * error_pos - self.kv * error_vel
        acc_fb = torch.min(torch.norm(acc_fb, dim=-1, keepdim=True), torch.tensor(self.max_fb_acc, device=self.device)) * F.normalize(acc_fb, 2, -1)
        # compute desired thrust, F_des: (num_envs, 3)
        F_des = self.mass * (acc_fb + self.g * z_w + acc_feedforward)
        # project the desired thrust onto the actual body frame z-axis
        u1 =  torch.sum(F_des * z_B, dim=-1, keepdim=True)    # shape: (num_envs,)

        # compute desired rotation matrix
        z_B_des = F.normalize(F_des, 2, -1)       # desired z-axis in body frame, shape: (num_envs, 3)
        x_C_des = torch.stack([
            torch.cos(goal_yaw), 
            torch.sin(goal_yaw), 
            torch.zeros_like(goal_yaw)], dim=-1)  # shape: (num_envs, 3)
        y_B_des = F.normalize(torch.cross(z_B_des, x_C_des, dim=-1), 2, -1)    # desired y-axis
        x_B_des = torch.cross(y_B_des, z_B_des, dim=-1)    # desired x-axis
        R_des = torch.stack([x_B_des, y_B_des, z_B_des], dim=-1)    # desired rotation matrix, shape: (num_envs, 3, 3) 
        quat_des = quat_from_matrix(R_des)    # desired quaternion

        # compute rotation error
        error_R = 0.5 * (
            torch.bmm(R_des.transpose(-2, -1), current_R) 
            - torch.bmm(current_R.transpose(-2, -1), R_des)
        ) # error_R is a skew-symmetric matrix
        # turn skew-symmetric matrix to vector
        error_R = torch.stack([
            error_R[:, 2, 1],
            error_R[:, 0, 2],
            error_R[:, 1, 0]
        ], dim=-1)   # shape: (num_envs, 3)

        body_rate = quat_rotate_inverse(cur_quat, cur_ang_vel)
        error_rate = body_rate
        ang_acc = (
            - self.kq * error_R
            - self.kw * error_rate
            + body_rate.cross(body_rate)
        )
        torque = (self.inertial @ ang_acc.unsqueeze(-1)).squeeze(-1)    # shape: (num_envs, 3)


        return torch.cat([u1, torque], dim=-1), quat_des    # shape: (num_envs, 4)




        




        
