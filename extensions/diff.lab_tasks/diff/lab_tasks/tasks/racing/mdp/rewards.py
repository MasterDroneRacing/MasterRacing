# *******************************************************************************
# *                                                                             *
# *  Private and Confidential                                                   *
# *                                                                             *
# *  Unauthorized copying of this file, via any medium is strictly prohibited.  *
# *  Proprietary and confidential.                                              *
# *                                                                             *
# *  © 2024 OctiLab. All rights reserved.                                       *
# *                                                                             *
# *  Author: Su Yang                                                            *
# *  Data: 2025/01/19                                            	            *
# *  Contact:                                                                   *
# *  Description: None                                                          *
# *******************************************************************************

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation
from diff.lab.utils import get_uav_collision_num_ray, LATTICE_TENSOR
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi 
from diff.lab.envs.mdp.actions.diff_action import DiffActions

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def distance_to_goal(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    _desired_pos_b = env.command_manager.get_command(command_name)[:, :3]
    return _desired_pos_b.norm(2, -1)

def lin_vel_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
)-> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    lin_vel = torch.sum(torch.square(asset.data.root_lin_vel_b), dim=1)
    return lin_vel


def ang_vel_reward(
    env: ManagerBasedRLEnv,  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
)-> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ang_vel = torch.sum(torch.square(asset.data.root_ang_vel_b * torch.tensor([1.0, 1.0, 5.0], device=env.device)), dim=1)
    return ang_vel

def target_reward(
        env: ManagerBasedRLEnv, command_name: str
        ) -> torch.Tensor:
    _desired_pos_b = env.command_manager.get_command(command_name)[:, :3]
    distance_to_goal = torch.norm(_desired_pos_b, p=2, dim=-1)
    distance_to_goal_mapped = 1 / (1 + distance_to_goal)
    return distance_to_goal_mapped

def orientation_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    orientation = robot.data.root_quat_w
    return 1 / (1 + torch.norm(orientation - torch.tensor([1, 0, 0, 0], device=env.device), p=2, dim=-1))


def move_in_dir(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.2
):
    robot: Articulation = env.scene[robot_cfg.name]
    lin_vel_b = robot.data.root_lin_vel_b[:, :3]
    target_vec = env.command_manager.get_command("desired_pos_b")[:, :3]
    distance_to_goal = torch.norm(target_vec, p=2, dim=-1)
    return torch.where(distance_to_goal < threshold, 1.0, (F.normalize(lin_vel_b, 2, -1) * F.normalize(target_vec, 2, -1)).sum(-1))

def time_penalty(
    env: ManagerBasedRLEnv,
    threshold: float = 0.2
):
    distance = env.command_manager.get_command("desired_pos_b")[:, :3].norm(2, -1)
    return env.cfg.decimation * env.cfg.sim.dt * torch.ones(env.num_envs, device=env.device) * (distance > threshold)

def reach_target(
    env: ManagerBasedRLEnv,
    threshold: float = 0.2
):
    distance = env.command_manager.get_command("desired_pos_b")[:, :3].norm(2, -1)
    return distance < threshold

def body_ang_acc_l2(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="body"),
):
    robot: Articulation = env.scene[robot_cfg.name]
    body_ang_acc = robot.data.body_ang_acc_w[:, robot_cfg.body_ids].norm(2, dim=-1)
    return torch.sum(body_ang_acc, dim=-1)

def hover_state(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="body"),
    threshold: float = 0.2,
    ratio: float=0.5
):
    robot: Articulation = env.scene[robot_cfg.name]
    distance = env.command_manager.get_command("desired_pos_b")[:, :3].norm(2, -1)
    return (distance < threshold) / (1 + robot.data.root_vel_w.norm(2, -1) + ratio * robot.data.root_ang_vel_w.norm(2, -1))



###### VisFly hover task reward functions ######
def target_reward_visfly(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    _desired_pos_b = env.command_manager.get_command(command_name)[:, :3]
    distance_to_goal = torch.norm(_desired_pos_b, p=2, dim=-1)
    return distance_to_goal

def orientation_reward_visfly(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    orientation = robot.data.root_quat_w
    return torch.norm(orientation - torch.tensor([1, 0, 0, 0], device=env.device), p=2, dim=-1)

def lin_vel_reward_visfly(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    lin_vel = robot.data.root_lin_vel_w.norm(2, dim=-1)
    return lin_vel

def ang_vel_reward_visfly(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    ang_vel = robot.data.root_ang_vel_w.norm(2, dim=-1)
    return ang_vel

def constant_reward(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)

############# Racing Rewards ################
def progress_reward_mine(
    env: ManagerBasedRLEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    robot: Articulation = env.scene[robot_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    return F.cosine_similarity(robot.data.root_lin_vel_b, command_term.command_gt[:, :3], dim=-1)

def track_velocity(
        env: ManagerBasedRLEnv,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        threshold: float = 3.0
):
    robot: Articulation = env.scene[robot_cfg.name]
    return 1 / ((robot.data.root_lin_vel_b.norm(2, dim=-1) - threshold).pow(2) + 1)

def perception_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    command_term = env.command_manager.get_term(command_name)
    vec_to_gate_b = F.normalize(command_term.command_gt[:, :3], p=2, dim=-1)
    # compute the angle between the robot's forward direction and the vector to the gate
    forward_vec = torch.tensor([[1.0, 0.0, 0.0]], device=env.device)
    return F.cosine_similarity(vec_to_gate_b, forward_vec, dim=-1)

def body_rate_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.root_ang_vel_w.norm(2, -1)

def command_body_rate_penalty(
    env: ManagerBasedRLEnv,
    action_name: str = "force_torque",
) -> torch.Tensor:
    action_term: DiffActions = env.action_manager.get_term(action_name)
    max_omega = action_term.action_scale[:, 1:4]
    return torch.norm(env.action_manager.action[:, 1:4].tanh() * max_omega, p=2, dim=1)

def command_rate_penalty(
    env: ManagerBasedRLEnv,
    action_name: str = "force_torque",
) -> torch.Tensor:
    action_term: DiffActions = env.action_manager.get_term(action_name)
    raw_action = env.action_manager.action.tanh()
    raw_prev_action = env.action_manager.prev_action.tanh()
    # convert to ctbr
    ctbr = raw_action * action_term.action_scale + action_term.action_offset
    ctbr_prev = raw_prev_action * action_term.action_scale + action_term.action_offset
    return torch.sum(torch.square(ctbr - ctbr_prev), dim=1)

def lin_vel_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.root_lin_vel_w.norm(2, -1)
    
def success_cross(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.3
):  
    robot: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    vec_to_gate_w = command_term.gate_pose_gt_w[:, :3] - robot.data.root_state_w[:, :3]
    return (vec_to_gate_w.norm(2, -1) < threshold).float() * (1 / (vec_to_gate_w.norm(2, -1)**2 + 1))

def collision_penalty_custom(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
)-> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]

    num_collision = get_uav_collision_num_ray(
        env.scene.terrain.diff_warp_meshes["terrain"],
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        0.09,
        0.05,
        1e3,
        LATTICE_TENSOR.to(env.scene.device),
    )

    return (num_collision > 2.0).float()

def penalize_bad_pose(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
)-> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    euler = euler_xyz_from_quat(robot.data.root_quat_w)
    roll = wrap_to_pi(euler[0])
    pitch = wrap_to_pi(euler[1])

    return torch.bitwise_or((roll.abs() > torch.pi / 2), (pitch.abs() > torch.pi / 2))
