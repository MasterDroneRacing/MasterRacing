# *******************************************************************************
# *                                                                             *
# *  Private and Confidential                                                   *
# *                                                                             *
# *  Unauthorized copying of this file, via any medium is strictly prohibited.  *
# *  Proprietary and confidential.                                              *
# *                                                                             *
# *  © 2025 DiffLab. All rights reserved.                                       *
# *                                                                             *
# *  Author: Yu Feng                                                            *
# *  Data: 2025/02/16     	             *
# *  Contact: yu-feng@sjtu.edu.cn                                               *
# *  Description: None                                                          *
# *******************************************************************************
from diff.lab.envs import ManagerBasedDiffRLEnv
import torch
import torch.nn.functional as F
from isaaclab.managers import SceneEntityCfg

'''
# NOTE: loss function template
def loss_fn(
    env: ManagerBasedDiffRLEnv,
    aligned_states: torch.Tensor, (only active when use_diff_states is True)
    actions: torch.Tensor, (onlyactive  when use_action is True)
    ---more param---
)-> torch.Tensor:
    ---function body---
    return loss
'''
# NOTE: aligned_states: [p, q, v, w], where p is position, q is quaternion, v is linear velocity, w is angular velocity
def target_diff(
    env: ManagerBasedDiffRLEnv,
    aligned_states: torch.Tensor,
    command_name: str,
) -> torch.Tensor:
    desired_pos_w = env.command_manager.get_term(command_name).command_world[:, :3] - env.scene.env_origins
    current_pos_w = aligned_states[:, :3]
    distance_to_goal = torch.norm(desired_pos_w - current_pos_w, p=2, dim=-1)
    return distance_to_goal

def orientation_diff(
    env: ManagerBasedDiffRLEnv,
    aligned_states: torch.Tensor,
):
    orientation = aligned_states[:, 3:7]
    return (orientation - torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)).norm(2, -1)    

def move_in_dir_diff(
    env: ManagerBasedDiffRLEnv,
    aligned_states: torch.Tensor,
    command_name: str,
    threshold: float = 0.1,
) -> torch.Tensor:
    desired_pos_w = env.command_manager.get_term(command_name).command_world[:, :3] - env.scene.env_origins
    current_pos_w = aligned_states[:, :3]
    desired_dir = desired_pos_w - current_pos_w
    distance = desired_dir.norm(2, -1)
    lin_vel_w = aligned_states[:, 7:10]
    return (1 - F.cosine_similarity(lin_vel_w, desired_dir, dim=-1)) * (distance > threshold).float()

def smooth_vel_diff(
    env: ManagerBasedDiffRLEnv,
    aligned_states: torch.Tensor,
    ratio: float = 0.1,
):
    return aligned_states[:, 7:10].norm(2, -1) + ratio * aligned_states[:, 10:13].norm(2, -1)
    
    

########################## Racing Losses ##########################
def racing_target_diff(
    env: ManagerBasedDiffRLEnv,
    aligned_states: torch.Tensor,
    command_name: str,
) -> torch.Tensor:
    desired_pos_w = env.command_manager.get_term(command_name).gate_pose_gt_w[:, :3] - env.scene.env_origins
    current_pos_w = aligned_states[:, :3]
    distance_to_goal = torch.norm(desired_pos_w - current_pos_w, p=2, dim=-1)
    return distance_to_goal

def racing_direction_diff(
    env: ManagerBasedDiffRLEnv,
    aligned_states: torch.Tensor,
    command_name: str,
) -> torch.Tensor:
    desired_pos_w = env.command_manager.get_term(command_name).gate_pose_gt_w[:, :3] - env.scene.env_origins
    current_pos_w = aligned_states[:, :3]
    desired_dir = desired_pos_w - current_pos_w
    desired_dir = desired_dir / desired_dir.norm(dim = 0)
    lin_vel_w = aligned_states[:, 7:10]
    reward = (lin_vel_w * desired_dir).sum(dim = -1)
    return -reward

def racing_falling_diff(
    env: ManagerBasedDiffRLEnv,
    aligned_states: torch.Tensor,
) -> torch.Tensor:
    current_pos_w_z = aligned_states[:, 2]
    loss_falling = 1.0 / (1.0 + 1.0 * current_pos_w_z + 10.0 * current_pos_w_z**2)
    return loss_falling

# def racing_vel_diff(
#     env: ManagerBasedDiffRLEnv,
#     aligned_states: torch.Tensor,
# ) -> torch.Tensor:
#     lin_vel_w_z = aligned_states[:, 9]
#     loss_vel = torch.relu(-lin_vel_w_z)
#     return loss_vel

def racing_vel_diff(
    env: ManagerBasedDiffRLEnv,
    aligned_states: torch.Tensor,
) -> torch.Tensor:
    lin_vel_w_z = aligned_states[:, 7:10]
    loss_vel = torch.mean(lin_vel_w_z**2, dim = -1)
    return loss_vel