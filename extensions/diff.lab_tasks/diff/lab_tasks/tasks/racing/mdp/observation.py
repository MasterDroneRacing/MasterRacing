from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import FrameTransformerData
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat
from isaaclab.sensors import TiledCamera, Camera, RayCasterCamera
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv,ManagerBasedEnv

def desired_position_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), command_name: str = "desired_pos_b") -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    desired_pos_b = env.command_manager.get_command(command_name)[:, :3]
    return desired_pos_b

def base_orientation_r(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), add_noise: bool = False) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    q_gt = asset.data.root_quat_w.clone()

    if add_noise:
        euler = torch.randn(env.num_envs, 3, device=env.device) * 0.05
        q_noise = math_utils.quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
        q_gt = math_utils.quat_mul(q_gt, q_noise)
        
    base_orientation = matrix_from_quat(q_gt)
    return base_orientation[:, 2, :]

def base_orientation_q(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    base_orientation = asset.data.root_quat_w
    return base_orientation

def modified_generated_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    return env.command_manager._terms[command_name].command.clone()

def modified_generated_commands_gt(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    return env.command_manager._terms[command_name].command_gt.clone()

def modified_base_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), add_noise: bool = False) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if add_noise:
        return asset.data.root_com_lin_vel_b * (1 + torch.randn_like(asset.data.root_com_lin_vel_b) * 0.03)
    return asset.data.root_com_lin_vel_b

def modified_last_action(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    ctbr = env.action_manager.get_term(action_name).raw_actions.tanh() * env.action_manager.get_term(action_name).action_scale + env.action_manager.get_term(action_name).action_offset
    ctbr[:, 0] /= env.action_manager.get_term(action_name)._robot_mass
    return ctbr

def depth_image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "distance_to_image_plane",
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = True,
    add_noise: bool = False,
) -> torch.Tensor:
    assert ("distance_to" in data_type) or ("depth" in data_type), f"Only depth images are supported, got {data_type}"
    # extract the used quantities (to enable type-hinting)
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]

    # obtain the input image
    images = sensor.data.output[data_type].clone()

    # depth image conversion
    if (data_type == "distance_to_camera") and convert_perspective_to_orthogonal:
        images = math_utils.orthogonalize_perspective_depth(images, sensor.data.intrinsic_matrices)
    
    if add_noise:
        images *= (1 + torch.randn_like(images) * 0.02)

    # rgb/depth image normalization
    if normalize:
        if "distance_to" in data_type or "depth" in data_type:
            images[images == float("inf")] = 0
            images[images > 10] = 10
            images /= 10

    return images.view(images.shape[0], -1)


def cross_obs(
    env: ManagerBasedRLEnv,
    reward_name:  str = "success_cross",
):  
    if hasattr(env, "reward_manager") and hasattr(env.reward_manager, "_step_reward") and reward_name in env.reward_manager._term_names:
        return (env.reward_manager._step_reward[:, env.reward_manager._term_names.index(reward_name)] > 0).float().unsqueeze(-1)
    else:
        return torch.zeros(env.num_envs, 1, device=env.device)
    
   