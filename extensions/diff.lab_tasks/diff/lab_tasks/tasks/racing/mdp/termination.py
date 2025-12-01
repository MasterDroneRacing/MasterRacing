from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.command_manager import CommandTerm

def out_of_bound(
        env: ManagerBasedRLEnv, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        bounds: tuple = (-0.05, 10.0)
    ) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    died = torch.logical_or(asset.data.root_pos_w[:, 2] < bounds[0], asset.data.root_pos_w[:, 2] > bounds[1])
    return died

def bad_pose(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
)-> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    euler = euler_xyz_from_quat(robot.data.root_quat_w)
    roll = wrap_to_pi(euler[0])
    pitch = wrap_to_pi(euler[1])

    return torch.bitwise_or((roll.abs() > torch.pi / 2), (pitch.abs() > torch.pi / 2))
