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
# *  Data: 2025/04/28     	             *
# *  Contact: yu-feng@sjtu.edu.cn                                               *
# *  Description: None                                                          *
# *******************************************************************************
from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils

from isaaclab.envs.mdp.events import _randomize_prop_by_op
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from diff.lab.envs.mdp.actions.diff_action import DiffActions
from diff.lab.terrains import TerrainImporterCfg as diff_TerrainImporterCfg
from isaaclab.sensors import RayCasterCameraCfg,RayCasterCamera
def randomize_articulation_mass_and_inertia(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    mass_distribution_params: tuple[float, float] = (-0.01, 0.01),
    mass_operation: Literal["add", "scale", "abs"] = "add",
    mass_distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    inertia_distribution_params: tuple[float, float] = (0.9, 1.1),
    inertia_operation: Literal["add", "scale", "abs"] = "scale",
    inertia_distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform"
):  
    # NOTE recompute inertia by default, since the mass randomization is done on the default values

    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    assert isinstance(asset, Articulation), f"Only support randomize the inertias of articulation"

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")


    ############################# Modify mass ######################################
    # get the current masses of the bodies (num_assets, num_bodies)
    masses = asset.root_physx_view.get_masses()

    # apply randomization on default values
    # this is to make sure when calling the function multiple times, the randomization is applied on the
    # default values and not the previously randomized values
    masses[env_ids[:, None], body_ids] = asset.data.default_mass[env_ids[:, None], body_ids].clone()

    # sample from the given range
    # note: we modify the masses in-place for all environments
    #   however, the setter takes care that only the masses of the specified environments are modified
    masses = _randomize_prop_by_op(
        masses, mass_distribution_params, env_ids, body_ids, operation=mass_operation, distribution=mass_distribution
    )

    # set the mass into the physics simulation
    asset.root_physx_view.set_masses(masses, env_ids)


    ############################# Modify inertia ######################################
    # recompute inertia tensors
    # compute the ratios of the new masses to the initial masses
    ratios = masses[env_ids[:, None], body_ids] / asset.data.default_mass[env_ids[:, None], body_ids]
    # scale the inertia tensors by the the ratios
    # since mass randomization is done on default values, we can use the default inertia tensors
    inertias = asset.root_physx_view.get_inertias()
    # inertia has shape: (num_envs, num_bodies, 9) for articulation
    inertias[env_ids[:, None], body_ids] = (
        asset.data.default_inertia[env_ids[:, None], body_ids] * ratios[..., None]
    )

    # sample from the given range
    # note: we modify the inertias in-place for all environments
    #   however, the setter takes care that only the inertias of the specified environments are modified    
    if inertia_operation == "scale":
        # scale the inertia tensors by the given range
        scale = torch.rand((len(env_ids), len(body_ids), 9), device=inertias.device) \
                * (inertia_distribution_params[1] - inertia_distribution_params[0]) + inertia_distribution_params[0]
        
        inertias[env_ids[:, None], body_ids] = inertias[env_ids[:, None], body_ids] * scale

    # set the inertia into physics simulation
    asset.root_physx_view.set_inertias(inertias, env_ids)

def randomize_rate_controller_gain_and_thrust_delay(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    action_name: str,
    pid_scale_factor: tuple[float, float] = (0.9, 1.1),
    thrust_delay_scale_factor: tuple[float, float] = (0.9, 1.1)
):
    action_term:DiffActions = env.action_manager.get_term(action_name)
    if hasattr(action_term.controller, "rate_gain_p"):
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=env.device)
        action_term.controller.rate_gain_p[env_ids] = action_term.controller.rate_gain_p[env_ids] * \
            ( torch.rand((len(env_ids), 3), device=env.device) * (pid_scale_factor[1] - pid_scale_factor[0]) + pid_scale_factor[0] )
    if hasattr(action_term.controller, "rate_gain_i"):
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=env.device)
        action_term.controller.rate_gain_i[env_ids] = action_term.controller.rate_gain_i[env_ids] * \
            ( torch.rand((len(env_ids), 3), device=env.device) * (pid_scale_factor[1] - pid_scale_factor[0]) + pid_scale_factor[0] )
    if hasattr(action_term.controller, "rate_gain_d"):
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=env.device)
        action_term.controller.rate_gain_d[env_ids] = action_term.controller.rate_gain_d[env_ids] * \
            ( torch.rand((len(env_ids), 3), device=env.device) * (pid_scale_factor[1] - pid_scale_factor[0]) + pid_scale_factor[0] )
    if hasattr(action_term.controller, "thrust_ctrl_delay"):
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=env.device)
        action_term.controller.thrust_ctrl_delay[env_ids] = action_term.controller.thrust_ctrl_delay[env_ids] * \
            ( torch.rand((len(env_ids), 1), device=env.device) * (thrust_delay_scale_factor[1] - thrust_delay_scale_factor[0]) + thrust_delay_scale_factor[0] )
    if hasattr(action_term.controller, "torque_ctrl_delay"):
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=env.device)
        action_term.controller.torque_ctrl_delay[env_ids] = action_term.controller.torque_ctrl_delay[env_ids] * \
            ( torch.rand((len(env_ids), 3), device=env.device) * (thrust_delay_scale_factor[1] - thrust_delay_scale_factor[0]) + thrust_delay_scale_factor[0] )

def reset_root_state_racing(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    
    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    roll_samples = rand_samples[:, 3]
    pitch_samples = rand_samples[:, 4]
    _yaw_samples = rand_samples[:, 5]
    next_gate_id = env.scene.terrain.extras["next_gate_id"][env.scene.terrain.terrain_types, env.scene.terrain.terrain_levels]
    gate_pose = env.scene.terrain.extras["gate_pose"].to(torch.float32)
    gate_position = (gate_pose[env.scene.terrain.terrain_types, env.scene.terrain.terrain_levels, next_gate_id, :3] + env.scene.env_origins)[env_ids]
    towards_vec = gate_position - positions
    yaw_samples = math_utils.wrap_to_pi(torch.atan2(towards_vec[:, 1], towards_vec[:, 0]))
    yaw_samples = yaw_samples + _yaw_samples
    orientations_delta = math_utils.quat_from_euler_xyz(roll_samples, pitch_samples, yaw_samples)
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_link_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_com_velocity_to_sim(velocities, env_ids=env_ids)


def reset_terrain_period(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
):
    """
    Reset the terrain periodically.
    """
    print("Re-generate terrain...")
    from time import time
    t0 = time()
    import omni.isaac.core.utils.prims as prim_utils
    prim_utils.delete_prim(env.scene.terrain.cfg.prim_path + "/terrain")
    for asset_name, asset_cfg in env.scene.cfg.__dict__.items():
        if isinstance(asset_cfg, diff_TerrainImporterCfg):
            # terrains are special entities since they define environment origins
            asset_cfg.num_envs = env.scene.cfg.num_envs
            asset_cfg.env_spacing = env.scene.cfg.env_spacing
            env.scene.terrain = asset_cfg.class_type(asset_cfg)
            env.scene.env_origins = env.scene.terrain.env_origins
    for asset_name, asset_cfg in env.scene.cfg.__dict__.items():
        if isinstance(asset_cfg, RayCasterCameraCfg):
            # reconfigure ray-caster cameras
            env.scene[asset_name]._initialize_warp_meshes()
    env.reset() # re-initialize the scene with the new terrain
    t1 = time()
    print(f"Terrain re-generated in {t1 - t0:.2f} seconds")