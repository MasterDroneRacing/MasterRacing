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
# *  Data: 2025/01/19                                           	            *
# *  Contact: None                                                              *
# *  Description: None                                                          *
# *******************************************************************************
"""Sub-module containing command generators for pose tracking."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import compute_pose_error, quat_from_euler_xyz, quat_unique, quat_apply_inverse, quat_mul, quat_inv, yaw_quat, quat_apply
from .rewards import command_rate_penalty

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import WorldPoseCommandCfg, RacingCommandCfg

class UniformWorldPoseCommand(CommandTerm):
    """Command generator for generating pose commands uniformly.

    The command generator generates poses by sampling positions uniformly within specified
    regions in cartesian space. For orientation, it samples uniformly the euler angles
    (roll-pitch-yaw) and converts them into quaternion representation (w, x, y, z).

    The position and orientation commands are generated in the base frame of the robot, and not the
    simulation world frame. This means that users need to handle the transformation from the
    base frame to the simulation world frame themselves.

    .. caution::

        Sampling orientations uniformly is not strictly the same as sampling euler angles uniformly.
        This is because rotations are defined by 3D non-Euclidean space, and the mapping
        from euler angles to rotations is not one-to-one.

    """

    cfg: WorldPoseCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: WorldPoseCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.env = env

        # create buffers
        # -- commands: (x, y, z, qw, qx, qy, qz) in root frame
        self.pose_command_w = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "UniformPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command. Shape is (num_envs, 7).

        The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
        """
        return self.pose_command_b

    @property
    def command_world(self,) -> torch.Tensor:
        return self.pose_command_w
    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # compute the error
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.root_state_w[:, :3],
            self.robot.data.root_state_w[:, 3:7],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new pose targets
        # -- position
        r = torch.empty(len(env_ids), device=self.device)
        self.pose_command_w[env_ids, :3] = self.robot.data.root_state_w[env_ids, :3]
        self.pose_command_w[env_ids, 0] += r.uniform_(*self.cfg.ranges.pos_x)
        self.pose_command_w[env_ids, 1] += r.uniform_(*self.cfg.ranges.pos_y)
        self.pose_command_w[env_ids, 2] += r.uniform_(*self.cfg.ranges.pos_z)
        
        # -- orientation
        euler_angles = torch.zeros_like(self.pose_command_w[env_ids, :3])
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        # make sure the quaternion has real part as positive
        self.pose_command_w[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat

    def _update_command(self):
        """Re-target the position command to the current root state."""
        target_vec = self.pose_command_w[:, :3] - self.robot.data.root_pos_w[:, :3]
        self.pose_command_b[:, :3] = quat_apply_inverse(self.robot.data.root_quat_w, target_vec)
        self.pose_command_b[:, 3:] = quat_mul(quat_inv(self.robot.data.root_quat_w), self.pose_command_w[:, 3:])

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                # -- goal pose
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                # -- current body pose
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal pose
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        # -- current root pose
        root_pose_w = self.robot.data.root_state_w
        self.current_pose_visualizer.visualize(root_pose_w[:, :3], root_pose_w[:, 3:7])


class RacingCommand(CommandTerm):
    cfg: RacingCommandCfg

    def __init__(self, cfg: RacingCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.env = env
        # cmd: x,y,z,qw,qx,qy,qz
        self.gate_pose_w = torch.zeros(self.num_envs, 7, device=self.device)
        self.gate_pose_gt_w = torch.zeros(self.num_envs, 7, device=self.device)
        if self.cfg.consecutive_commands:
            self.next_gate_pose_w = torch.zeros(self.num_envs, 7, device=self.device)
            self.next_gate_pose_gt_w = torch.zeros(self.num_envs, 7, device=self.device)

        # self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        # self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["accumulate_gates"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["action_rate"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["avg_lin_spd"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["avg_ang_spd"] = torch.zeros(self.num_envs, device=self.device)

        # initialize
        self.gate_pose = self.env.scene.terrain.extras["gate_pose"].to(torch.float32)
        self.gate_id = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        if self.cfg.consecutive_commands:
            self.next_gate_id = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)

        # initialize intial noise for all envs
        self.noise_range_pos_x = torch.tensor(cfg.noise_ranges.pos_x, device=self.device).view(1, 2).repeat(self.num_envs, 1)
        self.noise_range_pos_y = torch.tensor(cfg.noise_ranges.pos_y, device=self.device).view(1, 2).repeat(self.num_envs, 1)
        self.noise_range_pos_z = torch.tensor(cfg.noise_ranges.pos_z, device=self.device).view(1, 2).repeat(self.num_envs, 1)
        self.noise_range_roll = torch.tensor(cfg.noise_ranges.roll, device=self.device).view(1, 2).repeat(self.num_envs, 1)
        self.noise_range_pitch = torch.tensor(cfg.noise_ranges.pitch, device=self.device).view(1, 2).repeat(self.num_envs, 1)
        self.noise_range_yaw = torch.tensor(cfg.noise_ranges.yaw, device=self.device).view(1, 2).repeat(self.num_envs, 1)
        self.noise_level = torch.ones(self.num_envs, 1, device=self.device)

    def __str__(self) -> str:
        msg = "RacingCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg
    
    @property
    def command(self,) -> torch.Tensor:
        vec_to_gate = self.gate_pose_w[:, :3] - self.robot.data.root_state_w[:, :3]
        gate_pose_b = torch.zeros_like(self.gate_pose_w)
        gate_pose_b[:, :3] = quat_apply_inverse(self.robot.data.root_quat_w, vec_to_gate)
        gate_pose_b[:, 3:] = quat_mul(quat_inv(self.robot.data.root_quat_w), self.gate_pose_w[:, 3:])
        if self.cfg.consecutive_commands:
            gate_to_gate = self.next_gate_pose_w[:, :3] - self.gate_pose_w[:, :3]
            gate_to_gate_pose_b = torch.zeros_like(self.next_gate_pose_w)
            gate_to_gate_pose_b[:, :3] = quat_apply_inverse(self.robot.data.root_quat_w, gate_to_gate)
            gate_to_gate_pose_b[:, 3:] = quat_mul(quat_inv(self.robot.data.root_quat_w), self.next_gate_pose_w[:, 3:])
            return torch.cat((gate_pose_b[:, :3], gate_to_gate_pose_b[:, :3]), dim=-1)
        else:
            return gate_pose_b[:, :3]

    
    @property
    def command_w(self,) -> torch.Tensor:
        return self.gate_pose_w.clone()

    @property
    def command_gt_w(self,) -> torch.Tensor:
        return self.gate_pose_gt_w.clone()

    @property
    def command_gt(self, ) -> torch.Tensor:
        vec_to_gate_gt = self.gate_pose_gt_w[:, :3] - self.robot.data.root_state_w[:, :3]
        gate_pose_gt_b = torch.zeros_like(self.gate_pose_gt_w)
        gate_pose_gt_b[:, :3] = quat_apply_inverse(self.robot.data.root_quat_w, vec_to_gate_gt)
        gate_pose_gt_b[:, 3:] = quat_mul(quat_inv(self.robot.data.root_quat_w), self.gate_pose_gt_w[:, 3:])
        if self.cfg.consecutive_commands:
            gate_to_gate_gt = self.next_gate_pose_gt_w[:, :3] - self.gate_pose_gt_w[:, :3]
            gate_to_gate_pose_gt_b = torch.zeros_like(self.next_gate_pose_gt_w)
            gate_to_gate_pose_gt_b[:, :3] = quat_apply_inverse(self.robot.data.root_quat_w, gate_to_gate_gt)
            gate_to_gate_pose_gt_b[:, 3:] = quat_mul(quat_inv(self.robot.data.root_quat_w), self.next_gate_pose_gt_w[:, 3:])
            return torch.cat((gate_pose_gt_b[:, :3], gate_to_gate_pose_gt_b[:, :3]), dim=-1)
        else:
            return gate_pose_gt_b
    
    def _update_metrics(self):
        # compute the error
        pos_error, rot_error = compute_pose_error(
            self.gate_pose_gt_w[:, :3],
            self.gate_pose_gt_w[:, 3:],
            self.robot.data.root_state_w[:, :3],
            self.robot.data.root_state_w[:, 3:7],
        )
        # self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        # self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)
        self.metrics["accumulate_gates"] += (torch.norm(pos_error, dim=-1) < self.cfg.update_threshold)
        self.metrics["action_rate"] = command_rate_penalty(self.env, "force_torque")
        self.metrics["avg_lin_spd"] = self.robot.data.root_lin_vel_w.norm(2, -1)
        self.metrics["avg_ang_spd"] = self.robot.data.root_ang_vel_b.norm(2, -1)

    def _resample_command(self, env_ids: Sequence[int]):
        # update the gate pose
        self.gate_pose = self.env.scene.terrain.extras["gate_pose"].to(torch.float32)
        # reset metrics
        self.metrics["accumulate_gates"][env_ids] = 0
        # reset to start gate id
        self.gate_id[env_ids] = self.env.scene.terrain.extras["next_gate_id"][self.env.scene.terrain.terrain_types, self.env.scene.terrain.terrain_levels][env_ids]
        if self.cfg.consecutive_commands:
            self.next_gate_id[env_ids] = (self.gate_id[env_ids] + 1) % self.env.scene.terrain.cfg.terrain_generator.sub_terrains["circular"].num_gate
        # reset self.gate_pose_w
        position = (self.gate_pose[self.env.scene.terrain.terrain_types, self.env.scene.terrain.terrain_levels, self.gate_id, :3] + self.env.scene.env_origins)[env_ids]
        quat = self.gate_pose[self.env.scene.terrain.terrain_types, self.env.scene.terrain.terrain_levels, self.gate_id, 3:][env_ids]
        self.gate_pose_gt_w[env_ids, :3] = position.clone()
        self.gate_pose_gt_w[env_ids, 3:] = quat.clone()
        self.gate_pose_w[env_ids, :3] = position.clone()
        self.gate_pose_w[env_ids, 3:] = quat.clone()
        if self.cfg.consecutive_commands:
            next_position = (self.gate_pose[self.env.scene.terrain.terrain_types, self.env.scene.terrain.terrain_levels, self.next_gate_id, :3] + self.env.scene.env_origins)[env_ids]
            next_quat = self.gate_pose[self.env.scene.terrain.terrain_types, self.env.scene.terrain.terrain_levels, self.next_gate_id, 3:][env_ids]
            self.next_gate_pose_gt_w[env_ids, :3] = next_position.clone()
            self.next_gate_pose_gt_w[env_ids, 3:] = next_quat.clone()
            self.next_gate_pose_w[env_ids, :3] = next_position.clone()
            self.next_gate_pose_w[env_ids, 3:] = next_quat.clone()
        if self.cfg.add_noise:
            r = torch.empty(len(env_ids), device=self.device)
            self.gate_pose_w[env_ids, 0] += self.noise_range_pos_x[env_ids, 0] + r.uniform_() * (self.noise_range_pos_x[env_ids, 1] - self.noise_range_pos_x[env_ids, 0])
            self.gate_pose_w[env_ids, 1] += self.noise_range_pos_y[env_ids, 0] + r.uniform_() * (self.noise_range_pos_y[env_ids, 1] - self.noise_range_pos_y[env_ids, 0])
            self.gate_pose_w[env_ids, 2] += self.noise_range_pos_z[env_ids, 0] + r.uniform_() * (self.noise_range_pos_z[env_ids, 1] - self.noise_range_pos_z[env_ids, 0])
            euler_angles = torch.zeros_like(self.gate_pose_w[env_ids, :3])
            euler_angles[:, 0] = self.noise_range_roll[env_ids, 0] + r.uniform_() * (self.noise_range_roll[env_ids, 1] - self.noise_range_roll[env_ids, 0])
            euler_angles[:, 1] = self.noise_range_pitch[env_ids, 0] + r.uniform_() * (self.noise_range_pitch[env_ids, 1] - self.noise_range_pitch[env_ids, 0])
            euler_angles[:, 2] = self.noise_range_yaw[env_ids, 0] + r.uniform_() * (self.noise_range_yaw[env_ids, 1] - self.noise_range_yaw[env_ids, 0])
            noise_quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
            self.gate_pose_w[env_ids, 3:] = quat_unique(quat_mul(noise_quat, quat)) if self.cfg.make_quat_unique else quat_mul(noise_quat, quat)
            if self.cfg.consecutive_commands:
                r = torch.empty(len(env_ids), device=self.device)
                self.next_gate_pose_w[env_ids, 0] += self.noise_range_pos_x[env_ids, 0] + r.uniform_() * (self.noise_range_pos_x[env_ids, 1] - self.noise_range_pos_x[env_ids, 0])
                self.next_gate_pose_w[env_ids, 1] += self.noise_range_pos_y[env_ids, 0] + r.uniform_() * (self.noise_range_pos_y[env_ids, 1] - self.noise_range_pos_y[env_ids, 0])
                self.next_gate_pose_w[env_ids, 2] += self.noise_range_pos_z[env_ids, 0] + r.uniform_() * (self.noise_range_pos_z[env_ids, 1] - self.noise_range_pos_z[env_ids, 0])
                euler_angles = torch.zeros_like(self.next_gate_pose_w[env_ids, :3])
                euler_angles[:, 0] = self.noise_range_roll[env_ids, 0] + r.uniform_() * (self.noise_range_roll[env_ids, 1] - self.noise_range_roll[env_ids, 0])
                euler_angles[:, 1] = self.noise_range_pitch[env_ids, 0] + r.uniform_() * (self.noise_range_pitch[env_ids, 1] - self.noise_range_pitch[env_ids, 0])
                euler_angles[:, 2] = self.noise_range_yaw[env_ids, 0] + r.uniform_() * (self.noise_range_yaw[env_ids, 1] - self.noise_range_yaw[env_ids, 0])
                noise_quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
                self.next_gate_pose_w[env_ids, 3:] = quat_unique(quat_mul(noise_quat, next_quat)) if self.cfg.make_quat_unique else quat_mul(noise_quat, next_quat)

    def _update_command(self):
        pos_error = torch.norm(self.gate_pose_gt_w[:, :3] - self.robot.data.root_state_w[:, :3], dim=-1)
        achieved_env_ids = pos_error < self.cfg.update_threshold
        # update next gate ids
        self.gate_id[achieved_env_ids] = (self.gate_id[achieved_env_ids] + 1) % self.env.scene.terrain.cfg.terrain_generator.sub_terrains["circular"].num_gate
        if self.cfg.consecutive_commands:
            self.next_gate_id[achieved_env_ids] = (self.gate_id[achieved_env_ids] + 1) % self.env.scene.terrain.cfg.terrain_generator.sub_terrains["circular"].num_gate
        position = (self.gate_pose[self.env.scene.terrain.terrain_types, self.env.scene.terrain.terrain_levels, self.gate_id, :3] + self.env.scene.env_origins)[achieved_env_ids]
        quat = self.gate_pose[self.env.scene.terrain.terrain_types, self.env.scene.terrain.terrain_levels, self.gate_id, 3:][achieved_env_ids]
        self.gate_pose_gt_w[achieved_env_ids, :3] = position.clone()
        self.gate_pose_gt_w[achieved_env_ids, 3:] = quat.clone()
        self.gate_pose_w[achieved_env_ids, :3] = position.clone()
        self.gate_pose_w[achieved_env_ids, 3:] = quat.clone()
        if self.cfg.consecutive_commands:
            next_position = (self.gate_pose[self.env.scene.terrain.terrain_types, self.env.scene.terrain.terrain_levels, self.next_gate_id, :3] + self.env.scene.env_origins)[achieved_env_ids]
            next_quat = self.gate_pose[self.env.scene.terrain.terrain_types, self.env.scene.terrain.terrain_levels, self.next_gate_id, 3:][achieved_env_ids]
            self.next_gate_pose_gt_w[achieved_env_ids, :3] = next_position.clone()
            self.next_gate_pose_gt_w[achieved_env_ids, 3:] = next_quat.clone()
            self.next_gate_pose_w[achieved_env_ids, :3] = next_position.clone()
            self.next_gate_pose_w[achieved_env_ids, 3:] = next_quat.clone()

        if self.cfg.add_noise and achieved_env_ids.any():
            r = torch.empty(int((achieved_env_ids).sum()), device=self.device)
            self.gate_pose_w[achieved_env_ids, 0] += self.noise_range_pos_x[achieved_env_ids, 0] + r.uniform_() * (self.noise_range_pos_x[achieved_env_ids, 1] - self.noise_range_pos_x[achieved_env_ids, 0])
            self.gate_pose_w[achieved_env_ids, 1] += self.noise_range_pos_y[achieved_env_ids, 0] + r.uniform_() * (self.noise_range_pos_y[achieved_env_ids, 1] - self.noise_range_pos_y[achieved_env_ids, 0])
            self.gate_pose_w[achieved_env_ids, 2] += self.noise_range_pos_z[achieved_env_ids, 0] + r.uniform_() * (self.noise_range_pos_z[achieved_env_ids, 1] - self.noise_range_pos_z[achieved_env_ids, 0])
            euler_angles = torch.zeros_like(self.gate_pose_w[achieved_env_ids, :3])
            euler_angles[:, 0] = self.noise_range_roll[achieved_env_ids, 0] + r.uniform_() * (self.noise_range_roll[achieved_env_ids, 1] - self.noise_range_roll[achieved_env_ids, 0])
            euler_angles[:, 1] = self.noise_range_pitch[achieved_env_ids, 0] + r.uniform_() * (self.noise_range_pitch[achieved_env_ids, 1] - self.noise_range_pitch[achieved_env_ids, 0])
            euler_angles[:, 2] = self.noise_range_yaw[achieved_env_ids, 0] + r.uniform_() * (self.noise_range_yaw[achieved_env_ids, 1] - self.noise_range_yaw[achieved_env_ids, 0])
            noise_quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
            self.gate_pose_w[achieved_env_ids, 3:] = quat_unique(quat_mul(noise_quat, quat)) if self.cfg.make_quat_unique else quat_mul(noise_quat, quat)
            if self.cfg.consecutive_commands:
                r = torch.empty(int((achieved_env_ids).sum()), device=self.device)
                self.next_gate_pose_w[achieved_env_ids, 0] += self.noise_range_pos_x[achieved_env_ids, 0] + r.uniform_() * (self.noise_range_pos_x[achieved_env_ids, 1] - self.noise_range_pos_x[achieved_env_ids, 0])
                self.next_gate_pose_w[achieved_env_ids, 1] += self.noise_range_pos_y[achieved_env_ids, 0] + r.uniform_() * (self.noise_range_pos_y[achieved_env_ids, 1] - self.noise_range_pos_y[achieved_env_ids, 0])
                self.next_gate_pose_w[achieved_env_ids, 2] += self.noise_range_pos_z[achieved_env_ids, 0] + r.uniform_() * (self.noise_range_pos_z[achieved_env_ids, 1] - self.noise_range_pos_z[achieved_env_ids, 0])
                euler_angles = torch.zeros_like(self.next_gate_pose_w[achieved_env_ids, :3])
                euler_angles[:, 0] = self.noise_range_roll[achieved_env_ids, 0] + r.uniform_() * (self.noise_range_roll[achieved_env_ids, 1] - self.noise_range_roll[achieved_env_ids, 0])
                euler_angles[:, 1] = self.noise_range_pitch[achieved_env_ids, 0] + r.uniform_() * (self.noise_range_pitch[achieved_env_ids, 1] - self.noise_range_pitch[achieved_env_ids, 0])
                euler_angles[:, 2] = self.noise_range_yaw[achieved_env_ids, 0] + r.uniform_() * (self.noise_range_yaw[achieved_env_ids, 1] - self.noise_range_yaw[achieved_env_ids, 0])
                noise_quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
                self.next_gate_pose_w[achieved_env_ids, 3:] = quat_unique(quat_mul(noise_quat, next_quat)) if self.cfg.make_quat_unique else quat_mul(noise_quat, next_quat)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                # -- goal pose
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                # -- goal pose gt
                self.goal_pose_gt_visualizer = VisualizationMarkers(self.cfg.goal_pose_gt_visualizer_cfg)
                # -- current body pose
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
            self.goal_pose_gt_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)
                self.goal_pose_gt_visualizer.set_visibility(False)
    
    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal pose
        self.goal_pose_visualizer.visualize(self.gate_pose_w[:, :3], self.gate_pose_w[:, 3:])
        # -- goal pose gt
        self.goal_pose_gt_visualizer.visualize(self.gate_pose_gt_w[:, :3], self.gate_pose_gt_w[:, 3:])
        # -- current root pose
        root_pose_w = self.robot.data.root_state_w
        self.current_pose_visualizer.visualize(root_pose_w[:, :3], root_pose_w[:, 3:7])

    def update_noise_level(self, envs_id: torch.Tensor, enhanced: torch.Tensor, decayed: torch.Tensor, enhanced_prop=0.01, decayed_prop=0.01):        
        assert len(envs_id) == len(enhanced), "envs_id, enhanced and decayed should have the same length"
        up = torch.where(enhanced, (1. + enhanced_prop), 1.).view(-1, 1)
        down = torch.where(decayed,  (1. - decayed_prop), 1.).view(-1, 1)
        self.noise_level[envs_id] *= up
        self.noise_level[envs_id] *= down
        self.noise_range_pos_x[envs_id] *= up
        self.noise_range_pos_x[envs_id] *= down
        self.noise_range_pos_y[envs_id] *= up
        self.noise_range_pos_y[envs_id] *= down
        self.noise_range_pos_z[envs_id] *= up
        self.noise_range_pos_z[envs_id] *= down
        self.noise_range_roll[envs_id] *= up
        self.noise_range_roll[envs_id] *= down
        self.noise_range_pitch[envs_id] *= up
        self.noise_range_pitch[envs_id] *= down
        self.noise_range_yaw[envs_id] *= up
        self.noise_range_yaw[envs_id] *= down