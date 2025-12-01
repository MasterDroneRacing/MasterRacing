from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import yaml
import os.path as osp

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm
from diff.lab.controllers import ThrustController, PSController, LVController, CTBRController
from ..dynamics.droneDynamics import DroneDynamics

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from .diff_action_cfg import DiffActionCfg

class DiffActions(ActionTerm):
    cfg: DiffActionCfg
    def __init__(self, cfg:DiffActionCfg, env:ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self.cfg = cfg
        self.env = env
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.dt = self.env.cfg.sim.dt * self.env.cfg.decimation
        
        # check type
        assert cfg.command_type == "PSController" or cfg.command_type == "LVController" or cfg.command_type == "CTBRController" \
        or cfg.command_type == "ThrustController", "Currently only support `CTBRController`, `LVController`, `PSController`, `ThrustController`"
        
        self.command_type = cfg.command_type

        self.cmd_scaled = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        # body ids for force and torque acting on 
        self.rotor_ids, self.rotor_names = self.robot.find_bodies(self.cfg.rotor_names, preserve_order = True)
        self.body_ids, self.body_names = self.robot.find_bodies("body", preserve_order = True)
        # raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        # quadrotor's control input
        self.force_rotors = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self.force_body = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self.torque_rotors = torch.zeros(self.num_envs, 4, 3, device=self.device)
        self.torque_body = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # quadrotor's state
        self.states_all = torch.zeros(self.num_envs, 13, device=self.device)
        self.pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.lin_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.ang_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.lin_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.ang_vel_b = torch.zeros(self.num_envs, 3, device=self.device)
        
        # drone's physical paramter (NOTE Should be set up in USD file).
        self._robot_mass = self.robot.root_physx_view.get_masses().sum(-1).to(self.device)     # kg
        self._robot_weight = (self._robot_mass * abs(self.env.cfg.sim.gravity[2]))    # N
        # TODO update the robot inertia computation
        # self.robot_inertia = self.robot.root_physx_view.get_inertias().sum(1).to(self.device).reshape(-1, 3, 3)
        self.robot_inertia = torch.tensor([[0.0015, 0.0, 0.0, 0.0, 0.002, 0.0, 0.0, 0.0, 0.004]], device=self.device).reshape(1, 3, 3).repeat(self.num_envs, 1, 1)

        # initialize drone dynamics model
        self.drone_dynamics = DroneDynamics(num_envs = self.num_envs, 
                                            mass = self._robot_mass, 
                                            inertia = self.robot_inertia,
                                            dt = self.dt,
                                            decimation = self.env.cfg.decimation,
                                            random_drag = self.cfg.random_drag,
                                            device=self.device)
        
        # Additional parameters for controller
        self.controller_cfg = self.cfg.controller_cfg
        # get scale factor for action pre-processing
        self._get_scale_factor()

        if self.command_type != "ThrustController":
            self.controller = self.controller_cfg.class_type(cfg=self.controller_cfg, num_envs=self.num_envs, device=self.device, mass=self._robot_mass, inertia=self.robot_inertia, dt=self.dt)
        else:   # ThrustController 
            self.controller = ThrustController(num_envs=self.num_envs, device=self.device, params={
                "motor_tau": self.controller_cfg.motor_tau,
                "dt": self.dt,
                "thrustmap": self.controller_cfg.thrustmap,
                "arm_length": self.controller_cfg.arm_length,
                "kappa": self.controller_cfg.kappa
            })
        # for domain randmization
        self.thr_est_error = 1 + torch.randn(self.num_envs, device=self.device) * 0.02

        # action buffer for simulating action delay
        self.action_lag = self.cfg.action_lag
        self.action_buffer = [torch.zeros(self.num_envs, self.action_dim, device=self.device) for _ in range(self.action_lag)]

    """
    Properties.
    """
    @property
    def action_dim(self) -> int:
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions
    
    """
    Operations.
    """

    def get_terminated(self):
        terminated = self.env.termination_manager.terminated
        terminated_env_ids = terminated.nonzero(as_tuple=False).squeeze(-1)
        return terminated_env_ids
    
    def get_timeout(self):
        timeout = self.env.termination_manager.time_outs
        timeout_env_ids = timeout.nonzero(as_tuple=False).squeeze(-1)
        return timeout_env_ids
    
    def get_dones(self):
        dones = self.env.termination_manager.dones
        dones_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        return dones_ids
    
    def get_state_from_sim(self):
        """
        All states here are non-differentiable (directly obtained from isaac-sim).
        """
        self.states_all = self.robot.data.root_state_w.clone()
        self.states_all[:, :3] = self.states_all[:, :3] -  self.env.scene.env_origins[:, :3]
        self.pos = self.states_all[:, :3]
        self.quat = self.states_all[:, 3:7]
        self.lin_vel_w = self.states_all[:, 7:10]
        self.ang_vel_w = self.states_all[:, 10:13]
        self.ang_vel_b = math_utils.quat_apply_inverse(self.quat, self.ang_vel_w)  #self.robot.data.root_ang_vel_w
        self.lin_vel_b = math_utils.quat_apply_inverse(self.quat, self.lin_vel_w)  #self.robot.data.root_lin_vel_b
        self.lin_acc_w = self.robot.data.body_lin_acc_w[:, 0, :].clone()
        self.ang_acc_w = self.robot.data.body_ang_acc_w[:, 0, :].clone()
        self.lin_acc_b = math_utils.quat_apply_inverse(self.quat, self.lin_acc_w)  #self.robot.data.body_lin_acc_b[:, 0, :]
        self.ang_acc_b = math_utils.quat_apply_inverse(self.quat, self.ang_acc_w)  #self.robot.data.body_ang_acc_b[:, 0, :]
        
        return {
            "pos": self.pos,
            "quat": self.quat,
            "lin_vel_w": self.lin_vel_w,
            "ang_vel_w": self.ang_vel_w,
            "lin_vel_b": self.lin_vel_b,
            "ang_vel_b": self.ang_vel_b,
            "lin_acc_w": self.lin_acc_w,
            "ang_acc_w": self.ang_acc_w,
            "lin_acc_b": self.lin_acc_b,
            "ang_acc_b": self.ang_acc_b
        }

    def process_actions(self, actions: torch.Tensor):   # [NOTE] called every environment step
        """
        Process actions using specified controller, and step dynamics to get the analytic gradient.
        """
        if self.action_lag > 0:
            # simulate action delay
            self.action_buffer.append(actions.clone())
            actions = self.action_buffer.pop(0)

        self._raw_actions = actions.clone().detach()
        # get ground truth states
        cur_state_full = self.get_state_from_sim()

        # action preprocessing
        if self.cfg.sim2real_test:          # sim2real test mode, the inputs are a_zb and body-rates
            self.cmd_scaled = actions.clone().detach()
            self.cmd_scaled[:, 0] *= self._robot_mass       # convert a_zb to force
        else:   # training mode, the inputs are normalized actions
            actions_clipped = actions.tanh()
            self.cmd_scaled = actions_clipped * self.action_scale + self.action_offset
            self.cmd_scaled[:, 0] *= self.thr_est_error     # [NOTE] add noise to thrust due to thrust estimation error
        # reset self.force_body
        self.force_body = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # compute thrust and torque
        if self.command_type != "ThrustController":
            # self.controller: PSController | LVController | CTBRController
            thrusts_now, thrust_torque_now = self.controller.compute(cur_state_full, self.cmd_scaled)

            self._processed_actions = thrust_torque_now.clone().detach()
            self.torque_body[:,0,:] = self._processed_actions[:,1:4]
            self.force_body[:,0,2] = self._processed_actions[:,0]
            # simulate the differential dynamics
            self.nominal_next_state, self.a = self.drone_dynamics.step(thrust_torque_now)  

        else:  
            # [TODO] fix bug in ThrustController
            # self.controller:ThrustControlLoop
            assert (self.cmd_scaled[:, 0]>=0).all()
            thrusts_now = self.controller.update(self.cmd_scaled)

            self._processed_actions = thrusts_now.clone().detach()
            self.force_rotors[:, :, 2] =  self._processed_actions
            thrust_torque_now = thrusts_now @ self.controller.B_allocation_T
            self.torque_body[:,0,:] = thrust_torque_now[:,1:4].clone().detach()

            self.nominal_next_state, self.a = self.drone_dynamics.step(thrust_torque_now)
        # [NOTE] apply air drag
        self.force_body = self.force_body - (self.drone_dynamics.drag_coeffs * self.lin_vel_b * self.lin_vel_b.abs() + self.drone_dynamics.h_force_drag_coeffs * self.lin_vel_b)[:,None]
        # compose full actions vector
        self.force_all = torch.cat((self.force_rotors,self.force_body), dim = 1)
        self.torque_all = torch.cat((self.torque_rotors,self.torque_body), dim = 1)


    def apply_actions(self):    # [NOTE] called every simulation step
        self.robot.set_external_force_and_torque(self.force_all, self.torque_all, body_ids=self.rotor_ids + self.body_ids)

    def align_dynamics(self, ):
        """
        Align dynamics with issac-sim simulation. \n
        Mathmatics Description:
            `s_t = grad_decay_factor * ( s_t - s_t.detach() ) + _s_t `
        (s_t from dynamics model, _s_t from isaac_sim)
        """
        self.get_state_from_sim()   # update states after applying actions
        next_aligned_state = self.drone_dynamics.align(self.states_all, self.nominal_next_state)
        return next_aligned_state, self.nominal_next_state, self.a
        
    def reset_idx(self, env_ids):
        # [NOTE] This function should be called after environment reset
        # reset thrust controller
        self.controller.reset_idx(env_ids)
        # reset drone dynamics
        self.get_state_from_sim()
        self.drone_dynamics.reset_idx(env_ids)
        self.drone_dynamics.reset_state(self.states_all, env_ids)

        #
        self.thr_est_error[env_ids] = 1 + torch.randn(len(env_ids), device=self.device) * 0.01

    def reset(self, env_ids=None):
        if env_ids is None:
            self.reset_idx(torch.arange(self.num_envs, dtype=torch.int64, device=self.device))
        else:
            self.reset_idx(env_ids)
    
    def detach(self, ):
        # detach drone dynamics
        self.drone_dynamics.detach()
        # detach controller
        self.controller.detach()

    def _get_scale_factor(self, normal_range=(-1, 1), method="medium"):
        self.motor_omega = self.controller_cfg.motor_omega
        self.thrustmap = self.controller_cfg.thrustmap
        
        self.max_thrust_weight_ratio = self.cfg.max_thrust_weight_ratio
        
        self.max_rotor_thrust = self.thrustmap[0] * self.motor_omega[1] ** 2 + self.thrustmap[1] * self.motor_omega[1] + self.thrustmap[2]
        self.min_rotor_thrust = 0.0     # [NOTE] minimum thrust is 0

        # scale and offset for ThrustAndBodyRateLoop
        if self.command_type == "CTBRController":
            self.body_rate_bound = self.controller_cfg.body_rate_bound
            if method == "medium":
                body_rate_threshold = torch.ones(self.num_envs, 3, device=self.device) * self.body_rate_bound[1]
                self.action_scale = torch.hstack([ (self._robot_weight * self.max_thrust_weight_ratio / (normal_range[1] - normal_range[0]))[:,None], body_rate_threshold])  
                self.action_offset = torch.hstack([ (self._robot_weight * self.max_thrust_weight_ratio / (normal_range[1] - normal_range[0]))[:, None], torch.zeros(self.num_envs, 3, device=self.device)])
            
            elif method == "min_max":
                self.action_scale = torch.tensor(
                    [4 * (self.max_rotor_thrust - self.min_rotor_thrust) / (normal_range[1] - normal_range[0]), 
                    self.body_rate_bound[1], 
                    self.body_rate_bound[1], 
                    self.body_rate_bound[1]], device= self.device)[None].repeat(self.num_envs, 1)

                self.action_offset = torch.tensor([4 * (self.max_rotor_thrust + self.min_rotor_thrust) / (normal_range[1] - normal_range[0]), 0.0 ,0.0 ,0.0], device=self.device)
                self.action_offset = self.action_offset[None].repeat(self.num_envs, 1)
            else:
                raise ValueError("Invalid method")
        
        # scale for VelocityAndYawLoop
        elif self.command_type == "LVController":
            self.lin_vel_bound = self.cfg.lin_vel_bound
            self.action_scale = torch.tensor([3.1415926, self.lin_vel_bound[1], self.lin_vel_bound[1], self.lin_vel_bound[1]], device=self.device)[None].repeat(self.num_envs, 1)
            self.action_offset = torch.tensor([[0.0, 0.0 ,0.0 ,0.0]], device=self.device)[None].repeat(self.num_envs, 1)

        # scale for PositionAndYawLoop
        elif  self.command_type == "PSController":
            self.pos_bound = self.cfg.pos_bound
            self.action_scale = torch.tensor([3.1415926, self.pos_bound[1], self.pos_bound[1], self.pos_bound[1]], device=self.device)[None].repeat(self.num_envs, 1)
            self.action_offset = torch.tensor([[0.0, 0.0 ,0.0 ,0.0]], device=self.device)[None].repeat(self.num_envs, 1)
        
        # scale and offset for SingleRotorThrustLoop
        elif self.command_type == "ThrustController":
            if method == "medium":
                self.action_scale = self._robot_weight[:,None].repeat(1, 4) / 4
                self.action_offset = self._robot_weight[:,None].repeat(1, 4) / 4

            elif method == "min_max":
                self.action_scale = torch.ones(self.num_envs, 4, device=self.device) * (self.max_rotor_thrust - self.min_rotor_thrust) / (normal_range[1] - normal_range[0])
                self.action_offset = torch.ones(self.num_envs, 4, device=self.device) * (self.max_rotor_thrust + self.min_rotor_thrust) / (normal_range[1] - normal_range[0]),
                    
            else:
                raise ValueError("Invalid method")

        else:
            raise ValueError("Invalid command type")