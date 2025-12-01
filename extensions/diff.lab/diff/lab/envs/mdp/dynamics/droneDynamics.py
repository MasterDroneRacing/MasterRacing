import torch
import yaml
import os.path as osp

import isaaclab.utils.math as math_utils


class DroneDynamics():
    def __init__(self, num_envs, mass, inertia, dt, decimation, random_drag:bool=True, device:str = "cuda"):
        self.random_drag: bool = random_drag
        current_dir = osp.dirname(osp.abspath(__file__))
        controller_param_path = osp.join(current_dir,"dynamics.yaml")
        with open(controller_param_path, "r") as f:
            self.controller_params = yaml.safe_load(f)
        self.dt = dt
        self.decimation = decimation
        self.num_envs = num_envs
        self.device = device
        self.mass = mass
        self.inertia = inertia
        self.inertia_inv = torch.inverse(inertia)
        # Air drag coefficients
        self._drag_coeffs = torch.tensor(self.controller_params["drag_2_coeffs"], device=self.device)[None].repeat(self.num_envs, 1) * self.mass.unsqueeze(-1)
        # self._drag_coeffs = _drag_coeffs * 0.5 * 1.225 * torch.tensor([0.01, 0.01, 0.03], device=self.device)        # F_d = 1/2 * rho * v^2 * C_d * A
        self._h_force_drag_coeffs = torch.tensor(self.controller_params["drag_1_coeffs"], device=self.device)[None].repeat(self.num_envs, 1) * self.mass.unsqueeze(-1)
        self.drag_coeffs = self._drag_coeffs.clone()
        self.h_force_drag_coeffs = self._h_force_drag_coeffs.clone()
        self.drag_1_randomness = self.controller_params["drag_1_randomness"]
        self.drag_2_randomness = self.controller_params["drag_2_randomness"]
        self._z_drag_coeff = self.controller_params["z_drag_coeff"]
        self.z_drag_coeff = torch.ones(self.num_envs, device=self.device) * self._z_drag_coeff
        self.z_drag_randomness = self.controller_params["z_drag_randomness"]
        self.drag_coeffs[:, 2] = self.drag_coeffs[:, 2] * self._z_drag_coeff
        self.h_force_drag_coeffs[:, 2] = self.h_force_drag_coeffs[:, 2] * self._z_drag_coeff

        self.pos = torch.zeros(self.num_envs,3,device=self.device)
        self.quat = torch.zeros(self.num_envs,4,device=self.device)
        self.lin_vel_w = torch.zeros(self.num_envs,3,device=self.device)
        self.ang_vel_w = torch.zeros(self.num_envs,3,device=self.device)
        self.lin_vel_b = torch.zeros(self.num_envs,3,device=self.device)
        self.ang_vel_b = torch.zeros(self.num_envs,3,device=self.device)
        self.g = torch.tensor([0,0,-float(self.controller_params["g"])],device = self.device)
        self.z_b = torch.tensor([[0.0, 0.0, 1.0]],device = self.device)
        self.omega2quat = torch.tensor([[0.0, 1.0, 0.0, 0.0],
                                    [0.0, 0.0, 1.0, 0.0],
                                    [0.0, 0.0, 0.0, 1.0],],device=self.device)
        
        self.decay_factor = self.controller_params["grad_decay_factor"]

    def reset_idx(self, idx):
        # reset the drag coeffs
        if self.random_drag:
            self.z_drag_coeff[idx] = torch.ones(len(idx), device=self.device) * self._z_drag_coeff + torch.rand(len(idx), device=self.device) * self.z_drag_randomness
            self.drag_coeffs[idx] = self._drag_coeffs[idx] + torch.rand(len(idx), 3, device=self.device) * self.drag_2_randomness
            self.drag_coeffs[idx, 2] = self.drag_coeffs[idx, 2] * self.z_drag_coeff[idx]
            self.h_force_drag_coeffs[idx] = self._h_force_drag_coeffs[idx] + torch.rand(len(idx), 3, device=self.device) * self.drag_1_randomness
            self.h_force_drag_coeffs[idx, 2] = self.h_force_drag_coeffs[idx, 2] * self.z_drag_coeff[idx]

        if len(idx) == self.num_envs:
            self.pos = self.pos.detach()
            self.quat = self.quat.detach()
            self.lin_vel_w = self.lin_vel_w.detach()
            self.ang_vel_w = self.ang_vel_w.detach()
            self.lin_vel_b = self.lin_vel_b.detach()
            self.ang_vel_b = self.ang_vel_b.detach()
            return
        pos = self.pos.clone()
        pos[idx] = pos[idx].detach()
        self.pos = pos

        quat = self.quat.clone()
        quat[idx] = quat[idx].detach()
        self.quat = quat

        lin_vel_w = self.lin_vel_w.clone()
        lin_vel_w[idx] = lin_vel_w[idx].detach()
        self.lin_vel_w = lin_vel_w

        ang_vel_w = self.ang_vel_w.clone()
        ang_vel_w[idx] = ang_vel_w[idx].detach()
        self.ang_vel_w = ang_vel_w

        lin_vel_b = self.lin_vel_b.clone()
        lin_vel_b[idx] = lin_vel_b[idx].detach()
        self.lin_vel_b = lin_vel_b

        ang_vel_b = self.ang_vel_b.clone()
        ang_vel_b[idx] = ang_vel_b[idx].detach()
        self.ang_vel_b = ang_vel_b

    def detach(self, ):
        self.pos = self.pos.clone().detach()
        self.quat = self.quat.clone().detach()
        self.lin_vel_w = self.lin_vel_w.clone().detach()
        self.ang_vel_w = self.ang_vel_w.clone().detach()
        self.lin_vel_b = self.lin_vel_b.clone().detach()
        self.ang_vel_b = self.ang_vel_b.clone().detach()

    @property
    def state(self):
        return torch.hstack([self.pos, self.quat, self.lin_vel_w, self.ang_vel_w])
    
    def set_state(self, states):
        self.pos = states["pos"]
        self.quat = states["quat"]
        self.lin_vel_w = states["lin_vel_w"]
        self.ang_vel_w = states["ang_vel_w"]
        self.ang_vel_b = states["ang_vel_b"]
        self.lin_vel_b = states["lin_vel_b"]
    
    def reset_state(self, states, idx):
        self.pos[idx] = states[idx,:3]
        self.quat[idx] = states[idx,3:7]
        self.lin_vel_w[idx] = states[idx,7:10]
        self.ang_vel_w[idx] = states[idx,10:13]
        self.ang_vel_b[idx] = math_utils.quat_apply_inverse(self.quat[idx], self.ang_vel_w[idx])
        self.lin_vel_b[idx] = math_utils.quat_apply_inverse(self.quat[idx], self.lin_vel_w[idx])

    def step(self, thrust_torque_now):
        # steping by policy dt
        thrust =  thrust_torque_now[:,:1] @ self.z_b
        torque = thrust_torque_now[:,1:4]
        # [NOTE] consider air drag
        thrust = thrust - self.drag_coeffs * self.lin_vel_b * self.lin_vel_b.abs()  - self.h_force_drag_coeffs * self.lin_vel_b
        thrust_w = math_utils.quat_apply(self.quat, thrust)
        a = self.g + thrust_w / self.mass[:, None]
        alpha = (self.inertia_inv @ torque.unsqueeze(-1)).squeeze(-1) - (self.inertia_inv @ torch.linalg.cross(self.ang_vel_b, (self.inertia @ self.ang_vel_b.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1)).squeeze(-1)
        self.pos = self.pos + self.lin_vel_w * self.dt + 0.5 * a * self.dt * self.dt
        self.quat = self.quat + 0.5 * math_utils.quat_mul(self.quat, self.ang_vel_b @ self.omega2quat) * self.dt
        self.quat = self.quat / torch.norm(self.quat, dim = 1, keepdim= True)
        self.lin_vel_w = self.lin_vel_w + a * self.dt
        self.lin_vel_b = math_utils.quat_apply_inverse(self.quat, self.lin_vel_w)
        self.ang_vel_b = self.ang_vel_b + alpha * self.dt
        self.ang_vel_w = math_utils.quat_apply(self.quat, self.ang_vel_b)
        return torch.hstack([self.pos, self.quat, self.lin_vel_w, self.ang_vel_w]), a

    def step_subtle(self, thrust_torque_now):
        # steping by sim dt
        dt = self.dt / self.decimation
        for _ in range(self.decimation):
            thrust =  thrust_torque_now[:,:1] @ self.z_b
            torque = thrust_torque_now[:,1:4]
            thrust_w = math_utils.quat_apply(self.quat, thrust)
            a = self.g + thrust_w / self.mass[:, None]
            alpha = (self.inertia_inv @ torque.unsqueeze(-1)).squeeze(-1) - (self.inertia_inv @ torch.linalg.cross(self.ang_vel_b, (self.inertia @ self.ang_vel_b.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1)).squeeze(-1)
            self.pos = self.pos + self.lin_vel_w * dt + 0.5 * a * dt * dt
            self.quat = self.quat + 0.5 * math_utils.quat_mul(self.quat, self.ang_vel_b @ self.omega2quat) * dt
            self.quat = self.quat / torch.norm(self.quat, dim = 1, keepdim= True)
            self.lin_vel_w = self.lin_vel_w + a * dt
            self.lin_vel_b = math_utils.quat_apply_inverse(self.quat, self.lin_vel_w)
            self.ang_vel_b = self.ang_vel_b + alpha * dt
            self.ang_vel_w = math_utils.quat_apply(self.quat, self.ang_vel_b)

        return torch.hstack([self.pos, self.quat, self.lin_vel_w, self.ang_vel_w]), a

    def align(self, ref_state, nominal_state):
        # long horizon version ([NOTE] multi-steps gradients)
        # return self.decay_factor * (nominal_state - nominal_state.detach()) + ref_state
        # short horizon version ([NOTE] one-step gradients)
        ref_pos = ref_state[:,:3]
        ref_quat = ref_state[:,3:7]
        ref_lin_vel_w = ref_state[:,7:10]
        ref_lin_vel_b = math_utils.quat_apply_inverse(ref_quat, ref_lin_vel_w)
        ref_ang_vel_w = ref_state[:,10:13]
        ref_ang_vel_b = math_utils.quat_apply_inverse(ref_quat, ref_ang_vel_w)

        nominal_pos = nominal_state[:,:3]
        nominal_quat = nominal_state[:,3:7]
        nominal_lin_vel_w = nominal_state[:,7:10]
        nominal_lin_vel_b = math_utils.quat_apply_inverse(nominal_quat, nominal_lin_vel_w)
        nominal_ang_vel_w = nominal_state[:,10:13]
        nominal_ang_vel_b = math_utils.quat_apply_inverse(nominal_quat, nominal_ang_vel_w)

        self.pos = self.decay_factor * (nominal_pos - nominal_pos.detach()) + ref_pos
        self.quat = self.decay_factor * (nominal_quat - nominal_quat.detach()) + ref_quat
        self.lin_vel_w = self.decay_factor * (nominal_lin_vel_w - nominal_lin_vel_w.detach()) + ref_lin_vel_w
        self.ang_vel_w = self.decay_factor * (nominal_ang_vel_w - nominal_ang_vel_w.detach()) + ref_ang_vel_w
        self.lin_vel_b = self.decay_factor * (nominal_lin_vel_b - nominal_lin_vel_b.detach()) + ref_lin_vel_b
        self.ang_vel_b = self.decay_factor * (nominal_ang_vel_b - nominal_ang_vel_b.detach()) + ref_ang_vel_b

        return torch.hstack([self.pos, self.quat, self.lin_vel_w, self.ang_vel_w])