# *******************************************************************************
# *                                                                             *
# *  Private and Confidential                                                   *
# *                                                                             *
# *  Unauthorized copying of this file, via any medium is strictly prohibited.  *
# *  Proprietary and confidential.                                              *
# *                                                                             *
# *  © 2024 DiffLab. All rights reserved.                                       *
# *                                                                             *
# *  Author: Yu Feng, Su Yang                                                   *
# *  Data: 2025/03/10     	             *
# *  Contact: yu-feng@sjtu.edu.cn                                               *
# *  Description: None                                                          *
# *******************************************************************************
import torch

class ThrustController(): 
    '''
    simulate thrust motor model. \n
    Transform four given expected propeller force into motor speed, and simulate the step process to produce a delayed real force response. \n
    Mathmatics Description:
        `f_i = k2 * w^2 + k1 * w + k0`
        `w_t+1 = e^(-dt/tau) * w_t + (1 - e^(-dt/tau)) * w_des `
        (w means motor speed, f_i means single propeller force)
        (k2, k1, k0, dt, tau need to be calibrated.)
    '''
    
    def __init__(self, device, num_envs, params):
        self.device = device
        self.num_envs = num_envs
        self.motor_tau = float(params["motor_tau"])
        self.dt = float(params["dt"])
        self._motor_tau_inv = torch.tensor(1 / self.motor_tau,device = device)
        self._c = torch.exp(-self._motor_tau_inv * self.dt)
        self.thrustmap = (
            float(params["thrustmap"][0]),
            float(params["thrustmap"][1]),
            float(params["thrustmap"][2])
        )
        self.arm_length = float(params["arm_length"])
        self.kappa = float(params["kappa"]) # c_M / c_T = D_p * C_M / C_T
        self.t_BM_ = (
            self.arm_length
            * torch.tensor(0.5).sqrt()
            * torch.tensor([
                [1, -1, -1, 1],
                [-1, -1, 1, 1],
            ])
        )
        self.B_allocation = torch.vstack(
            [torch.ones(1, 4), self.t_BM_, self.kappa * torch.tensor([1, -1, 1, -1])]
        ).to(self.device)
        self.B_allocation_inv = torch.inverse(self.B_allocation)
        self.B_allocation_T = self.B_allocation.transpose(0,1)
        self.B_allocation_inv_T = self.B_allocation_inv.transpose(0,1)

        self.omega_realtime = torch.zeros([num_envs, 4],device=self.device)
    
    @property
    def motor_omegas(self,):
        return self.omega_realtime
    
    @property
    def motor_thrusts(self,):
        return self.Omega2Thrust(self.motor_omegas)
    
    def reset_idx(self, envs_id):
        if len(envs_id) == self.num_envs:
            self.omega_realtime = self.omega_realtime.detach()
            self.omega_realtime.zero_()
            return
        clone = self.omega_realtime.clone()
        clone[envs_id] = clone[envs_id].detach()
        clone[envs_id] = torch.zeros_like(clone[envs_id])
        self.omega_realtime = clone
        
    def reset(self,):
        self.reset_idx(torch.arange(self.num_envs, dtype=torch.long, device=self.device))

    def detach(self, ):
        self.omega_realtime = self.omega_realtime.clone().detach()

    def Thrust2Omega(self, thrusts):
        scale = 1 / (2 * self.thrustmap[0])
        omega = scale * (
                -self.thrustmap[1]
                + torch.sqrt(
                    self.thrustmap[1] ** 2
                    - 4 * self.thrustmap[0] * (self.thrustmap[2] - thrusts)
                )
        )
        return omega

    def Omega2Thrust(self, omega):
        return self.thrustmap[0] * omega * omega + self.thrustmap[1] * omega + self.thrustmap[2]

    # from desired thrusts update realtime thrusts
    def update(self, thrusts_des):
        omega_des = self.Thrust2Omega(thrusts_des)
        self.omega_realtime = self._c * self.omega_realtime + (1 - self._c) * omega_des
        thrusts_realtime = self.Omega2Thrust(self.omega_realtime)
        return thrusts_realtime
         