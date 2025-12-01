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
# *  Data: 2025/03/11     	             *
# *  Contact: yu-feng@sjtu.edu.cn                                               *
# *  Description: None                                                          *
# *******************************************************************************

from __future__ import annotations

import torch
from torch import nn
from standalone.rsl_rl.ext.modules.actor_critic_recurrent import ActorCritic, ActorCriticRecurrent, Memory
# from rsl_rl.modules.actor_critic_recurrent import ActorCriticRecurrent, Memory
from standalone.rsl_rl.ext.utils.utils import unpad_trajectories
from torch.distributions import Normal

def resolve_nn_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None

class VisionActorCritic(ActorCritic):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        img_res: tuple[int, int], # resolution of the image
        dim_hidden_input: int,         # dimension of rnn input
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs
    ):
        self.use_auxiliary_loss = kwargs.get("use_auxiliary_loss", False)
        if self.use_auxiliary_loss:
            print("VisionActorCritic is working with auxiliary observations, currently only support auxiliary observations with a single dimension.")
            kwargs.pop("use_auxiliary_loss", None)

        if kwargs:
            print(
                "VisionActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),
            )
        
        super().__init__(
            num_actor_obs=dim_hidden_input,
            num_critic_obs=dim_hidden_input,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        )
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Invalid noise_std_type: {self.noise_std_type}")

        self.activation = resolve_nn_activation(activation)

        self.img_res = img_res

        # Depth: height: 72, width: 96
        assert img_res[0] == 72 and img_res[1] == 96, f"Image dimension should be 72x96, got {img_res}"
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, 3, bias=False),  # 24x32x16
            nn.BatchNorm2d(16),
            self.activation,
            nn.Conv2d(16, 32, 3, 3, bias=False),    # 8x10x32
            nn.BatchNorm2d(32),
            self.activation,
            nn.Conv2d(32, 64, 2, 2, bias=False),    # 4x5x64
            nn.BatchNorm2d(64),
            self.activation,
            nn.Flatten(),   # 1280
            nn.Linear(1280, dim_hidden_input),    # 128
        )

        # state: num_actor_state_obs
        self.state_enc = nn.Linear(num_actor_obs - img_res[0] * img_res[1], dim_hidden_input)

        if self.use_auxiliary_loss:
            # auxiliary decoder
            self.aux_decoder = nn.Linear(dim_hidden_input, 1)

    
    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def update_distribution(self, observations):
        img = observations[:, -self.img_res[0] * self.img_res[1]:].view(-1, 1, self.img_res[0], self.img_res[1])
        state = observations[:, :-self.img_res[0] * self.img_res[1]]
        feat = self.activation(self.stem(img) + self.state_enc(state))
        mean = self.actor(feat)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Invalid noise_std_type: {self.noise_std_type}")
        
        self.distribution = Normal(mean, std)

    def act_inference(self, observations):
        img = observations[:, -self.img_res[0] * self.img_res[1]:].view(-1, 1, self.img_res[0], self.img_res[1])
        state = observations[:, :-self.img_res[0] * self.img_res[1]]
        feat = self.activation(self.stem(img) + self.state_enc(state))
        mean = self.actor(feat)
        return mean, feat
    
    def evaluate(self, critic_obs, **kwargs):
        img = critic_obs[:, -self.img_res[0] * self.img_res[1]:].view(-1, 1, self.img_res[0], self.img_res[1])
        state = critic_obs[:, :-self.img_res[0] * self.img_res[1]]
        feat = self.activation(self.stem(img) + self.state_enc(state))
        return self.critic(feat)
        