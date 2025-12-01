# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from typing import Literal
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

@configclass
class RslRlPpoVisionActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "VisionActorCritic"
    img_res: tuple[int, int] = (72, 96)
    dim_hidden_input: int = 192
    init_noise_std=1.0
    actor_hidden_dims=[128, 128]
    critic_hidden_dims=[128, 128]
    activation="lrelu"
    noise_std_type: Literal["scalar", "log"] = "scalar"




@configclass
class QuadcopterVisionPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 4000
    save_interval = 500
    experiment_name = "racing_ppo_l2c2_vision"
    run_name="baseline_test"
    empirical_normalization = False
    policy = RslRlPpoVisionActorCriticCfg()

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPOL2C2",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    # algorithm.__setattr__("grad_penalty_coef_schedule",[0.1, 0.1, 700, 1000])
    # algorithm.__setattr__("use_auxiliary_loss", True)
    policy.__setattr__("use_auxiliary_loss", True)


