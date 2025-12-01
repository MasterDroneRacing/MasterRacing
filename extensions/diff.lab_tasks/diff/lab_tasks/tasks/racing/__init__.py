# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Quacopter environment.
"""

import gymnasium as gym

from . import agents

from .racing_ctbr_env import QuadcopterRacingCTBREnvCfg
##
# Register Gym environments.
##

gym.register(
    id="DiffLab-Quadcopter-CTBR-Racing-v0",
    entry_point="diff.lab.envs:ManagerBasedDiffRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterRacingCTBREnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadcopterVisionPPORunnerCfg",
    },

)
