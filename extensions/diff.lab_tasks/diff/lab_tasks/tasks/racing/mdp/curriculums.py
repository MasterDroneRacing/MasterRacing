# *******************************************************************************
# *                                                                             *
# *  Private and Confidential                                                   *
# *                                                                             *
# *  Unauthorized copying of this file, via any medium is strictly prohibited.  *
# *  Proprietary and confidential.                                              *
# *                                                                             *
# *  © 2024 OctiLab. All rights reserved.                                       *
# *                                                                             *
# *  Author: Yu Feng                                                            *
# *  Data: 2025/02/18     	             *
# *  Contact: yu-feng@sjtu.edu.cn                                               *
# *  Description: None                                                          *
# *******************************************************************************
from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def racing_terrain_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    cmd_name: str = "next_gate_pose",
    move_on_threshold: int = 3,
    move_down_threshold: int = 2,
) -> torch.Tensor:
    command_term = env.command_manager.get_term(cmd_name)
    accumulate_gates = command_term.metrics["accumulate_gates"]
    move_on = accumulate_gates[env_ids] >= move_on_threshold
    move_down = accumulate_gates[env_ids] < move_down_threshold
    terrain: TerrainImporter = env.scene.terrain    # type: ignore
    terrain.update_env_origins(env_ids, move_on, move_down)
    return torch.mean(terrain.terrain_levels.float())

def racing_cmd_noise_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    cmd_name: str = "next_gate_pose",
    enhance_threshold: int = 3,
    decay_threshold: int = 2,
    enhance_percent: float = 0.01,
    decay_percent: float = 0.01,
):
    command_term = env.command_manager.get_term(cmd_name)
    accumulate_gates = command_term.metrics["accumulate_gates"]
    enhanced = accumulate_gates[env_ids] >= enhance_threshold
    decayed = accumulate_gates[env_ids] < decay_threshold
    command_term.update_noise_level(env_ids, enhanced, decayed, enhance_percent, decay_percent)
    return command_term.noise_level.mean()
    