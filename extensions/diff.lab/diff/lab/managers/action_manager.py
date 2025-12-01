# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action manager for processing actions sent to the environment."""

from __future__ import annotations

import inspect
import torch
import weakref
from abc import abstractmethod
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING

import omni.kit.app

from isaaclab.assets import AssetBase

from isaaclab.managers.manager_base import ManagerBase
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.managers.action_manager import ActionTerm, ActionManager

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

class DiffActionManager(ActionManager):
    
    def process_action(self, action: torch.Tensor):
        """Processes the actions sent to the environment.

        Note:
            This function should be called once per environment step.

        Args:
            action: The actions to process.
        """
        # check if action dimension is valid
        if self.total_action_dim != action.shape[1]:
            raise ValueError(f"Invalid action shape, expected: {self.total_action_dim}, received: {action.shape[1]}.")
        # store the input actions
        self._prev_action[:] = self._action
        self._action[:] = action.to(self.device).clone().detach()       # NOTE detach to prevent gradient tracking

        # split the actions and apply to each tensor
        idx = 0
        for term in self._terms.values():
            term_actions = action[:, idx : idx + term.action_dim]
            term.process_actions(term_actions)
            idx += term.action_dim

    