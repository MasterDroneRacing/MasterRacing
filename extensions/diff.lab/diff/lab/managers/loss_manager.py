# *******************************************************************************
# *                                                                             *
# *  Private and Confidential                                                   *
# *                                                                             *
# *  Unauthorized copying of this file, via any medium is strictly prohibited.  *
# *  Proprietary and confidential.                                              *
# *                                                                             *
# *  © 2025 DiffLab. All rights reserved.                                       *
# *                                                                             *
# *  Author: Yu Feng                                                            *
# *  Data: 2025/02/16     	             *
# *  Contact: yu-feng@sjtu.edu.cn                                               *
# *  Description: None                                                          *
# *******************************************************************************
"""Loss manager for computing differential loss term for a given world"""
from __future__ import annotations

import torch
from prettytable import PrettyTable
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import ManagerBase, ManagerTermBase
from .loss_term_cfg import LossTermCfg

if TYPE_CHECKING:
    from diff.lab.envs import ManagerBasedDiffRLEnv

class LossManager(ManagerBase):
    _env: ManagerBasedDiffRLEnv

    def __init__(self, cfg: object, env: ManagerBasedDiffRLEnv):
        # create buffers to parse and store terms
        self._term_names: list[str] = list()
        self._term_cfgs: list[LossTermCfg] = list()
        self._class_term_cfgs: list[LossTermCfg] = list()

        super().__init__(cfg, env)

        self._episode_sums = dict()
        for term_name in self._term_names:
            self._episode_sums[term_name] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._loss_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)  # differential loss
        self._loss_buf_detach = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)  # non-differential loss
        self._step_loss = torch.zeros((self.num_envs, len(self._term_names)), dtype=torch.float, device=self.device)

    def __str__(self,) -> str:
        msg = f"<LossManager> contains {len(self._term_names)} active terms.\n"

        # create table for term information
        table = PrettyTable()
        table.title = "Active Loss Terms"
        table.field_names = ["Index", "Name", "Weight"]
        # set alignment of table columns
        table.align["Name"] = "l"
        table.align["Weight"] = "r"
        # add info on each term
        for index, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            table.add_row([index, name, term_cfg.weight])
        # convert table to string
        msg += table.get_string()
        msg += "\n"

        return msg
    
    @property
    def active_terms(self) -> list[str]:
        """Name of active loss terms."""
        return self._term_names
    
    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        extras = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Loss/" + key] = episodic_sum_avg / self._env.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        for term_cfg in self._class_term_cfgs:
            term_cfg.func.reset(env_ids=env_ids)

        return extras

    def compute(self, dt:float, aligned_states:torch.Tensor, action:torch.Tensor):
        self._loss_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._loss_buf_detach = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for name, term_cfg in zip(self._term_names, self._term_cfgs):
            if term_cfg.weight == 0.0:
                continue
            if term_cfg.use_diff_states:        # differential states [NOTE] ignore dt
                if not term_cfg.use_action:
                    value = term_cfg.func(self._env, aligned_states, **term_cfg.params) * term_cfg.weight   
                else:
                    value = term_cfg.func(self._env, aligned_states, action, **term_cfg.params) * term_cfg.weight
                self._step_loss[:, self._term_names.index(name)] = value.clone().detach()
                self._loss_buf += value
            else:                            # non-differential states [NOTE] ignore dt
                value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight
                self._step_loss[:, self._term_names.index(name)] = value
                self._loss_buf_detach += value

            self._episode_sums[name] += (value * dt).clone().detach()
            
        return self._loss_buf, self._loss_buf_detach

    def set_term_cfg(self, term_name: str, cfg: LossTermCfg):
        if term_name not in self._term_names:
            raise ValueError(f"Loss term '{term_name}' not found.")
        self._term_cfgs[self._term_names.index(term_name)] = cfg

    def get_term_cfg(self, term_name: str)->LossTermCfg:
        if term_name not in self._term_names:
            raise ValueError(f"Loss term '{term_name}' not found.")
        return self._term_cfgs[self._term_names.index(term_name)]
    
    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        terms = []
        for idx, name in enumerate(self._term_names):
            terms.append((name, [self._step_loss[env_idx, idx].cpu().item()]))
        return terms
    
    def log_all_active_terms(self, ):
        terms = []
        for idx, name in enumerate(self._term_names):
            terms.append((name, self._step_loss[:, idx].mean().cpu().item()))
        return terms
    
    def _prepare_terms(self):
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        for term_name, term_cfg in cfg_items:
            if term_cfg is None:
                continue
            if not isinstance(term_cfg, LossTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type LossTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            if not isinstance(term_cfg.weight, (float, int)):
                raise TypeError(
                    f"Weight for the term '{term_name}' is not of type float or int."
                    f" Received: '{type(term_cfg.weight)}'."
                )
            if not isinstance(term_cfg.use_diff_states, bool):
                raise TypeError(
                    f"Use_diff_states for the term '{term_name}' is not of type bool."
                    f" Received: '{type(term_cfg.use_diff_states)}'."
                )
            if term_cfg.use_diff_states:
                if not term_cfg.use_action:
                    self._resolve_common_term_cfg(term_name, term_cfg, min_argc=2)
                else:
                    self._resolve_common_term_cfg(term_name, term_cfg, min_argc=3)
            else:
                self._resolve_common_term_cfg(term_name, term_cfg, min_argc=1)

            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)
            if isinstance(term_cfg.func, ManagerTermBase):
                self._class_term_cfgs.append(term_cfg)