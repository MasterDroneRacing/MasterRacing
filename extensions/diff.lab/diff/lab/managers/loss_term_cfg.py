import torch
from isaaclab.utils import configclass
from isaaclab.managers.manager_term_cfg import ManagerTermBaseCfg
from collections.abc import Callable
from dataclasses import MISSING

@configclass
class LossTermCfg(ManagerTermBaseCfg):
    """Configuration for a loss term."""

    func: Callable[..., torch.Tensor] = MISSING
    """The name of the function to be called.

    This function should take the environment object and any other parameters
    as input and return the loss signals as torch float tensors of
    shape (num_envs,).
    """

    weight: float = MISSING
    """The weight of the loss term.

    This is multiplied with the loss term's value to compute the final
    loss.

    Note:
        If the weight is zero, the loss term is ignored.
    """

    use_diff_states: bool = True
    """ True for using differential states, False for not using.
    """

    use_action: bool = False
    """ True for using action, False for not using.

    Note:
        Only active when use_diff_states is True.
    """