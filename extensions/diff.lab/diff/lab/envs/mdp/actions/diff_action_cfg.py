# *******************************************************************************
# *                                                                             *
# *  Private and Confidential                                                   *
# *                                                                             *
# *  Unauthorized copying of this file, via any medium is strictly prohibited.  *
# *  Proprietary and confidential.                                              *
# *                                                                             *
# *  © 2024 OctiLab. All rights reserved.                                       *
# *                                                                             *
# *  Author: Su Yang                                                            *
# *  Data: 2025/01/19     	                                                    *
# *  Contact:                                                                   *
# *  Description: None                                                          *
# *******************************************************************************
from isaaclab.utils import configclass
from isaaclab.managers import ActionTermCfg, ActionTerm
from .diff_action import DiffActions
from dataclasses import MISSING
from diff.lab.controllers.controller_diff_cfg import CTBRControllerCfg, LVControllerCfg, PSControllerCfg, ControllerBaseCfg
from typing import Tuple
@configclass
class DiffActionCfg(ActionTermCfg):
    """Configuration for the joint position action term.

    See :class:`JointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = DiffActions
    asset_name: str = "robot"
    rotor_names: str = "m.*_prop"
    # Currently only support `ThrustAndBodyRateLoop`(CTBRController), `VelocityAndYawLoop`(LVController), `PositionAndYawLoop`(PSController), `SingleRotorThrustLoop`(ThrustController) 
    command_type: str = "CTBRController"

    # Controller parameters
    controller_cfg: ControllerBaseCfg = MISSING
     
    gravity: float = 9.81
    random_drag: bool = True    # whether to randomize the drag coefficients
    action_lag: int = 1         # action lag in simulation steps, used to simulate the delay of the controller

    # add sim2real test mode
    sim2real_test: bool = False

    # action rescaling
    pos_bound: Tuple[float, float] = (-1.0, 1.0)            # max position in meters, when using PSController
    lin_vel_bound: Tuple[float, float] = (-5.0, 5.0)        # max velocity in m/s, when using LVController
    max_thrust_weight_ratio: float = 3.0  # Maximum thrust to weight ratio, when using CTBRController
