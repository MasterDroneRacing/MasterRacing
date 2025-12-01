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
# *  Data: 2025/03/06                                                           *
# *  Contact: yu-feng@sjtu.edu.cn                                               *
# *  Description: Configuration classes for controllers                         *
# *******************************************************************************
from dataclasses import MISSING
from isaaclab.utils import configclass
from typing import List, Tuple
import torch
from .controller_diff import CTBRController, LVController, PSController

@configclass
class ControllerBaseCfg:
    class_type: type = MISSING
    # Physical parameters
    arm_length: float = 0.09  # Arm length of the quadrotor
    kappa: float = 0.016  # Thrust to torque ratio
    motor_tau: float = 0.0001   # Motor delay time constant
    motor_omega: Tuple[float, float] = (150, 3000)  # Motor speed range (min, max)
    thrustmap: List[float] = [
        1.3298253500372892e-06, 
        0.0038360810526746033, 
        -1.7689986848125325
        ]  # Thrust mapping coefficients
    g: float = 9.81  # Gravity constant

    # [NOTE] Whether simulates motor model
    # Recommended to set to FALSE due to some bugs on thrust-controller
    use_motor_model: bool = False  # Use motor model for thrust calculation
    # only applied when use_motor_model is FALSE
    thrust_ctrl_delay: float = 0.03  # Thrust control delay
    torque_ctrl_delay: Tuple[float, float, float] = (0.02, 0.02, 0.02)  # Torque control delay

@configclass
class CTBRControllerCfg(ControllerBaseCfg):
    class_type: type = CTBRController
    
    # Physical parameters
    body_rate_bound: List[float] = [-12.0, 12.0] # Body rate bounds

    # Controller parameters
    # [NOTE] All gains are obtained when use_motor_model is FALSE!!!!
    rate_gain_p: List[float] = [50.0, 50.0, 50.0]  # Rate control gains (X, Y, Z)
    rate_gain_i: List[float] = [0.0, 0.0, 0.0]  # Rate control integral gains (X, Y, Z)
    rate_gain_d: List[float] = [0.0, 0.0, 0.0]  # Rate control derivative gains (X, Y, Z)

@configclass
class LVControllerCfg(ControllerBaseCfg):
    class_type: type = LVController

    # Physical parameters
    max_feedback_accel: float = 20.0  # Maximum thrust acceleration
    body_rate_bound: List[float] = [-12.0, 12.0] # Body rate bounds

    # Controller parameters
    # [NOTE] All gains are obtained when use_motor_model is FALSE!!!!
    speed_gain: List[float] = [10.0, 10.0, 20.0]  # Speed control gain
    pose_gain: List[float] = [18.0, 18.0, 20.0]  # Pose control gains 
    rate_gain: List[float] = [180.0, 180.0, 200.0]  # Rate control gains 

@configclass
class PSControllerCfg(ControllerBaseCfg):
    class_type: type = PSController
    
    # Physical parameters
    max_feedback_accel: float = 20.0  # Maximum thrust acceleration
    body_rate_bound: List[float] = [-12.0, 12.0] # Body rate bounds

    # Controller parameters
    # [NOTE] All gains are obtained when use_motor_model is FALSE!!!!!!
    speed_gain: List[float] = [5.0, 5.0, 5.0]  # Speed control gain
    pose_gain: List[float] = [20.0, 20.0, 20.0]  # Pose control gains 
    rate_gain: List[float] = [150.0, 150.0, 150.0]  # Rate control gains 
    pos_gain: List[float] = [3.0, 3.0, 3.0]  # Position control gains
    
