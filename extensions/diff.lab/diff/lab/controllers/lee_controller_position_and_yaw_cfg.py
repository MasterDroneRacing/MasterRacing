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
# *  Data: 2025/03/06     	             *
# *  Contact: yu-feng@sjtu.edu.cn                                               *
# *  Description: None                                                          *
# *******************************************************************************
from isaaclab.utils import configclass
from .lee_controller_position_and_yaw import LeePositionAndYawController
from typing import Literal
from dataclasses import MISSING
import torch
@configclass
class LeePositionAndYawControllerCfg:
    class_type: type = LeePositionAndYawController

    # [NOTE]: the following default parameters are well-tuned in trajactory-tracking.
    k_p: list = [8.0, 8.0, 14.0]
    k_v: list = [4.0, 4.0, 5.0]
    k_q: list = [150.0, 150.0, 200.0]
    k_w: list = [15.0, 15.0, 20.0]

    max_fb_acc: float = 20.0
    mass: float = MISSING
    inertial: torch.Tensor = MISSING
    gravity_norm: float = 9.81


