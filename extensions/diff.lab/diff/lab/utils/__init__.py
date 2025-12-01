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
# *  Data: 2025/03/05     	             *
# *  Contact: yu-feng@sjtu.edu.cn                                               *
# *  Description: None                                                          *
# *******************************************************************************
import torch
from .mesh_tools import *
from .visualization import *
# space lattice
LATTICE_TENSOR = torch.tensor([
    [ 0.0,  0.0,  0.0],
    [ 1.0,  1.0,  1.0],
    [ 1.0, -1.0,  1.0],
    [-1.0,  1.0,  1.0],
    [-1.0, -1.0,  1.0],
    [ 1.0,  1.0, -1.0],
    [ 1.0, -1.0, -1.0],
    [-1.0,  1.0, -1.0],
    [-1.0, -1.0, -1.0],
    [ 0.5,  0.5,  0.5],
    [ 0.5, -0.5,  0.5],
    [-0.5,  0.5,  0.5],
    [-0.5, -0.5,  0.5],
    [ 0.5,  0.5, -0.5],
    [ 0.5, -0.5, -0.5],
    [-0.5,  0.5, -0.5],
    [-0.5, -0.5, -0.5],
])