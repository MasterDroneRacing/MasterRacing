from dataclasses import MISSING
from typing import Literal
from collections.abc import Callable
import diff.lab.terrains.trimesh.racing_terrains as racing_terrains
from isaaclab.utils import configclass
from isaaclab.terrains import SubTerrainBaseCfg
import numpy as np
############################## Configuration for Racing Terrain ##############################
@configclass
class SquareRacingTrackTerrainCfg(SubTerrainBaseCfg):
    function : Callable = racing_terrains.SquareRacingTrackTerrain

    radius: list[float] = [2.0, 5.0]           # meters

    num_gate: int = 4
    gate_size: list[float] = [0.4, 0.5]           # meters
    gate_thickness: list[float] = [0.04, 0.08]      # meters
    pos_noise_scale: list[float] = [0.1, 1.0]       # meters
    rot_noise_scale: list[float] = [0.0, 30.0]       # degrees
    only_yaw: bool = True                    # only yaw rotation

    add_obs: bool = False
    add_border: bool = False
    add_ground_obs: bool = False

    num_ground_obs: list[float] = [1, 4]
    num_wall_seg: list[float] = [1, 5]
    wall_size: list[float] = [0.4, 1.0]           # meters
    wall_thickness: list[float] = [0.04, 0.08]      # meters
    num_orbit_seg: list[float] = [1, 5]

    # obstacle dense control parameters, the smaller the value, the denser the obstacles, gathering to the middle points of two adjacent gates
    adj_dir_shift_prop : list[float] = [0.2, 0.5]
    radius_dir_shift_prop : list[float] = [0.2, 0.5]


@configclass 
class ZigzagRacingTerrainCfg(SubTerrainBaseCfg):
    function : Callable = racing_terrains.ZigzagRacingTerrain
    
    track_length = 20.0
    num_gate = 7
    gate_size: list[float] = [0.4, 0.5]           # meters
    gate_thickness: list[float] = [0.04, 0.08]      # meters
    pos_noise_scale: list[float] = [0.1, 1.0]       # meters
    pos_z_noise_scale: list[float] = [0.1, 1.0]       # meters
    rot_noise_scale: list[float] = [0.0, 30.0]       # degrees
    only_yaw: bool = True                    # only yaw rotation

    add_obs: bool = False
    add_border: bool = False
    add_ground_obs: bool = False

    num_ground_obs: list[float] = [1, 4]
    num_wall_seg: list[float] = [1, 5]
    wall_size: list[float] = [0.4, 1.0]           # meters
    wall_thickness: list[float] = [0.04, 0.08]      # meters
    num_orbit_seg: list[float] = [1, 5]

    # obstacle dense control parameters, the smaller the value, the denser the obstacles, gathering to the middle points of two adjacent gates
    adj_dir_shift_prop : list[float] = [1, 2]
    radius_dir_shift_prop : list[float] = [10, 10]

    no_obs_range = 0.8

@configclass 
class EllipseRacingTerrainCfg(SubTerrainBaseCfg):
    function : Callable = racing_terrains.EllipseRacingTerrain
    
    gate_distance = 5.0 # distance between last and next gate
    num_gate:int = 8
    gate_size: list[float] = [0.4, 0.5]           # meters
    gate_thickness: list[float] = [0.04, 0.08]      # meters
    pos_noise_scale: list[float] = [0.1, 1.0]       # meters
    rot_noise_scale: list[float] = [0.0, 30.0]       # degrees
    only_yaw: bool = True                    # only yaw rotation

    add_obs: bool = False
    add_border: bool = False
    add_ground_obs: bool = False

    short_axis_prop: list[float] = [1.414, 0.8]
    long_axis_prop: list[float] = [3.1414, 4.8]
    num_ground_obs: list[float] = [1, 4]
    num_wall_seg: list[float] = [1, 5]
    wall_size: list[float] = [0.4, 1.0]           # meters
    wall_thickness: list[float] = [0.04, 0.08]      # meters
    num_orbit_seg: list[float] = [1, 5]

    # obstacle dense control parameters, the smaller the value, the denser the obstacles, gathering to the middle points of two adjacent gates
    adj_dir_shift_prop : list[float] = [1, 2]
    radius_dir_shift_prop : list[float] = [10, 10]

    no_obs_range = 0.8