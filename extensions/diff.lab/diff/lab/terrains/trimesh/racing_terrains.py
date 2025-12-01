from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import racing_terrains_cfg
import numpy as np
import scipy.spatial.transform as tf
from isaaclab.terrains.trimesh.utils import make_border
from .utils import *
import random


def SquareRacingTrackTerrain(difficulty: float, cfg: racing_terrains_cfg.SquareRacingTrackTerrainCfg):
    """
    Generate a square racing track with gates, walls and orbits.
    Args:
        cfg: Configuration for the square racing track.
        difficulty: Difficulty of the racing track.

    Returns:
        A dictionary containing the generated terrain.
    """
    meshes_list = list()
    # parse the configuration
    radius = random.uniform(cfg.radius[0], cfg.radius[1])     # radius of the circular track
    num_gate = cfg.num_gate
    only_yaw = cfg.only_yaw

    # STEP 0: generate the border
    if cfg.add_border:
        border_width = 0.5
        border_center = (cfg.size[0] / 2, cfg.size[1] / 2, 0.0)
        border_inner_size= (cfg.size[0] - 2 * border_width, cfg.size[1] - 2 * border_width)
        meshes_list += make_border(cfg.size, border_inner_size, 6.0, border_center)

    # STEP 1: generate the gate firstly
    gate_size = cfg.gate_size[1] - (cfg.gate_size[1] - cfg.gate_size[0]) * difficulty
    gate_thickness = cfg.gate_thickness[0] + (cfg.gate_thickness[1] - cfg.gate_thickness[0]) * difficulty
    gate_pos_noise_scale = difficulty * (cfg.pos_noise_scale[1] - cfg.pos_noise_scale[0]) + cfg.pos_noise_scale[0]
    rot_noise_scale = difficulty * (cfg.rot_noise_scale[1] - cfg.rot_noise_scale[0]) + cfg.rot_noise_scale[0]
    
    # generate the gates pose
    theta = np.linspace(0, 2 * np.pi, num_gate, endpoint=False)
    gate_pts = np.zeros((num_gate, 3),dtype=np.float32)
    gate_pts[:,0] = np.cos(theta) * radius
    gate_pts[:,1] = np.sin(theta) * radius
    gate_pts[:,2] = 1.0     # the height of the gate above the ground

    gate_pts[:,0] += cfg.size[0] / 2
    gate_pts[:,1] += cfg.size[1] / 2
    gate_pts[:,2] += 0

    gate_euler = np.zeros((num_gate,3),dtype=np.float32)
    gate_euler[:,0] = 90.0      # roll
    gate_euler[:,1] = theta / np.pi * 180.0  # pitch

    # add noise to the gate pose
    gate_pos_noise = np.random.uniform(-1, 1, (num_gate, 3)) * gate_pos_noise_scale
    gate_rot_noise = np.random.uniform(-1, 1, (num_gate, 3)) * rot_noise_scale
    if only_yaw:
        gate_rot_noise[:,0] = 0.0
        gate_rot_noise[:,2] = 0.0
    gate_pts += gate_pos_noise      # shape: (num_gate, 3)
    gate_pts[:, 2] = gate_pts[:,2].clip(0.8, 2.0)       # clip the height of the gate
    gate_euler += gate_rot_noise    # shape: (num_gate, 3)

    # add noise to the gate shape
    gate_w = gate_size + np.random.uniform(-0.05, 0.05, num_gate)
    gate_h = gate_size + np.random.uniform(-0.05, 0.05, num_gate)
    gate_thickness = gate_thickness + np.random.uniform(-1, 1, num_gate) / 5 * gate_thickness
    gate_edge_thickness = np.random.uniform(0.15, 0.25, num_gate)
    gate_meshes = []
    # gate_euler = np.zeros_like(gate_euler) # for debug
    # randomly reverse the sequence
    reverse = 1
    if random.random() < 0.5:
        gate_pts = gate_pts[::-1, :]
        gate_euler = gate_euler[::-1, :]
        reverse = -1
    for i in range(num_gate):
        gate_meshes.append(
            make_gate((gate_w[i] + 2 * gate_edge_thickness[i], gate_h[i] + 2 * gate_edge_thickness[i], gate_thickness[i]),
                      (gate_w[i], gate_h[i], gate_thickness[i]),
                      gate_pts[i],
                      gate_euler[i]))
    meshes_list += gate_meshes

    adj_dir_shift_prop = difficulty * (cfg.adj_dir_shift_prop[1] - cfg.adj_dir_shift_prop[0]) + cfg.adj_dir_shift_prop[0]
    radius_dir_shift_prop = difficulty * (cfg.radius_dir_shift_prop[1] - cfg.radius_dir_shift_prop[0]) + cfg.radius_dir_shift_prop[0]
    num_wall_seg = int((difficulty * (cfg.num_wall_seg[1] - cfg.num_wall_seg[0]) + cfg.num_wall_seg[0]) * radius / cfg.radius[1])
    num_orbit_seg = int((difficulty * (cfg.num_orbit_seg[1] - cfg.num_orbit_seg[0]) + cfg.num_orbit_seg[0]) * radius / cfg.radius[1])
    
    # randomly select one segment as the start point
    # in this segment, there will be no any obstacles
    start_seg = random.randint(0, num_gate-1)
    # origin = (gate_pts[start_seg] + gate_pts[(start_seg + 1) % num_gate]) / 2    # start point
    origin = gate_pts[(start_seg + 1) % num_gate] - reverse * random.uniform(2, 4) * np.array([np.cos(gate_euler[(start_seg + 1) % num_gate][1] / 180 * np.pi + np.pi / 2), np.sin(gate_euler[(start_seg + 1) % num_gate][1] / 180 * np.pi + np.pi / 2), 0])
    origin[2] = random.uniform(0.7, 1.5)   # random height

    # STEP 2: generate the walls and orbits
    if cfg.add_obs:
        for i in range(num_gate):
            if i == start_seg:
                continue
            mid_pts = (gate_pts[i] + gate_pts[(i+1)%num_gate]) / 2
            vec_dir = gate_pts[(i+1)%num_gate] - gate_pts[i]
            # generate the walls
            for _ in range(num_wall_seg):
                offset_1 = vec_dir / 2 * np.random.uniform(-adj_dir_shift_prop, adj_dir_shift_prop)
                while True:
                    r = np.random.uniform(-10, 10, 3)
                    cross_product = np.cross(vec_dir, r)
                    if not np.allclose(cross_product, np.zeros(3)):
                        break
                offset_2 = cross_product / np.linalg.norm(cross_product) * np.random.uniform(-radius_dir_shift_prop, radius_dir_shift_prop) * radius
                pt = mid_pts + offset_1 + offset_2
                pt[2] = random.uniform(0.5, 3.0)   # random height
                # random orientation
                random_euler = np.random.uniform(-180, 180, 3)
                wall_size = np.random.uniform(cfg.wall_size[0], cfg.wall_size[1], 2)
                wall_thickness = np.random.uniform(cfg.wall_thickness[0], cfg.wall_thickness[1])
                wall_mesh = make_wall((wall_size[0], wall_size[1], wall_thickness), pt, random_euler)
                meshes_list.append(wall_mesh)
            # generate the orbits
            for _ in range(num_orbit_seg):
                offset_1 = vec_dir / 2 * np.random.uniform(-adj_dir_shift_prop, adj_dir_shift_prop)
                while True:
                    r = np.random.uniform(-10, 10, 3)
                    cross_product = np.cross(vec_dir, r)
                    if not np.allclose(cross_product, np.zeros(3)):
                        break
                offset_2 = cross_product / np.linalg.norm(cross_product) * np.random.uniform(-radius_dir_shift_prop, radius_dir_shift_prop) * radius
                pt = mid_pts + offset_1 + offset_2
                pt[2] = random.uniform(0.5, 3.0)   # random height
                # random orientation
                random_euler = np.random.uniform(-180, 180, 3)                
                orbit_mesh = make_orbit(pt, random_euler)
                meshes_list.append(orbit_mesh)
            if cfg.add_ground_obs:
                num_ground_obs_seg = int((difficulty * (cfg.num_ground_obs[1] - cfg.num_ground_obs[0]) + cfg.num_ground_obs[0]) * radius / cfg.radius[1])
                for _ in range(num_ground_obs_seg):
                    offset_1 = vec_dir / 2 * np.random.uniform(-adj_dir_shift_prop, adj_dir_shift_prop)
                    while True:
                        r = np.random.uniform(-10, 10, 3)
                        cross_product = np.cross(vec_dir, r)
                        if not np.allclose(cross_product, np.zeros(3)):
                            break
                    offset_2 = cross_product / np.linalg.norm(cross_product) * np.random.uniform(-radius_dir_shift_prop, radius_dir_shift_prop) * radius
                    pt = mid_pts + offset_1 + offset_2
                    # random orientation
                    ground_obs_mesh = make_ground_high_obs(pt, np.random.uniform(-180, 180, 3))
                    meshes_list.append(ground_obs_mesh)

                # randomly generate some small objects on the ground
                num_little_obj = random.randint(1, 4)
                for _ in range(num_little_obj):
                    offset_1 = vec_dir / 2 * np.random.uniform(-adj_dir_shift_prop, adj_dir_shift_prop)
                    while True:
                        r = np.random.uniform(-10, 10, 3)
                        cross_product = np.cross(vec_dir, r)
                        if not np.allclose(cross_product, np.zeros(3)):
                            break
                    offset_2 = cross_product / np.linalg.norm(cross_product) * np.random.uniform(-radius_dir_shift_prop, radius_dir_shift_prop) * radius
                    pos = mid_pts + offset_1 + offset_2
                    little_obj_mesh = make_ground_little_obj(pos, np.random.uniform(-180, 180, 3))
                    meshes_list.append(little_obj_mesh)

    # STEP 3: add planes
    terrain_height = 1.0
    pos = (cfg.size[0] / 2, cfg.size[1] / 2, -terrain_height / 2)
    dim = (cfg.size[0], cfg.size[1], terrain_height)
    ground = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(ground)

    # SETP 4: store gate pose information
    next_gate_id = (start_seg + 1) % num_gate

    gate_pose = np.zeros((num_gate, 6),dtype=np.float32)
    gate_pose[:,0:3] = gate_pts
    gate_pose[:,3:6] = gate_euler       # pitch, yaw, roll (left x, up y, forward z)
    
    return meshes_list, origin, {"gate_pose": gate_pose, "next_gate_id": next_gate_id}


def ZigzagRacingTerrain(difficulty: float, cfg: racing_terrains_cfg.ZigzagRacingTerrainCfg):
    """
    Generate a zigzag racing track with gates, walls and orbits.
    Args:
        cfg: Configuration for the zigzag racing track.
        difficulty: Difficulty of the racing track.
    Returns:
        A dictionary containing the generated terrain.
    """
    meshes_list = list()
    # parse the configuration
    track_length = cfg.track_length
    num_gate = cfg.num_gate
    only_yaw = cfg.only_yaw

    # STEP 1: generate the gates
    gate_size = cfg.gate_size[1] - (cfg.gate_size[1] - cfg.gate_size[0]) * difficulty
    gate_thickness = cfg.gate_thickness[0] + (cfg.gate_thickness[1] - cfg.gate_thickness[0]) * difficulty
    gate_pos_noise_scale = difficulty * (cfg.pos_noise_scale[1] - cfg.pos_noise_scale[0]) + cfg.pos_noise_scale[0]
    gate_pos_z_noise_scale = difficulty * (cfg.pos_z_noise_scale[1] - cfg.pos_z_noise_scale[0]) + cfg.pos_z_noise_scale[0]
    rot_noise_scale = difficulty * (cfg.rot_noise_scale[1] - cfg.rot_noise_scale[0]) + cfg.rot_noise_scale[0]

    gate_pts = np.zeros((num_gate, 3), dtype=np.float32)
    gate_euler = np.zeros((num_gate, 3), dtype=np.float32)

    theta = np.random.uniform(0, 2*np.pi)
    direction = np.array([np.cos(theta), np.sin(theta), 0])
    start_point = -0.5 * track_length * direction
    end_point = 0.5 * track_length * direction

    t_values = np.linspace(0, 1, num_gate)
    points = start_point + np.outer(t_values, end_point - start_point)

    for i in range(1, num_gate-1):
        noise_factor = t_values[i]

        noise_dir = np.array([-direction[1], direction[0], 0])
        noise_dir = noise_dir / np.linalg.norm(noise_dir)

        noise = 2.0 * (np.random.rand() - 0.5) * gate_pos_noise_scale * noise_factor * noise_dir
        points[i] += noise

        noise_dir = np.array([direction[0], direction[1], 0])
        noise_dir = noise_dir / np.linalg.norm(noise_dir)
        noise = 2.0 * (np.random.rand() - 0.5) * (track_length / (num_gate - 1) / 5) * noise_factor * noise_dir

        z_noise_dir = np.array([0, 0, 1])
        z_noise = 2.0 * (np.random.rand() - 0.5) * gate_pos_z_noise_scale * noise_factor * z_noise_dir
        points[i] += z_noise

    gate_euler[:,0] = 90.0
    gate_euler[:,1] = theta / np.pi * 180.0 + 90

    gate_pts = points
    gate_pts[:,0] += cfg.size[0] / 2
    gate_pts[:,1] += cfg.size[1] / 2
    gate_pts[:,2] += 1.0
    gate_pts[:, 2] = gate_pts[:,2].clip(0.8, 2.0)

    # Add noise to gate pose
    gate_rot_noise = np.random.uniform(-1, 1, (num_gate, 3)) * rot_noise_scale
    if only_yaw:
        gate_rot_noise[:, 0] = 0.0
        gate_rot_noise[:, 2] = 0.0
    gate_euler += gate_rot_noise

    # Add noise to gate shape
    gate_w = gate_size + np.random.uniform(-0.05, 0.05, num_gate)
    gate_h = gate_size + np.random.uniform(-0.05, 0.05, num_gate)
    gate_thickness = gate_thickness + np.random.uniform(-1, 1, num_gate) / 5 * gate_thickness
    gate_edge_thickness = np.random.uniform(0.15, 0.25, num_gate)
    gate_meshes = []
    for i in range(num_gate):
        gate_meshes.append(
            make_gate((gate_w[i] + 2 * gate_edge_thickness[i], gate_h[i] + 2 * gate_edge_thickness[i], gate_thickness[i]),
                      (gate_w[i], gate_h[i], gate_thickness[i]),
                      gate_pts[i],
                      gate_euler[i]))
    meshes_list += gate_meshes

    # Determine start point (origin)
    # Place origin before the first gate, aligned with its direction
    first_gate_direction = gate_pts[1, :] - gate_pts[0, :]
    first_gate_direction = first_gate_direction / np.linalg.norm(first_gate_direction)
    origin = gate_pts[0].copy() - first_gate_direction * random.uniform(2, 3)
    origin[2] = random.uniform(0.7, 1.5) # Random height

    # STEP 2: generate the walls and orbits
    if cfg.add_obs:
        adj_dir_shift_prop = difficulty * (cfg.adj_dir_shift_prop[1] - cfg.adj_dir_shift_prop[0]) + cfg.adj_dir_shift_prop[0]
        radius_dir_shift_prop = difficulty * (cfg.radius_dir_shift_prop[1] - cfg.radius_dir_shift_prop[0]) + cfg.radius_dir_shift_prop[0]
        num_wall_seg = int(difficulty * (cfg.num_wall_seg[1] - cfg.num_wall_seg[0]) + cfg.num_wall_seg[0])
        num_orbit_seg = int(difficulty * (cfg.num_orbit_seg[1] - cfg.num_orbit_seg[0]) + cfg.num_orbit_seg[0])

        # Iterate between gates to place obstacles
        for i in range(num_gate - 1):
            mid_pts = (gate_pts[i] + gate_pts[i+1]) / 2
            vec_dir = gate_pts[i+1] - gate_pts[i]

            # Generate walls
            cnt_wall = 0
            while cnt_wall < num_wall_seg:

                offset_1 = vec_dir / 2 * np.random.uniform(-adj_dir_shift_prop, adj_dir_shift_prop)

                # Generate a random vector not collinear with vec_dir for cross product
                while True:
                    r = np.random.uniform(-1, 1, 3)
                    cross_product = np.cross(vec_dir, r)
                    if not np.allclose(cross_product, np.zeros(3)):
                        break
                offset_2 = cross_product / np.linalg.norm(cross_product) * np.random.uniform(-radius_dir_shift_prop, radius_dir_shift_prop) * (cfg.gate_size[1] / 2) # Use gate size as a proxy for track width

                pt = mid_pts + offset_1 + offset_2
                pt[2] = random.uniform(0.5, 3.0)   # random height
                if np.linalg.norm(pt - gate_pts[i]) < cfg.no_obs_range or np.linalg.norm(pt - gate_pts[i+1]) < cfg.no_obs_range:
                    continue
                random_euler = np.random.uniform(-180, 180, 3)
                wall_size = np.random.uniform(cfg.wall_size[0], cfg.wall_size[1], 2)
                wall_thickness = np.random.uniform(cfg.wall_thickness[0], cfg.wall_thickness[1])
                wall_mesh = make_wall((wall_size[0], wall_size[1], wall_thickness), pt, random_euler)
                meshes_list.append(wall_mesh)
                cnt_wall += 1

            # Generate orbits
            cnt_orbit = 0
            while cnt_orbit < num_orbit_seg:
                offset_1 = vec_dir / 2 * np.random.uniform(-adj_dir_shift_prop, adj_dir_shift_prop)

                while True:
                    r = np.random.uniform(-1, 1, 3)
                    cross_product = np.cross(vec_dir, r)
                    if not np.allclose(cross_product, np.zeros(3)):
                        break
                offset_2 = cross_product / np.linalg.norm(cross_product) * np.random.uniform(-radius_dir_shift_prop, radius_dir_shift_prop) * (cfg.gate_size[1] / 2)

                pt = mid_pts + offset_1 + offset_2
                pt[2] = random.uniform(0.5, 3.0)   # random height
                if np.linalg.norm(pt - gate_pts[i]) < cfg.no_obs_range or np.linalg.norm(pt - gate_pts[i+1]) < cfg.no_obs_range:
                    continue
                random_euler = np.random.uniform(-180, 180, 3)                
                orbit_mesh = make_orbit(pt, random_euler)
                meshes_list.append(orbit_mesh)
                cnt_orbit += 1

            # Generate ground obstacles
            if cfg.add_ground_obs:
                num_ground_obs_seg = int(difficulty * (cfg.num_ground_obs[1] - cfg.num_ground_obs[0]) + cfg.num_ground_obs[0])
                cnt_ground_obs = 0
                while cnt_ground_obs < num_ground_obs_seg:
                    offset_1 = vec_dir / 2 * np.random.uniform(-adj_dir_shift_prop, adj_dir_shift_prop)

                    while True:
                        r = np.random.uniform(-1, 1, 3)
                        cross_product = np.cross(vec_dir, r)
                        if not np.allclose(cross_product, np.zeros(3)):
                            break
                    offset_2 = cross_product / np.linalg.norm(cross_product) * np.random.uniform(-radius_dir_shift_prop, radius_dir_shift_prop) * (cfg.gate_size[1] / 2)

                    pt = mid_pts + offset_1 + offset_2
                    if (np.linalg.norm(pt[:2] - gate_pts[i][:2]) < cfg.no_obs_range) or (np.linalg.norm(pt[:2] - gate_pts[i+1][:2]) < cfg.no_obs_range):
                        continue
                    random_euler = np.random.uniform(-180, 180, 3)
                    ground_obs_mesh = make_ground_high_obs(pt, random_euler)
                    meshes_list.append(ground_obs_mesh)
                    cnt_ground_obs += 1

                # randomly generate some small objects on the ground
                num_little_obj = random.randint(1, 4)
                for _ in range(num_little_obj):
                    offset_1 = vec_dir / 2 * np.random.uniform(-0.5, 0.5)
                    while True:
                        r = np.random.uniform(-1, 1, 3)
                        cross_product = np.cross(vec_dir, r)
                        if not np.allclose(cross_product, np.zeros(3)):
                            break
                    offset_2 = cross_product / np.linalg.norm(cross_product) * np.random.uniform(-radius_dir_shift_prop, radius_dir_shift_prop) * (cfg.gate_size[1] / 2)
                    pos = mid_pts + offset_1 + offset_2
                    if (np.linalg.norm(pos[:2] - gate_pts[i][:2]) < cfg.no_obs_range) or (np.linalg.norm(pos[:2] - gate_pts[i+1][:2]) < cfg.no_obs_range):
                        continue
                    little_obj_mesh = make_ground_little_obj(pos, np.random.uniform(-180, 180, 3))
                    meshes_list.append(little_obj_mesh)

    # STEP 3: add planes
    terrain_height = 1.0
    pos = (cfg.size[0] / 2, cfg.size[1] / 2, -terrain_height / 2)
    dim = (cfg.size[0], cfg.size[1], terrain_height)
    ground = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(ground)

    # STEP 4: store gate pose information
    next_gate_id = 0 # Always start with the first gate in the sequence

    gate_pose = np.zeros((num_gate, 6),dtype=np.float32)
    gate_pose[:,0:3] = gate_pts
    gate_pose[:,3:6] = gate_euler       # pitch, yaw, roll (left x, up y, forward z)

    return meshes_list, origin, {"gate_pose": gate_pose, "next_gate_id": next_gate_id}


def EllipseRacingTerrain(
    difficulty: float, cfg: racing_terrains_cfg.EllipseRacingTerrainCfg
):
    """
    Generate an elliptical racing track with gates, walls and orbits.
    Args:
        cfg: Configuration for the elliptical racing track.
        difficulty: Difficulty of the racing track.

    Returns:
        A dictionary containing the generated terrain.
    """
    meshes_list = list()

    #
    a_coef = cfg.long_axis_prop[0] + difficulty * (cfg.long_axis_prop[1] - cfg.long_axis_prop[0])  # long axis coefficient
    b_coef = cfg.short_axis_prop[0] + difficulty * (cfg.short_axis_prop[1] - cfg.short_axis_prop[0])  # short axis coefficient

    # parse the configuration
    gate_distance = cfg.gate_distance # distance between last and next gate
    a_ellipse = a_coef * gate_distance
    b_ellipse = b_coef * gate_distance
    num_gate = cfg.num_gate
    only_yaw = cfg.only_yaw

    # STEP 1: generate the gates
    gate_size = cfg.gate_size[1] - (cfg.gate_size[1] - cfg.gate_size[0]) * difficulty
    gate_thickness = cfg.gate_thickness[0] + (cfg.gate_thickness[1] - cfg.gate_thickness[0]) * difficulty
    gate_pos_noise_scale = difficulty * (cfg.pos_noise_scale[1] - cfg.pos_noise_scale[0]) + cfg.pos_noise_scale[0]
    # gate_pos_z_noise_scale = difficulty * (cfg.pos_z_noise_scale[1] - cfg.pos_z_noise_scale[0]) + cfg.pos_z_noise_scale[0]
    rot_noise_scale = difficulty * (cfg.rot_noise_scale[1] - cfg.rot_noise_scale[0]) + cfg.rot_noise_scale[0]

    gate_pts = np.zeros((num_gate, 3), dtype=np.float32)
    gate_euler = np.zeros((num_gate, 3), dtype=np.float32)
    gate_euler[:,0] = 90.0      # roll

    theta = np.random.uniform(0, 2*np.pi)
    theta_euler = theta / np.pi * 180.0
    long_side_direction = np.array([np.cos(theta), np.sin(theta), 0])
    long_side_start_point = -0.5 * a_ellipse * long_side_direction
    long_side_end_point = 0.5 * a_ellipse * long_side_direction

    short_side_direction = np.array([-np.sin(theta), np.cos(theta), 0])
    short_side_start_point = -0.5 * b_ellipse * short_side_direction
    short_side_end_point = 0.5 * b_ellipse * short_side_direction

    # Generate points along the elliptical path
    points = np.zeros_like(gate_pts)

    points[0] = long_side_start_point
    points[4] = long_side_end_point
    gate_euler[0, 1] = theta_euler
    gate_euler[4, 1] = 180 + theta_euler

    points[2] = short_side_end_point
    points[6] = short_side_start_point
    gate_euler[2, 1] = theta_euler + 90
    gate_euler[6, 1] = theta_euler + 270

    points[1] = points[2] - gate_distance * long_side_direction
    points[3] = points[2] + gate_distance * long_side_direction
    points[5] = points[6] + gate_distance * long_side_direction
    points[7] = points[6] - gate_distance * long_side_direction
    gate_euler[1, 1] = theta_euler + 90
    gate_euler[3, 1] = theta_euler + 90
    gate_euler[5, 1] = theta_euler + 270
    gate_euler[7, 1] = theta_euler + 270

    # generate the gates pose
    gate_pts = points

    gate_pts[:,0] += cfg.size[0] / 2
    gate_pts[:,1] += cfg.size[1] / 2
    gate_pts[:,2] += 1.0

    # gate_euler = np.zeros((num_gate,3),dtype=np.float32)
    # gate_euler[:,0] = 90.0      # roll
    # gate_euler[:,1] = theta / np.pi * 180.0  # pitch

    # add noise to the gate pose
    gate_pos_noise = np.random.uniform(-1, 1, (num_gate, 3)) * gate_pos_noise_scale
    gate_rot_noise = np.random.uniform(-1, 1, (num_gate, 3)) * rot_noise_scale
    if only_yaw:
        gate_rot_noise[:,0] = 0.0
        gate_rot_noise[:,2] = 0.0
    gate_pts += gate_pos_noise      # shape: (num_gate, 3)
    gate_pts[:, 2] = gate_pts[:,2].clip(0.8, 2.0)       # clip the height of the gate
    gate_euler += gate_rot_noise    # shape: (num_gate, 3)

    # add noise to the gate shape
    gate_w = gate_size + np.random.uniform(-0.05, 0.05, num_gate)
    gate_h = gate_size + np.random.uniform(-0.05, 0.05, num_gate)
    gate_thickness = gate_thickness + np.random.uniform(-1, 1, num_gate) / 5 * gate_thickness
    gate_edge_thickness = np.random.uniform(0.15, 0.22, num_gate)
    gate_meshes = []
    # gate_euler = np.zeros_like(gate_euler) # for debug
    reverse = 1
    if random.random() < 0.5:
        gate_pts = gate_pts[::-1, :]
        gate_euler = gate_euler[::-1, :]
        reverse = -1
    for i in range(num_gate):
        gate_meshes.append(
            make_gate((gate_w[i] + 2 * gate_edge_thickness[i], gate_h[i] + 2 * gate_edge_thickness[i], gate_thickness[i]),
                      (gate_w[i], gate_h[i], gate_thickness[i]),
                      gate_pts[i],
                      gate_euler[i]))

    meshes_list += gate_meshes

    adj_dir_shift_prop = difficulty * (cfg.adj_dir_shift_prop[1] - cfg.adj_dir_shift_prop[0]) + cfg.adj_dir_shift_prop[0]
    radius_dir_shift_prop = difficulty * (cfg.radius_dir_shift_prop[1] - cfg.radius_dir_shift_prop[0]) + cfg.radius_dir_shift_prop[0]
    num_wall_seg = int((difficulty * (cfg.num_wall_seg[1] - cfg.num_wall_seg[0]) + cfg.num_wall_seg[0]) )
    num_orbit_seg = int((difficulty * (cfg.num_orbit_seg[1] - cfg.num_orbit_seg[0]) + cfg.num_orbit_seg[0]))
    
    # randomly select one segment as the start point
    # in this segment, there will be no any obstacles
    start_seg = random.randint(0, num_gate-1)
    next_seg = (start_seg + 1) % num_gate
    seg_vec_norm = gate_pts[next_seg] - gate_pts[start_seg]
    seg_vec_norm = seg_vec_norm / np.linalg.norm(seg_vec_norm)
    # origin = (gate_pts[start_seg] + gate_pts[(start_seg + 1) % num_gate]) / 2    # start point
    origin = gate_pts[(start_seg) % num_gate] + seg_vec_norm * random.uniform(2, 3)  # start point
    origin[2] = random.uniform(0.7, 1.5)   # random height

    # STEP 2: generate the walls and orbits
    if cfg.add_obs:
        for i in range(num_gate):
            if i == start_seg:
                continue
            mid_pts = (gate_pts[i] + gate_pts[(i+1)%num_gate]) / 2
            vec_dir = gate_pts[(i+1)%num_gate] - gate_pts[i]
            # generate the walls
            for _ in range(num_wall_seg):
                offset_1 = vec_dir / 2 * np.random.uniform(-adj_dir_shift_prop, adj_dir_shift_prop)
                while True:
                    r = np.random.uniform(-10, 10, 3)
                    cross_product = np.cross(vec_dir, r)
                    if not np.allclose(cross_product, np.zeros(3)):
                        break
                offset_2 = cross_product / np.linalg.norm(cross_product) * np.random.uniform(-radius_dir_shift_prop, radius_dir_shift_prop) * gate_distance
                pt = mid_pts + offset_1 + offset_2
                pt[2] = random.uniform(0.5, 3.0)   # random height
                # random orientation
                random_euler = np.random.uniform(-180, 180, 3)
                wall_size = np.random.uniform(cfg.wall_size[0], cfg.wall_size[1], 2)
                wall_thickness = np.random.uniform(cfg.wall_thickness[0], cfg.wall_thickness[1])
                wall_mesh = make_wall((wall_size[0], wall_size[1], wall_thickness), pt, random_euler)
                meshes_list.append(wall_mesh)
            # generate the orbits
            for _ in range(num_orbit_seg):
                offset_1 = vec_dir / 2 * np.random.uniform(-adj_dir_shift_prop, adj_dir_shift_prop)
                while True:
                    r = np.random.uniform(-10, 10, 3)
                    cross_product = np.cross(vec_dir, r)
                    if not np.allclose(cross_product, np.zeros(3)):
                        break
                offset_2 = cross_product / np.linalg.norm(cross_product) * np.random.uniform(-radius_dir_shift_prop, radius_dir_shift_prop) * gate_distance
                pt = mid_pts + offset_1 + offset_2
                pt[2] = random.uniform(0.5, 3.0)   # random height
                # random orientation
                random_euler = np.random.uniform(-180, 180, 3)                
                orbit_mesh = make_orbit(pt, random_euler)
                meshes_list.append(orbit_mesh)
            if cfg.add_ground_obs:
                num_ground_obs_seg = int((difficulty * (cfg.num_ground_obs[1] - cfg.num_ground_obs[0]) + cfg.num_ground_obs[0]))
                for _ in range(num_ground_obs_seg):
                    offset_1 = vec_dir / 2 * np.random.uniform(-adj_dir_shift_prop, adj_dir_shift_prop)
                    while True:
                        r = np.random.uniform(-10, 10, 3)
                        cross_product = np.cross(vec_dir, r)
                        if not np.allclose(cross_product, np.zeros(3)):
                            break
                    offset_2 = cross_product / np.linalg.norm(cross_product) * np.random.uniform(-radius_dir_shift_prop, radius_dir_shift_prop) * gate_distance
                    pt = mid_pts + offset_1 + offset_2
                    # random orientation
                    ground_obs_mesh = make_ground_high_obs(pt, np.random.uniform(-180, 180, 3))
                    meshes_list.append(ground_obs_mesh)

                # randomly generate some small objects on the ground
                num_little_obj = random.randint(1, 2)
                for _ in range(num_little_obj):
                    offset_1 = vec_dir / 2 * np.random.uniform(-adj_dir_shift_prop, adj_dir_shift_prop)
                    while True:
                        r = np.random.uniform(-10, 10, 3)
                        cross_product = np.cross(vec_dir, r)
                        if not np.allclose(cross_product, np.zeros(3)):
                            break
                    offset_2 = cross_product / np.linalg.norm(cross_product) * np.random.uniform(-radius_dir_shift_prop, radius_dir_shift_prop) * gate_distance
                    pos = mid_pts + offset_1 + offset_2
                    little_obj_mesh = make_ground_little_obj(pos, np.random.uniform(-180, 180, 3))
                    meshes_list.append(little_obj_mesh)

    # STEP 3: add planes
    terrain_height = 1.0
    pos = (cfg.size[0] / 2, cfg.size[1] / 2, -terrain_height / 2)
    dim = (cfg.size[0], cfg.size[1], terrain_height)
    ground = trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos))
    meshes_list.append(ground)

    # SETP 4: store gate pose information
    next_gate_id = (start_seg + 1) % num_gate

    gate_pose = np.zeros((num_gate, 6),dtype=np.float32)
    gate_pose[:,0:3] = gate_pts
    gate_pose[:,3:6] = gate_euler       # pitch, yaw, roll (left x, up y, forward z)
    
    return meshes_list, origin, {"gate_pose": gate_pose, "next_gate_id": next_gate_id}

