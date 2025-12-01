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
# *  Data: 2025/03/04     	             *
# *  Contact: yu-feng@sjtu.edu.cn                                               *
# *  Description: None                                                          *
# *******************************************************************************
import warp as wp
import torch

############################# legacy code #############################
# @wp.kernel(enable_backward=False)
# def check_uav_collision_kernel(
#     mesh: wp.uint64,
#     uav_position: wp.array(dtype=wp.vec3),
#     uav_orientation: wp.array(dtype=wp.vec4),
#     collision_num: wp.array(dtype=wp.int32),
#     max_dist: float = 1e6,
#     arm_length: float = 0.1,
#     point_mass_mode: int = False,
# ):
#     # get the thread id
#     tid = wp.tid()

#     if point_mass_mode: # only check the mass-point
#         query = wp.mesh_query_point_sign_normal(mesh, uav_position[tid], max_dist, 1e-3)
#         if query.result:
#             if query.sign < 0.0:
#                 collision_num[tid] = 1
#             else:
#                 collision_num[tid] = 0
#     else:   # check four cornels on the uav box collider
#         center_pt = uav_position[tid]
#         lf_vec = wp.vec3( 0.707 * arm_length,  0.707 * arm_length, 0.0)
#         lr_vec = wp.vec3(-0.707 * arm_length,  0.707 * arm_length, 0.0)
#         rf_vec = wp.vec3( 0.707 * arm_length, -0.707 * arm_length, 0.0)
#         rr_vec = wp.vec3(-0.707 * arm_length, -0.707 * arm_length, 0.0)
#         q = wp.quat(uav_orientation[tid][0], uav_orientation[tid][1], uav_orientation[tid][2], uav_orientation[tid][3])
#         lf_pt = center_pt + wp.quat_rotate(q, lf_vec)
#         lr_pt = center_pt + wp.quat_rotate(q, lr_vec)
#         rf_pt = center_pt + wp.quat_rotate(q, rf_vec)
#         rr_pt = center_pt + wp.quat_rotate(q, rr_vec)
#         query = wp.mesh_query_point_sign_normal(mesh, lf_pt, max_dist, 1e-3)
#         if query.result:
#             if query.sign < 0.0:
#                 collision_num[tid] += 1
#         query = wp.mesh_query_point_sign_normal(mesh, lr_pt, max_dist, 1e-3)
#         if query.result:
#             if query.sign < 0.0:
#                 collision_num[tid] += 1
#         query = wp.mesh_query_point_sign_normal(mesh, rf_pt, max_dist, 1e-3)
#         if query.result:
#             if query.sign < 0.0:
#                 collision_num[tid] += 1
#         query = wp.mesh_query_point_sign_normal(mesh, rr_pt, max_dist, 1e-3)
#         if query.result:
#             if query.sign < 0.0:
#                 collision_num[tid] += 1

# def get_uav_collision_num(
#     mesh: wp.Mesh,
#     uav_position: torch.Tensor,
#     uav_orientation: torch.Tensor,
#     point_mass_mode: bool,
#     arm_length: float,
# )->torch.Tensor:
#     device = uav_position.device
#     torch_device = wp.device_to_torch(mesh.device)
#     position = uav_position.to(torch_device).view(-1, 3).contiguous()
#     # NOTE warp uses x,y,z,w quaternion, while isaac uses w,x,y,z
#     uav_orientation = torch.cat([uav_orientation[:, 1:], uav_orientation[:, 0:1]], dim=-1)
#     orientation = uav_orientation.to(torch_device).view(-1, 4).contiguous()
#     num_uav = position.shape[0]
#     num_collisions = torch.zeros(num_uav, dtype=torch.int32, device=torch_device).contiguous()

#     position_wp = wp.from_torch(position, dtype=wp.vec3)
#     orientation_wp = wp.from_torch(orientation, dtype=wp.vec4)
#     num_collisions_wp = wp.from_torch(num_collisions, dtype=wp.int32)

#     # run the kernel
#     wp.launch(
#         kernel=check_uav_collision_kernel,
#         dim=num_uav,
#         inputs = [
#             mesh.id,
#             position_wp,
#             orientation_wp,
#             num_collisions_wp,
#             1e6,
#             arm_length,
#             point_mass_mode
#         ],
#         device=mesh.device
#     )
#     wp.synchronize()
#     return num_collisions.to(device=device)

# import numpy as np

# def convert_to_warp_mesh(points: np.ndarray, indices: np.ndarray, device: str, support_winding_number) -> wp.Mesh:
#     """Create a warp mesh object with a mesh defined from vertices and triangles.

#     Args:
#         points: The vertices of the mesh. Shape is (N, 3), where N is the number of vertices.
#         indices: The triangles of the mesh as references to vertices for each triangle.
#             Shape is (M, 3), where M is the number of triangles / faces.
#         device: The device to use for the mesh.

#     Returns:
#         The warp mesh object.
#     """
#     return wp.Mesh(
#         points=wp.array(points.astype(np.float32), dtype=wp.vec3, device=device),
#         indices=wp.array(indices.astype(np.int32).flatten(), dtype=wp.int32, device=device),
#         support_winding_number=support_winding_number
#     )


# *********************************end legacy code**********************************************

@wp.kernel(enable_backward=False)
def check_uav_collision_ray_kernel(
    mesh: wp.uint64,
    uav_position: wp.array(dtype=wp.vec3),
    uav_orientation: wp.array(dtype=wp.vec4),
    collision_num: wp.array(dtype=wp.int32),
    lattices: wp.array(dtype=wp.vec3),
    num_lattices: int,
    max_dist: float = 1e6,
    arm_length: float = 0.1,
    height: float = 0.05,
):
    # get the thread id
    tid = wp.tid()
    t = float(0.0)  # hit distance along ray
    u = float(0.0)  # hit face barycentric u
    v = float(0.0)  # hit face barycentric v
    sign = float(0.0)  # hit face sign
    n = wp.vec3()  # hit face normal
    f = int(0)  # hit face index

    front_dir = wp.vec3(1.0, 0.0, 0.0)
    back_dir = wp.vec3(-1.0, 0.0, 0.0)
    left_dir = wp.vec3(0.0, 1.0, 0.0)
    right_dir = wp.vec3(0.0, -1.0, 0.0)
    up_dir = wp.vec3(0.0, 0.0, 1.0)
    down_dir = wp.vec3(0.0, 0.0, -1.0)

    if not lattices: # only check the mass-point
        hit_success = wp.mesh_query_ray(mesh,uav_position[tid],front_dir,max_dist, t, u, v, sign, n, f)
        if hit_success:
            if sign <= 0.0:
                collision_num[tid] = 1
            else:
                collision_num[tid] = 0
        hit_success = wp.mesh_query_ray(mesh,uav_position[tid],back_dir,max_dist, t, u, v, sign, n, f)
        if hit_success:
            if sign <= 0.0:
                collision_num[tid] = 1
            else:
                collision_num[tid] = 0
        hit_success = wp.mesh_query_ray(mesh,uav_position[tid],left_dir,max_dist, t, u, v, sign, n, f)
        if hit_success:
            if sign <= 0.0:
                collision_num[tid] = 1
            else:
                collision_num[tid] = 0
        hit_success = wp.mesh_query_ray(mesh,uav_position[tid],right_dir,max_dist, t, u, v, sign, n, f)
        if hit_success:
            if sign <= 0.0:
                collision_num[tid] = 1
            else:
                collision_num[tid] = 0
        hit_success = wp.mesh_query_ray(mesh,uav_position[tid],up_dir,max_dist, t, u, v, sign, n, f)
        if hit_success:
            if sign <= 0.0:
                collision_num[tid] = 1
            else:
                collision_num[tid] = 0
        hit_success = wp.mesh_query_ray(mesh,uav_position[tid],down_dir,max_dist, t, u, v, sign, n, f)
        if hit_success:
            if sign <= 0.0:
                collision_num[tid] = 1
            else:
                collision_num[tid] = 0
    else:   # check eight cornels on the uav box collider
        center_pt = uav_position[tid]
        q = wp.quat(uav_orientation[tid][0], uav_orientation[tid][1], uav_orientation[tid][2], uav_orientation[tid][3])
    
        for i in range(num_lattices):
            x_sign = lattices[i][0]
            y_sign = lattices[i][1]
            z_sign = lattices[i][2]
            vec = wp.vec3(x_sign * 0.707 * arm_length, y_sign * 0.707 * arm_length, z_sign * 0.5 * height)
            pt = center_pt + wp.quat_rotate(q, vec)
            # check the collision with the mesh
            hit_success = wp.mesh_query_ray(mesh,pt,front_dir,max_dist, t, u, v, sign, n, f)
            if hit_success:
                if sign <= 0.0:
                    collision_num[tid] += 1
                    continue
            hit_success = wp.mesh_query_ray(mesh,pt,back_dir,max_dist, t, u, v, sign, n, f)
            if hit_success:
                if sign <= 0.0:
                    collision_num[tid] += 1
                    continue
            hit_success = wp.mesh_query_ray(mesh,pt,left_dir,max_dist, t, u, v, sign, n, f)
            if hit_success:
                if sign <= 0.0:
                    collision_num[tid] += 1
                    continue
            hit_success = wp.mesh_query_ray(mesh,pt,right_dir,max_dist, t, u, v, sign, n, f)
            if hit_success:
                if sign <= 0.0:
                    collision_num[tid] += 1
                    continue
            hit_success = wp.mesh_query_ray(mesh,pt,up_dir,max_dist, t, u, v, sign, n, f)
            if hit_success:
                if sign <= 0.0:
                    collision_num[tid] += 1
                    continue
            hit_success = wp.mesh_query_ray(mesh,pt,down_dir,max_dist, t, u, v, sign, n, f)
            if hit_success:
                if sign <= 0.0:
                    collision_num[tid] += 1
                    continue

            

def get_uav_collision_num_ray(
    mesh: wp.Mesh,
    uav_position: torch.Tensor,
    uav_orientation: torch.Tensor,
    arm_length: float,
    height: float,
    max_dist: float = 1e6,
    lattices: torch.Tensor | None = None,
)->torch.Tensor:
    """
    Args:
        - `mesh`: wp.Mesh, usually the terrain mesh
        - `uav_position`: torch.Tensor, the position of uav, shape is (N, 3)
        - `uav_orientation`: torch.Tensor, the orientation of uav, shape is (N, 4), w,x,y,z quaternion
        - `arm_length`: float, the arm length of the uav
        - `height`: float, the height of the uav
        - `max_dist`: float, the maximum distance to check the collision
        - `lattices`: torch.Tensor, the lattices of the uav box collider, shape is (M, 3), M is the number of lattices
    Returns:
        - `num_collisions`: torch.Tensor, the number of collisions for each uav
    """
    device = uav_position.device
    torch_device = wp.device_to_torch(mesh.device)
    position = uav_position.to(torch_device).view(-1, 3).contiguous()
    # NOTE warp uses x,y,z,w quaternion, while isaac uses w,x,y,z
    uav_orientation = torch.cat([uav_orientation[:, 1:], uav_orientation[:, 0:1]], dim=-1)
    orientation = uav_orientation.to(torch_device).view(-1, 4).contiguous()
    num_uav = position.shape[0]
    num_collisions = torch.zeros(num_uav, dtype=torch.int32, device=torch_device).contiguous()

    position_wp = wp.from_torch(position, dtype=wp.vec3)
    if lattices is not None:
        lattices_wp = wp.from_torch(lattices, dtype=wp.vec3)
        num_lattices = lattices.shape[0]
    else:
        lattices_wp = None
        num_lattices = 0
    orientation_wp = wp.from_torch(orientation, dtype=wp.vec4)
    num_collisions_wp = wp.from_torch(num_collisions, dtype=wp.int32)

    # run the kernel
    wp.launch(
        kernel=check_uav_collision_ray_kernel,
        dim=num_uav,
        inputs = [
            mesh.id,
            position_wp,
            orientation_wp,
            num_collisions_wp,
            lattices_wp,
            num_lattices,
            max_dist,
            arm_length,
            height
    ],
        device=mesh.device
    )
    wp.synchronize()
    return num_collisions.to(device=device)