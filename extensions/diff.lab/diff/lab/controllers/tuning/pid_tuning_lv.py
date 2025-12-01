import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="controller tuning.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

assert args_cli.num_envs <= 1, "Only support single environment due to trajectory generation method."
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, Articulation
from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg, InteractiveScene
from isaaclab.utils import configclass
import diff.lab
from diff.lab_assets.quadcopter import DRONE_CFG
from diff.lab.controllers import PSControllerCfg, LVControllerCfg, CTBRControllerCfg
from isaaclab.utils.math import quat_rotate_inverse, quat_rotate, euler_xyz_from_quat, wrap_to_pi, quat_from_matrix
import random
import math
import isaaclab.utils.math as math_utils

@configclass
class SceneCfg(InteractiveSceneCfg):

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # drone
    robot: AssetBaseCfg = DRONE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

def get_state_from_sim(robot):
    """
    All states here are non-differentiable (directly obtained from isaac-sim).
    """
    states_all = robot.data.root_state_w.clone()
    pos = states_all[:, :3]
    quat = states_all[:, 3:7]
    lin_vel_w = states_all[:, 7:10]
    ang_vel_w = states_all[:, 10:13]
    ang_vel_b = math_utils.quat_rotate_inverse(quat, ang_vel_w)  #robot.data.root_ang_vel_w
    lin_vel_b = math_utils.quat_rotate_inverse(quat, lin_vel_w)  #robot.data.root_lin_vel_b
    
    return {
        "pos": pos,
        "quat": quat,
        "lin_vel_w": lin_vel_w,
        "ang_vel_w": ang_vel_w,
        "lin_vel_b": lin_vel_b,
        "ang_vel_b": ang_vel_b
    } 

def run(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot:Articulation = scene["robot"]

    # controller
    controller_cfg = LVControllerCfg(
        rate_gain=[180.0, 180.0, 200.0],
        pose_gain=[18.0, 18.0, 20.0],
        speed_gain=[10.0, 10.0, 20.0]
    )
    inertial=robot.root_physx_view.get_inertias()[0].sum(0).to(args_cli.device).reshape(3,3)
    _robot_mass = robot.root_physx_view.get_masses()[0].sum().to(args_cli.device)
    _robot_weight = (_robot_mass * 9.81).item()

    dt = sim.get_physics_dt()
    controller = controller_cfg.class_type(controller_cfg, args_cli.num_envs, args_cli.device, _robot_mass, inertial, dt)
    body_ids, body_names = robot.find_bodies("body", preserve_order = True)

    force_body = torch.zeros(args_cli.num_envs, 1, 3, device=args_cli.device)
    torque_body = torch.zeros(args_cli.num_envs, 1, 3, device=args_cli.device)

    t = 0.0

    cmd = torch.zeros(args_cli.num_envs, 4, device=args_cli.device)

    states_list = []
    yaw_list = []
    cmd_list = []
    t_list = []
    body_rate_des = []
    actual_body_rate = []
    euler_des = []
    euler_actual = []
    actions = []
    # ids, _ = robot.find_bodies("body", preserve_order = True)
    while simulation_app.is_running():
        states = get_state_from_sim(robot)
        # cmd[:, 0] = _robot_weight
        cmd[:, 0] = 1.0
        # cmd[:, -1] = 2.0 *math.cos(t * 8.0)
        cmd[:, 1:] = 1.0

        state_ = robot.data.root_state_w.clone()

        roll, pitch, yaw = math_utils.euler_xyz_from_quat(states['quat'])
        yaw_list.append(wrap_to_pi(yaw[0]))
        states_list.append(states['lin_vel_w'][0])
        cmd_list.append(cmd[0, :].clone())
        t_list.append(t)

        thrusts_now, thrust_torque_now, br_des, R_des = controller.compute(states, cmd)
        body_rate_des.append(br_des[0])
        actual_body_rate.append(states['ang_vel_b'][0])
        euler_des.append(wrap_to_pi(torch.stack(euler_xyz_from_quat(quat_from_matrix(R_des)), dim=-1)))
        euler_actual.append(wrap_to_pi(torch.stack(euler_xyz_from_quat(states['quat']), dim=-1)))
        actions.append(thrust_torque_now[0])
        _processed_actions = thrust_torque_now.clone().detach()
        torque_body[:,0,:] = _processed_actions[:,1:4]
        force_body[:,0,2] = _processed_actions[:,0]
        
        robot.set_external_force_and_torque(force_body, torque_body, body_ids=body_ids)
        scene.write_data_to_sim()
        sim.step()
        t += dt
        scene.update(dt)

        if t > 2.0:
            break

    return states_list, yaw_list, cmd_list, t_list, body_rate_des, actual_body_rate, euler_des, euler_actual, actions


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0., 0., 0.])
    scene_cfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    return run(sim, scene)




if __name__ == "__main__":
    # run the main function
    states_list, yaw_list, cmd_list, t_list, br_des, br_actual, euler_des, euler_actual, actions = main()

    cmd_w = torch.stack(cmd_list).cpu().numpy()
    state_w = torch.stack(states_list).cpu().numpy()
    yaw_w = torch.stack(yaw_list).cpu().numpy()
    br_des = torch.stack(br_des).cpu().numpy()
    br_actual = torch.stack(br_actual).cpu().numpy()
    euler_des = torch.stack(euler_des).cpu().numpy()[:, 0]
    euler_actual = torch.stack(euler_actual).cpu().numpy()[:, 0]
    actions = torch.stack(actions).cpu().numpy()


    import matplotlib.pyplot as plt

    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(t_list, cmd_w[:, 1], label="cmd vx")
    plt.plot(t_list, state_w[:, 0], label="Real vx")
    plt.legend()
    plt.subplot(4, 1, 2)
    plt.plot(t_list, cmd_w[:, 2], label="cmd vy")
    plt.plot(t_list, state_w[:, 1], label="Real vy")
    plt.legend()
    plt.subplot(4, 1, 3)
    plt.plot(t_list, cmd_w[:, 3], label="cmd vz")
    plt.plot(t_list, state_w[:, 2], label="Real vz")
    plt.legend()
    plt.subplot(4, 1, 4)
    plt.plot(t_list, cmd_w[:, 0], label="des_yaw")
    plt.plot(t_list, yaw_w, label="yaw")
    plt.legend()

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(t_list, br_des[:, 0], label="des_wx")
    plt.plot(t_list, br_actual[:, 0], label="actual_wx")
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(t_list, br_des[:, 1], label="des_wy")
    plt.plot(t_list, br_actual[:, 1], label="actual_wy")
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(t_list, br_des[:, 2], label="des_wz")
    plt.plot(t_list, br_actual[:, 2], label="actual_wz")
    plt.legend()

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(t_list, euler_des[:, 0], label="des_roll")
    plt.plot(t_list, euler_actual[:, 0], label="actual_roll")
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(t_list, euler_des[:, 1], label="des_pitch")
    plt.plot(t_list, euler_actual[:, 1], label="actual_pitch")
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(t_list, euler_des[:, 2], label="des_yaw")
    plt.plot(t_list, euler_actual[:, 2], label="actual_yaw")
    plt.legend()

    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(t_list, actions[:, 0], label="thrust")
    plt.legend()
    plt.subplot(4, 1, 2)
    plt.plot(t_list, actions[:, 1], label="torque_x")
    plt.legend()
    plt.subplot(4, 1, 3)
    plt.plot(t_list, actions[:, 2], label="torque_y")
    plt.legend()
    plt.subplot(4, 1, 4)
    plt.plot(t_list, actions[:, 3], label="torque_z")
    plt.legend()

    plt.show()


    # close sim app
    simulation_app.close()