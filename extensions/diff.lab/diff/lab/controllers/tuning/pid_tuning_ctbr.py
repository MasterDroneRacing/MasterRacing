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
    lin_acc_w = robot.data.body_lin_acc_w[:, 0, :].clone()
    ang_acc_w = robot.data.body_ang_acc_w[:, 0, :].clone()
    lin_acc_b = math_utils.quat_rotate_inverse(quat, lin_acc_w)  #robot.data.body_lin_acc_b[:, 0, :]
    ang_acc_b = math_utils.quat_rotate_inverse(quat, ang_acc_w)  #robot.data.body_ang_acc_b[:, 0, :]
    
    return {
        "pos": pos,
        "quat": quat,
        "lin_vel_w": lin_vel_w,
        "ang_vel_w": ang_vel_w,
        "lin_vel_b": lin_vel_b,
        "ang_vel_b": ang_vel_b,
        "lin_acc_w": lin_acc_w,
        "ang_acc_w": ang_acc_w,
        "lin_acc_b": lin_acc_b,
        "ang_acc_b": ang_acc_b
    }


def run(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot:Articulation = scene["robot"]

    # controller
    controller_cfg = CTBRControllerCfg(
                        rate_gain_p=[40, 40, 45],
                        rate_gain_i=[0.0, 0.0, 0.0],
                        rate_gain_d=[0.0005, 0.0005, 0.0003],
                        body_rate_bound=[-6, 6],
                        thrust_ctrl_delay=0.03,
                        torque_ctrl_delay=(0.02, 0.02, 0.02)
                        )
    inertial=robot.root_physx_view.get_inertias()[0].sum(0).to(args_cli.device).reshape(3,3)
    _robot_mass = robot.root_physx_view.get_masses()[0].sum().to(args_cli.device)
    _robot_weight = (_robot_mass * 9.81).item()

    dt = sim.get_physics_dt()
    print(dt)
    controller = controller_cfg.class_type(controller_cfg, args_cli.num_envs, args_cli.device, _robot_mass, inertial, dt)
    body_ids, body_names = robot.find_bodies("body", preserve_order = True)

    force_body = torch.zeros(args_cli.num_envs, 1, 3, device=args_cli.device)
    torque_body = torch.zeros(args_cli.num_envs, 1, 3, device=args_cli.device)

    t = 0.0

    cmd = torch.zeros(args_cli.num_envs, 4, device=args_cli.device)

    cmd_list = []
    t_list = []
    actual_body_rate = []
    actions = []
    # ids, _ = robot.find_bodies("body", preserve_order = True)
    while simulation_app.is_running():
        states = get_state_from_sim(robot)
        #cmd[:, 0] = _robot_weight
        cmd[:, 0] = 3 * math.sin(t * 10.0) + 3.0
        cmd[:, 1:] = 10 * math.cos(t * 10.0)

        cmd_list.append(cmd[0, :].clone())
        t_list.append(t)

        thrusts_now, thrust_torque_now = controller.compute(states, cmd)
        actual_body_rate.append(states['ang_vel_b'][0])

        actions.append(thrust_torque_now[0])
        _processed_actions = thrust_torque_now.clone().detach()
        torque_body[:,0,:] = _processed_actions[:,1:4]
        force_body[:,0,2] = _processed_actions[:,0]
        
        robot.set_external_force_and_torque(force_body, torque_body, body_ids=body_ids)
        scene.write_data_to_sim()
        sim.step()
        t += dt
        scene.update(dt)

        if t > 1.0:
            break

    return cmd_list, t_list, actual_body_rate, actions


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.03, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 2.5], [0., 0., 0.])
    scene_cfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[INFO]: Setup complete...")
    return run(sim, scene)

if __name__ == "__main__":
    # run the main function
    cmd_list, t_list, br_real, actions = main()

    cmd = torch.stack(cmd_list).cpu().numpy()
    br_real = torch.stack(br_real).cpu().numpy()
    actions = torch.stack(actions).cpu().numpy()


    import matplotlib.pyplot as plt

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(t_list, cmd[:, 1], label="wx_des")
    plt.plot(t_list, br_real[:, 0], label="wx_real")
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(t_list, cmd[:, 2], label="wy_des")
    plt.plot(t_list, br_real[:, 1], label="wy_real")
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(t_list, cmd[:, 3], label="wz_des")
    plt.plot(t_list, br_real[:, 2], label="wz_real")
    plt.legend()

    plt.figure()
    plt.subplot(4, 1, 1)
    plt.plot(t_list, cmd[:, 0], label="thrust_cmd", linestyle="--")
    plt.plot(t_list, actions[:, 0], label="thrust_real", linestyle=":")
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