"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on adding sensors on a robot.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, Articulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg, patterns
# from isaaclab.sensors import ImuCfg
# from isaaclab.sensors.imu.imu_noise import ImuNoiseCfg
from diff.lab.sensors.imu  import HifiImuCfg, ImuNoiseCfg
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from omni.isaac.lab_assets.anymal import ANYMAL_C_CFG  # isort: skip


@configclass
class SensorsSceneCfg(InteractiveSceneCfg):
    """Design the scene with sensors on the robot."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    imu = HifiImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        update_period=0.005,
        offset=HifiImuCfg.OffsetCfg(pos=(0., 0., 0.)),
        noise=ImuNoiseCfg(
            add_noise = True,
            g_std = (0.00272391738310747, 0.00248849782611228, 0.00272332577563485), # psd(1.926,1.76,1.926)e-4, 200Hz
            a_std = (0.00643187932253599, 0.00661386698561032, 0.00673225201283004), # psd(4.548,4.677,4.76)e-4
            gb_sta = (4.00424136983284e-14, 4.98197419961447e-15, -6.5696457219509e-15),
            ab_sta = (1.73301445792617e-13, -7.93732502701179e-13, -1.84847751355576e-13),
            gb_dyn = (7.6339404800228e-05, 4.50248175403541e-05, 8.75796277840371e-05),
            ab_dyn = (0.000252894096875598, 0.000349683866037958, 0.000323068534025731),
            gb_corr = (500, 700, 200),
            ab_corr = (40, 20, 100),
            arrw_std = (8.21484738626e-05, 4.54275740041735e-05, 0.000103299115514897), # psd(5.809,3.212,7.304)e-6
            vrrw_std = (0.00031522133759985, 0.000519606636158211, 0.000396688807571295) # psd(2.229,3.674,2.805)e-5
        )
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Run the simulator."""
    # Define simulation stepping
    sim_fre = 400
    imu_fre = 200
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Simulate physics
    while simulation_app.is_running():
        # # Reset
        # if count % 500 == 0:
        #     # reset counter
        #     count = 0
        #     # reset the scene entities
        #     # root state
        #     # we offset the root state by the origin since the states are written in simulation world frame
        #     # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
        #     root_state = scene["robot"].data.default_root_state.clone()
        #     root_state[:, :3] += scene.env_origins
        #     scene["robot"].write_root_state_to_sim(root_state)
        #     # set joint positions with some noise
        #     joint_pos, joint_vel = (
        #         scene["robot"].data.default_joint_pos.clone(),
        #         scene["robot"].data.default_joint_vel.clone(),
        #     )
        #     joint_pos += torch.rand_like(joint_pos) * 0.1
        #     scene["robot"].write_joint_state_to_sim(joint_pos, joint_vel)
        #     # clear internal buffers
        #     scene.reset()
        #     print("[INFO]: Resetting robot state...")
        # Apply default actions to the robot
        # -- generate actions/commands
        targets = scene["robot"].data.default_joint_pos
        # -- apply action to the robot
        scene["robot"].set_joint_position_target(targets)
        # -- write data to sim
        scene.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        scene.update(sim_dt)

        # if count == 12000:
        #     stop = 1
        # save to file
        if count % (sim_fre/imu_fre) == 0:
            robot: Articulation = scene["robot"]
            # print(robot.data.applied_torque)
            for i in range(2):
                lin_acc_b = scene["imu"].data.lin_acc_b[i, :].tolist()
                ang_vel_b = scene["imu"].data.ang_vel_b[i, :].tolist()
                root_state = scene["robot"].data.root_state_w[i, :].tolist()
                body_state = scene["robot"].data.body_state_w[i, :].tolist()
                body_acc   = scene["robot"].data.body_acc_w[i, :].tolist()
                num_instances = len(body_state)
                # print(np.array(root_state).shape)
                # print(np.array(body_state).shape)
                # print(np.array(body_acc).shape)
                # file_path = "imu_data_without_noise.txt"
                file_path = f"imu_data_f_sim_{sim_fre}_{i}.txt"
                if os.path.exists(file_path):
                    mode = "a"
                else:
                    mode = "w"
                with open(file_path, mode) as f:
                    if mode == "w":
                        f.write("count, lin_acc_b_x, lin_acc_b_y, lin_acc_b_z, ang_vel_b_x, ang_vel_b_y, ang_vel_b_z\n")
                    f.write(str(count) + ", " + str(lin_acc_b[0]) + ", " + str(lin_acc_b[1]) + ", " + str(lin_acc_b[2]) + ", "
                            + str(ang_vel_b[0]) + ", " + str(ang_vel_b[1]) + ", " + str(ang_vel_b[2]) + "\n")
                    
                # robot_state_path = f"robot_state_{i}.txt"
                # if os.path.exists(robot_state_path):
                #     mode = "a"
                # else:
                #     mode = "w"
                # with open(robot_state_path, mode) as f:
                #     if mode == "w":
                #         f.write("count, pos_w_x, pos_w_y, pos_w_z, quat_w_w, quat_w_x, quat_w_y, quat_w_z, lin_vel_w_x, lin_vel_w_y, lin_vel_w_z, ang_vel_w_x, ang_vel_w_y, ang_vel_w_z\n")
                #     f.write(str(count) + ", " + ",".join(map(str, root_state)) + "\n")
                    
                # for j in range(num_instances):
                #     body_state_path = f"body_state_{i}_{j}.txt"
                #     if os.path.exists(body_state_path):
                #         mode = "a"
                #     else:
                #         mode = "w"
                #     with open(body_state_path, mode) as f:
                #         if mode == "w":
                #             f.write("count, pos_w_x, pos_w_y, pos_w_z, quat_w_w, quat_w_x, quat_w_y, quat_w_z, lin_vel_w_x, lin_vel_w_y, lin_vel_w_z, ang_vel_w_x, ang_vel_w_y, ang_vel_w_z, lin_acc_w_x, lin_acc_w_y, lin_acc_w_z, ang_acc_w_x, ang_acc_w_y, ang_acc_w_z\n")
                #         f.write(str(count) + ", " + ",".join(map(str, body_state[j])) + "," + ",".join(map(str, body_acc[j])) + "\n")
        
        if(count % 10000 == 0):
            print("-------------------------------")
            print(count)
            print(scene["imu"])
            print("Received imu data: ", scene["imu"].data)

        if (count > sim_fre*1200*1 + sim_fre*5):
            break


def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.0025, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])
    # design scene
    scene_cfg = SensorsSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
