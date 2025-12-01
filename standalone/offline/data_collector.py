"""Script to play a checkpoint and collect data if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import standalone.rsl_rl.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--dataset", type=str, default="racing_data.h5", help="Name for *.h5 dataset.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# check format of dataset
if not args_cli.dataset.endswith(".h5"):
    raise ValueError(f"Dataset name '{args_cli.dataset}' must end with '.h5'.")

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

from standalone.rsl_rl.ext.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)
# Import extensions to set up environment tasks
import diff.lab # noqa: F401
import diff.lab_tasks  # noqa: F401

import h5py

def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # initialize the data collector
    h5_file = h5py.File("/data/racing_data/" + args_cli.dataset, "w")
    N_positive = 1000000
    N_negative = 1000000
    h5_file.create_dataset("features", shape=(N_positive + N_negative, 192), maxshape=(None, 192), dtype="float32")
    h5_file.create_dataset("supervision", shape=(N_positive + N_negative, 1), maxshape=(None, 1), dtype="int8")
    writer_index = 0
    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions, feat = policy(obs)
            # env stepping
            obs, _, _, infos = env.step(actions)

            # extract supervision data from infos
            supervision = infos["observations"]["auxiliary"]
            positive_indices = torch.nonzero(supervision[:, 0], as_tuple=False).squeeze(-1)
            negative_indices = torch.nonzero(supervision[:, 0] == 0, as_tuple=False).squeeze(-1)
            if len(positive_indices) > 0:
                if writer_index + len(positive_indices) >= N_positive + N_negative:
                    remaining_space = N_positive + N_negative - writer_index
                    positive_indices = positive_indices[:remaining_space]

                # store features and supervision in h5 file
                h5_file["features"][writer_index: writer_index + len(positive_indices)] = feat[positive_indices].cpu().numpy()
                h5_file["supervision"][writer_index: writer_index + len(positive_indices)] = supervision[positive_indices, 0].cpu().numpy().reshape(-1, 1)
                writer_index += len(positive_indices)
            if len(negative_indices) > 0:
                # store features and supervision in h5 file
                if len(negative_indices) > len(positive_indices):
                    negative_indices = negative_indices[:len(positive_indices)]
                if writer_index + len(negative_indices) >= N_positive + N_negative:
                    remaining_space = N_positive + N_negative - writer_index
                    negative_indices = negative_indices[:remaining_space]
                h5_file["features"][writer_index: writer_index + len(negative_indices)] = feat[negative_indices].cpu().numpy()
                h5_file["supervision"][writer_index: writer_index + len(negative_indices)] = supervision[negative_indices, 0].cpu().numpy().reshape(-1, 1)
                writer_index += len(negative_indices)
            if writer_index >= N_positive + N_negative:
                print("[INFO] H5 file is full, stopping data collection.")
                h5_file.close()
                break
            else:
                print(f"[INFO] Collected {writer_index} samples so far.")
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        
    # close the simulator
    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
