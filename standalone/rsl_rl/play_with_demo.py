"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--show_camera", action="store_true", default=False, help="Show camera of one environment.")
parser.add_argument("--use_auxiliary_head", action="store_true", default=False, help="Use auxiliary head for vision policy.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

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
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from standalone.rsl_rl.ext.utils import export_vision_policy_as_onnx
from isaaclab.sensors import RayCasterCamera
import cv2 as cv
# Import extensions to set up environment tasks
import diff.lab # noqa: F401
import diff.lab_tasks  # noqa: F401
from diff.lab.utils import visualize_depth

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
    if args_cli.use_auxiliary_head:
        print("[INFO] Using auxiliary head for vision policy.")
        decoder = ppo_runner.get_decoder()
    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    # export_policy_as_jit(
    #     ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    # )
    # export_policy_as_onnx(
    #     ppo_runner.alg.policy, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    # )
    export_vision_policy_as_onnx(
        ppo_runner.alg.policy,
        export_model_dir,
        ppo_runner.obs_normalizer,
        "vision_policy.onnx",
        False,
        env.unwrapped.scene['front_camera'].image_shape,
        (16,),
        use_auxiliary_head=args_cli.use_auxiliary_head,
    )

    # reset environment
    obs = env.get_observations()["policy"]
    timestep = 0
    prob = []
    actions_list = []
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions, feats = policy(obs)
            if args_cli.use_auxiliary_head:
                prob.append(decoder(feats).sigmoid()[0].item())
            # env stepping
            obs_dict, _, _, _ = env.step(actions)
            obs = obs_dict["policy"]
            actions_list.append(actions[0])
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        if args_cli.show_camera:
            camera: RayCasterCamera= env.unwrapped.scene['front_camera']
            depth = camera.data.output["distance_to_image_plane"]
            depth_vis = visualize_depth(depth[0, :, :, 0].cpu().numpy())
            if args_cli.use_auxiliary_head:
                # add probility text on the depth image
                prob_text = f"CP: {prob[-1]:.2f}"
                cv.putText(depth_vis, prob_text, (10, 10), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            cv.imshow("Depth (env 0)", depth_vis)
            cv.waitKey(1)
            if len(prob)==300:
                break
    # close the simulator
    env.close()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # plot the probability of the auxiliary head
    if args_cli.use_auxiliary_head:
        plt.plot(prob)
        plt.xlabel("Timestep")
        plt.ylabel("Probability")
        plt.title("Probability of the auxiliary head")
        plt.savefig(os.path.join(log_dir, "auxiliary_prob.png"))
        # plot actions
        actions_list = torch.stack(actions_list)
        plt.figure()
        plt.subplot(4, 1, 1)
        plt.plot((actions_list[:, 0].tanh() * 1.5 * 9.81 + 1.5 * 9.81).cpu().numpy(), label="thrust")
        plt.subplot(4, 1, 2)
        plt.plot((actions_list[:, 1].tanh() * 6).cpu().numpy(), label="wx")
        plt.subplot(4, 1, 3)
        plt.plot((actions_list[:, 2].tanh() * 6).cpu().numpy(), label="wy")
        plt.subplot(4, 1, 4)
        plt.plot((actions_list[:, 3].tanh() * 6).cpu().numpy(), label="wz")
        plt.title("Actions")
        plt.savefig(os.path.join(log_dir, "actions.png"))

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
