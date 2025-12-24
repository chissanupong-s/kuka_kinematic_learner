# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
# Play a trained skrl agent, but the user chooses the end-effector target
# via the "ee_pose" command defined in 6DOF_env_cfg.py.

import argparse
import os
import sys
import time
from typing import Sequence

import gymnasium as gym
import numpy as np
import torch



from isaaclab.app import AppLauncher
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils.hydra import hydra_task_config

import skrl
from skrl.utils.omniverse_isaaclab_utils import (
    get_checkpoint_path,
    get_published_pretrained_checkpoint,
)
from skrl.utils.runner import Runner

import original.tasks  # noqa: F401  # important to register your custom tasks


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Play trained PPO agent with manual EE goal (ee_pose command)"
)
parser.add_argument(
    "--task",
    type=str,
    required=True,
    help="Task name, e.g. original.tasks.manager_based.6DOF_env_cfg:OriginalEnvCfg_PLAY",
)
parser.add_argument(
    "--algorithm", type=str, default="ppo", help="RL algorithm (default: ppo)"
)
parser.add_argument(
    "--ml_framework", type=str, default="torch", help="ML backend (torch / jax)"
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of environments (use 1 for goal teleoperation).",
)
parser.add_argument(
    "--device",
    type=str,
    default=None,
    help="Simulation device override, e.g. cuda:0 or cpu.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="best",
    help=(
        "Checkpoint path or shortcut: "
        "'best' (default) / 'last' within experiment dir, "
        "or an absolute path to a .pt file."
    ),
)
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    default=False,
    help="Use a published pretrained checkpoint (if available for this task).",
)
parser.add_argument(
    "--real-time",
    action="store_true",
    default=False,
    help="Try to run approximately in real-time.",
)
parser.add_argument(
    "--video", action="store_true", default=False, help="Record video of rollout."
)
parser.add_argument(
    "--video_length", type=int, default=500, help="Video length (steps) when recording."
)

# Add AppLauncher args (e.g. --headless, --enable_cameras)
AppLauncher.add_app_launcher_args(parser)

# Parse CLI; keep hydra args separate
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
# Give hydra a clean argv
sys.argv = [sys.argv[0]] + hydra_args

# Launch Omniverse / Isaac app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Helper: Ask goal & set ee_pose command
# -----------------------------------------------------------------------------

def ask_goal_from_user(prev: Sequence[float] | None = None) -> list[float]:
    """Ask the user for an EE target [x, y, z]."""
    if prev is None:
        print("\nEnter desired EE target position [x, y, z] in meters.")
    else:
        print(f"\nCurrent goal: {prev}")

    raw = input("Goal xyz (comma-separated, blank = keep current): ").strip()
    if not raw and prev is not None:
        return list(prev)

    try:
        vals = [float(x) for x in raw.split(",")]
        if len(vals) != 3:
            raise ValueError("need exactly 3 numbers (x, y, z)")
        return vals
    except Exception as exc:
        print(f"[WARN] Could not parse goal ({exc}), keeping previous.")
        return list(prev) if prev is not None else [0.5, 0.0, 0.4]


def set_ee_pose_goal(env, goal_xyz: Sequence[float]) -> None:
    """Override the 'ee_pose' command with a custom [x, y, z] goal.

    This uses the command manager:

        env.command_manager.get_command("ee_pose")

    For a UniformPoseCommand, the command tensor is typically:
        [x, y, z, qw, qx, qy, qz]  (shape: [num_envs, 7])

    We modify only the position (0:3) and keep the orientation as-is.
    """
    # get the command tensor [num_envs, 7] (position + quaternion)
    cmd = env.command_manager.get_command("ee_pose")  # type: torch.Tensor

    # build position tensor
    pos = torch.as_tensor(goal_xyz, dtype=cmd.dtype, device=cmd.device)
    if pos.ndim == 1:
        pos = pos.unsqueeze(0)  # [1, 3]

    # broadcast to all envs
    pos = pos.expand(cmd.shape[0], 3)
    cmd[:, 0:3] = pos  # in-place update of the command buffer


# config shortcuts for hydra/skrl
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = (
    "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
)


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Play with skrl agent, controlling EE goal via ee_pose."""
    # ------------------------------------------------------------------
    # Override configs from CLI
    # ------------------------------------------------------------------
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    print("[INFO] Environment configuration:")
    print_dict(env_cfg)

    # Configure ML backend if needed
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    task_name = args_cli.task.split(":")[-1]

    # ------------------------------------------------------------------
    # Resolve checkpoint path
    # ------------------------------------------------------------------
    log_root_path = os.path.join(
        "logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"]
    )
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", task_name)
        if not resume_path:
            print("[ERROR] No published pretrained checkpoint found for this task.")
            return

    else:
        # Use explicit path or infer from experiment directory
        if args_cli.checkpoint in ["best", "last"]:
            # infer base checkpoint path
            base = get_checkpoint_path(
                log_root_path,
                run_dir=f".*_{algorithm}_{args_cli.ml_framework}",
                other_dirs=["checkpoints"],
            )
            if not base:
                print("[ERROR] Could not infer checkpoint path from logs.")
                return
            ckpt_dir = os.path.dirname(base)
            fname = "best_agent.pt" if args_cli.checkpoint == "best" else "agent.pt"
            resume_path = os.path.join(ckpt_dir, fname)
        else:
            resume_path = os.path.abspath(args_cli.checkpoint)

    if not os.path.isfile(resume_path):
        print(f"[ERROR] Checkpoint not found: {resume_path}")
        return

    print(f"[INFO] Using checkpoint: {resume_path}")
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # ------------------------------------------------------------------
    # Make environment
    # ------------------------------------------------------------------
    env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    # convert multi-agent to single-agent for PPO if needed
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # simulation timestep for real-time sleep
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # optional video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "episode_trigger": lambda episode_id: True,
            "step_trigger": None,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video with settings:")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # ------------------------------------------------------------------
    # Create runner & load agent (same as original play.py)
    # ------------------------------------------------------------------
    distributed = experiment_cfg["agent"]["rewards"]["distributed"]
    isaac_gym = env.spec is not None and env.spec.entry_point == "isaacgymenvs.make:make"
    if not distributed and not isaac_gym:
        # enable distributed rewards by default if needed
        experiment_cfg["agent"]["rewards"]["distributed"] = True

    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    runner.agent.set_running_mode("eval")

    # disable further logging/checkpointing while playing
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    # ------------------------------------------------------------------
    # Play loop with goal control via ee_pose
    # ------------------------------------------------------------------
    obs, _ = env.reset()
    timestep = 0

    # initial goal
    goal = ask_goal_from_user(prev=[0.5, 0.0, 0.3])
    set_ee_pose_goal(env, goal)

    while simulation_app.is_running():
        start_time = time.time()

        # every N steps, allow user to adjust goal
        if timestep % 200 == 0:
            goal = ask_goal_from_user(prev=goal)
            set_ee_pose_goal(env, goal)

        # agent inference
        with torch.inference_mode():
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            if hasattr(env, "possible_agents"):
                actions = {
                    a: outputs[-1][a].get("mean_actions", outputs[0][a])
                    for a in env.possible_agents
                }
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])

        # environment step
        obs, _, terminated, truncated, _ = env.step(actions)

        # reset if done
        if np.any(terminated) or np.any(truncated):
            obs, _ = env.reset()
            set_ee_pose_goal(env, goal)

        timestep += 1

        # stop after one video if recording
        if args_cli.video and timestep >= args_cli.video_length:
            break

        # approximate real-time
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
