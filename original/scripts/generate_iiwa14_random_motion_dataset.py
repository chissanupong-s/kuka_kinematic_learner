# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Generate a random motion dataset for KUKA iiwa14.

Example:
    ./isaaclab.sh -p original/scripts/generate_iiwa14_random_motion_dataset.py \
        --num_envs 1 \
        --num_samples 100000 \
        --output_csv data/iiwa14_random_motion.csv \
        --headless
"""

import argparse
import csv
import os

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------#
# 1. CLI + start Isaac Sim (AppLauncher *before* importing isaaclab.sim)
# ---------------------------------------------------------------------#

parser = argparse.ArgumentParser(
    description="Generate random motion dataset for KUKA iiwa14."
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of envs (usually 1 for dataset)."
)
parser.add_argument(
    "--num_samples",
    type=int,
    default=10000,
    help="Number of samples (rows) to record.",
)
parser.add_argument(
    "--output_csv",
    type=str,
    default="data/iiwa14_random_motion.csv",
    help="Output CSV file path.",
)

# add AppLauncher arguments (device, headless, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch Omniverse app (this loads omni.client and friends)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------#
# 2. Now import heavy Isaac Lab / Isaac Sim stuff
# ---------------------------------------------------------------------#

import numpy as np
import torch
from tqdm import tqdm  # progress bar

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from original.asset.iiwa14 import KUKA_IIWA14_CFG

# ---------------------------------------------------------------------#
# 3. Scene configuration
# ---------------------------------------------------------------------#


@configclass
class IIWA14SceneCfg(InteractiveSceneCfg):
    # Ground
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Light
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(0.75, 0.75, 0.75),
        ),
    )

    # Robot
    IIWA14bot: ArticulationCfg = KUKA_IIWA14_CFG.replace(
        prim_path="{ENV_REGEX_NS}/IIWA14bot"
    )


# ---------------------------------------------------------------------#
# 4. Dataset generation loop
# ---------------------------------------------------------------------#


def generate_dataset(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    num_samples: int,
    output_csv: str,
):
    """Run sim, move joints randomly, record EE pose/vel/acc and joint angles."""

    sim_dt = sim.get_physics_dt()
    robot = scene["IIWA14bot"]

    num_envs = scene.num_envs
    assert num_envs == 1, "Dataset script currently assumes num_envs == 1."

    # We assume first 7 joints are arm joints (ignore gripper if present)
    num_arm_joints = 7

    # Joint limits: shape (num_envs, num_joints, 2)
    joint_limits = robot.data.soft_joint_pos_limits[0, :num_arm_joints]  # (7, 2)
    joint_default = robot.data.default_joint_pos[:, :num_arm_joints]

    # Find EE body index for "robotiq_base_link"
    ee_entity_cfg = SceneEntityCfg(
        name="IIWA14bot",
        body_names=["robotiq_base_link"],
    )
    ee_entity_cfg.resolve(scene)
    ee_body_id = ee_entity_cfg.body_ids[0]

    print(f"[INFO] Using EE body 'robotiq_base_link' with index {ee_body_id}")

    # For random motion we change target every N steps
    steps_per_target = 200

    # Storage
    rows = []

    # Initial target
    joint_target = joint_default.clone()

    # Progress bar
    pbar = tqdm(total=num_samples, desc="Collecting samples", ncols=90)

    step_count = 0

    # Main collection loop: exactly num_samples iterations (unless user closes app)
    for _ in range(num_samples):
        # Allow user to abort by closing the app window
        if not simulation_app.is_running():
            print("[INFO] simulation_app no longer running, aborting early.")
            break

        # Every few steps, sample a new random joint target within limits
        if step_count % steps_per_target == 0:
            low = joint_limits[:, 0]
            high = joint_limits[:, 1]
            rand = torch.rand_like(joint_target[:, :num_arm_joints])
            joint_target[:, :num_arm_joints] = low + (high - low) * rand

            # (optional) keep gripper at default if there are extra joints
            if robot.data.default_joint_pos.shape[1] > num_arm_joints:
                joint_target[:, num_arm_joints:] = robot.data.default_joint_pos[
                    :, num_arm_joints:
                ]

            # Set PD-style targets (position + zero velocity)
            robot.set_joint_position_target(joint_target)
            zero_vel = torch.zeros_like(robot.data.default_joint_vel)
            robot.set_joint_velocity_target(zero_vel)

        # Step sim and update scene
        sim.step()
        scene.update(sim_dt)
        step_count += 1

        # Current joint positions (arm only)
        q = robot.data.joint_pos[:, :num_arm_joints].cpu().numpy()[0]

        # EE pose in world frame
        ee_pos = robot.data.body_link_pos_w[:, ee_body_id].cpu().numpy()[0]  # (3,)
        ee_quat = robot.data.body_link_quat_w[:, ee_body_id].cpu().numpy()[0]  # (4,)

        # EE velocities (world frame)
        ee_lin_vel = (
            robot.data.body_link_lin_vel_w[:, ee_body_id].cpu().numpy()[0]
        )  # (3,)
        ee_ang_vel = (
            robot.data.body_link_ang_vel_w[:, ee_body_id].cpu().numpy()[0]
        )  # (3,)

        # EE accelerations (world frame, from COM data)
        ee_lin_acc = (
            robot.data.body_com_lin_acc_w[:, ee_body_id].cpu().numpy()[0]
        )  # (3,)
        ee_ang_acc = (
            robot.data.body_com_ang_acc_w[:, ee_body_id].cpu().numpy()[0]
        )  # (3,)

        row = np.concatenate(
            [
                q,  # 7
                ee_pos,  # 3
                ee_quat,  # 4
                ee_lin_vel,  # 3
                ee_ang_vel,  # 3
                ee_lin_acc,  # 3
                ee_ang_acc,  # 3
            ]
        )
        rows.append(row)
        pbar.update(1)

    pbar.close()

    # Save CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    header = (
        [f"q{i+1}" for i in range(num_arm_joints)]
        + ["x", "y", "z"]
        + ["qw", "qx", "qy", "qz"]
        + ["vx", "vy", "vz"]
        + ["wx", "wy", "wz"]
        + ["ax", "ay", "az"]
        + ["alphax", "alphay", "alphaz"]
    )

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"[INFO] Saved {len(rows)} samples to '{output_csv}'")


# ---------------------------------------------------------------------#
# 5. Main
# ---------------------------------------------------------------------#


def main():
    # Simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    # Nice camera if not headless
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

    # Scene
    scene_cfg = IIWA14SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # Start sim
    sim.reset()
    print("[INFO] Setup complete, starting random-motion data collection...")

    generate_dataset(sim, scene, args_cli.num_samples, args_cli.output_csv)


if __name__ == "__main__":
    try:
        main()
    finally:
        # Always close the app so the CLI returns
        simulation_app.close()
