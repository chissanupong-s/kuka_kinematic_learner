# Generate a dynamic motion dataset for KUKA iiwa14 (EE = robotiq_base_link)
# using multiple parallel environments.
#
# Each env:
#   - is driven by joint drives via set_joint_position_target
#   - gets its own random joint-space targets
#
# For each sim step, we log one row per env:
#   q1..q7, qd1..qd7,
#   x,y,z, qw,qx,qy,qz,
#   vx,vy,vz, wx,wy,wz,
#   ax,ay,az, alphax,alphay,alphaz
#
# Total rows in CSV = num_envs * num_samples.
#
# Run example:
#   ./isaaclab.sh -p original/scripts/generate_iiwa14_dynamic_dataset_multi_env.py \
#       --num_envs 3 \
#       --num_samples 10000 \
#       --output_csv data/iiwa14_dynamic_multi_env.csv \
#       --headless

import argparse
import csv
import os

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------#
# 1. CLI + launch Isaac Sim
# ---------------------------------------------------------------------#

parser = argparse.ArgumentParser(
    description="Generate dynamic motion dataset for KUKA iiwa14 (multi-env)."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs.")
parser.add_argument(
    "--num_samples",
    type=int,
    default=10000,
    help="Number of samples PER ENV.",
)
parser.add_argument(
    "--output_csv",
    type=str,
    default="data/iiwa14_dynamic_multi_env.csv",
    help="Output CSV file path.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------#
# 2. Heavy imports (after app start)
# ---------------------------------------------------------------------#

import numpy as np
import torch
from tqdm import tqdm

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from original.asset.iiwa14 import KUKA_IIWA14_CFG, KUKA_IIWA14_6DOF_CFG, KUKA_IIWA14_5DOF_CFG

robot_selection = KUKA_IIWA14_5DOF_CFG

# ---------------------------------------------------------------------#
# 3. Scene config
# ---------------------------------------------------------------------#

@configclass
class IIWA14SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(0.75, 0.75, 0.75),
        ),
    )
    # IIWA14bot: ArticulationCfg = KUKA_IIWA14_CFG.replace(
    #     prim_path="{ENV_REGEX_NS}/IIWA14bot"
    # )
    IIWA14bot: ArticulationCfg = robot_selection.replace(
        prim_path="{ENV_REGEX_NS}/IIWA14bot"
    )

# ---------------------------------------------------------------------#
# 4. Dataset generation (multi-env, joint drives)
# ---------------------------------------------------------------------#

def generate_dataset(sim: sim_utils.SimulationContext,
                     scene: InteractiveScene,
                     num_samples_per_env: int,
                     output_csv: str):

    sim_dt = sim.get_physics_dt()
    robot = scene["IIWA14bot"]

    num_envs = scene.num_envs
    print(f"[INFO] Number of envs: {num_envs}")

    # Assume first 7 joints are arm DOFs
    num_arm_joints = 7

    # Joint limits: [num_envs, num_joints, 2]
    joint_limits = robot.data.soft_joint_pos_limits[:, :num_arm_joints].cpu().numpy()

    # Default joint states
    joint_pos_default = robot.data.default_joint_pos.clone()  # [num_envs, n_joints]
    joint_vel_default = robot.data.default_joint_vel.clone()

    # Resolve EE body index (same for all envs)
    ee_entity_cfg = SceneEntityCfg(
        name="IIWA14bot", body_names=["robotiq_base_link"]
    )
    ee_entity_cfg.resolve(scene)
    ee_body_id = ee_entity_cfg.body_ids[0]
    print(f"[INFO] Using EE body 'robotiq_base_link' with index {ee_body_id}")

    # Reset robots to default
    robot.write_joint_state_to_sim(joint_pos_default, joint_vel_default)

    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])

    scene.write_data_to_sim()
    sim.step()
    scene.update(sim_dt)

    # How many sim steps to keep a given target before sampling a new one
    steps_per_target = 30

    # Target tensor: [num_envs, n_joints]
    joint_target_full = robot.data.default_joint_pos.clone()

    # Total samples = per-env * num_envs
    total_samples = num_samples_per_env * num_envs
    rows = []
    pbar = tqdm(total=total_samples, desc="Collecting dynamic samples", ncols=100)

    step_count = 0

    for step in range(num_samples_per_env):
        if not simulation_app.is_running():
            print("[INFO] simulation_app closed; stopping early.")
            break

        # Every few steps, sample new targets for ALL envs
        if step_count % steps_per_target == 0:
            # For each env, sample its own arm target within its joint limits
            for env_id in range(num_envs):
                lower = joint_limits[env_id, :, 0]
                upper = joint_limits[env_id, :, 1]
                q_target_np = lower + (upper - lower) * np.random.rand(num_arm_joints).astype(np.float32)
                q_target_arm = torch.tensor(
                    q_target_np, device=robot.data.joint_pos.device
                )
                joint_target_full[env_id, :num_arm_joints] = q_target_arm

        # Apply joint position targets (uses drives)
        robot.set_joint_position_target(joint_target_full)

        # Push targets & states into sim
        scene.write_data_to_sim()

        # Step physics
        sim.step()
        scene.update(sim_dt)
        step_count += 1

        # Record one row per env
        for env_id in range(num_envs):
            q_now = robot.data.joint_pos[env_id, :num_arm_joints].cpu().numpy()
            qd_now = robot.data.joint_vel[env_id, :num_arm_joints].cpu().numpy()

            ee_pos = robot.data.body_link_pos_w[env_id, ee_body_id].cpu().numpy()
            ee_quat = robot.data.body_link_quat_w[env_id, ee_body_id].cpu().numpy()

            ee_lin_vel = robot.data.body_link_lin_vel_w[env_id, ee_body_id].cpu().numpy()
            ee_ang_vel = robot.data.body_link_ang_vel_w[env_id, ee_body_id].cpu().numpy()

            ee_lin_acc = robot.data.body_com_lin_acc_w[env_id, ee_body_id].cpu().numpy()
            ee_ang_acc = robot.data.body_com_ang_acc_w[env_id, ee_body_id].cpu().numpy()

            row = np.concatenate(
                [
                    q_now,        # 7
                    qd_now,       # 7
                    ee_pos,       # 3
                    ee_quat,      # 4
                    ee_lin_vel,   # 3
                    ee_ang_vel,   # 3
                    ee_lin_acc,   # 3
                    ee_ang_acc,   # 3
                ]
            )
            rows.append(row)

        # update progress by num_envs samples per step
        pbar.update(num_envs)

        # optional tiny debug
        if step < 3:
            print(
                f"[DEBUG] step {step}: env0 q[0]={rows[-num_envs][0]:.3f}, "
                f"|v_lin|={np.linalg.norm(rows[-num_envs][7+7+3+4:7+7+3+4+3]):.4f}"
            )

    pbar.close()

    # Save CSV
    dirpath = os.path.dirname(output_csv)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    header = (
        [f"q{i+1}" for i in range(num_arm_joints)]
        + [f"qd{i+1}" for i in range(num_arm_joints)]
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

    print(f"[INFO] Saved {len(rows)} samples to '{output_csv}' "
          f"(num_envs={num_envs}, num_samples_per_env={num_samples_per_env})")


# ---------------------------------------------------------------------#
# 5. Main
# ---------------------------------------------------------------------#

def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    # Camera for non-headless runs
    sim.set_camera_view([3.5, 0.0, 3.0], [0.0, 0.0, 0.5])

    scene_cfg = IIWA14SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.reset()
    print("[INFO] Setup complete, starting dynamic data collection...")
    generate_dataset(sim, scene, args_cli.num_samples, args_cli.output_csv)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
