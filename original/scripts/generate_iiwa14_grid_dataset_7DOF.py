# generate_iiwa14_grid_dataset.py
#
# Generate a *static grid* kinematics dataset for KUKA iiwa14 (EE = robotiq_base_link)
# by sweeping joints from lower to upper limits in fixed angle steps.
#
# For each joint configuration, we record:
#   q1..q7, x, y, z, qw, qx, qy, qz
#
# Run example (for 6DOF config, 10° steps, limited samples):
#   ./isaaclab.sh -p original/scripts/generate_iiwa14_grid_dataset.py \
#       --num_envs 1 \
#       --step_deg 10.0 \
#       --max_samples 100000 \
#       --output_csv data/iiwa14_grid_6dof_10deg.csv \
#       --headless
#
# NOTE: Full joint-space grids get astronomically large very fast.
#       Start with small ranges / few joints / larger step_deg to test.

import argparse
import csv
import os
import math
from itertools import product

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------#
# 1. CLI + launch Isaac Sim
# ---------------------------------------------------------------------#

parser = argparse.ArgumentParser(
    description="Generate static joint-space grid dataset for KUKA iiwa14."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs (recommend 1).")
parser.add_argument(
    "--step_deg",
    type=float,
    default=5.0,
    help="Joint increment in degrees between grid samples.",
)
parser.add_argument(
    "--max_samples",
    type=int,
    default=None,
    help="Optional safety cap on number of samples (stops early if exceeded).",
)
parser.add_argument(
    "--output_csv",
    type=str,
    default="data/iiwa14_grid_dataset.csv",
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

# Choose which robot config you want (7DOF, 6DOF, 5DOF, etc.)
# For example:
#   robot_selection = KUKA_IIWA14_CFG        # full 7 DOF
#   robot_selection = KUKA_IIWA14_6DOF_CFG   # 6 DOF version
#   robot_selection = KUKA_IIWA14_5DOF_CFG   # 5 DOF version
robot_selection = KUKA_IIWA14_CFG  # <-- change this as needed
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
    IIWA14bot: ArticulationCfg = robot_selection.replace(
        prim_path="{ENV_REGEX_NS}/IIWA14bot"
    )

# ---------------------------------------------------------------------#
# 4. Grid dataset generation
# ---------------------------------------------------------------------#

def generate_grid_dataset(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    step_deg: float,
    max_samples: int,
    output_csv: str,
):
    sim_dt = sim.get_physics_dt()
    robot = scene["IIWA14bot"]

    num_envs = scene.num_envs
    print(f"[INFO] Number of envs: {num_envs}")
    if num_envs < 1:
        raise ValueError("num_envs must be >= 1")

    # We'll assume the first 7 joints are the arm DOFs
    num_arm_joints = 7

    # Joint limits: [num_envs, num_joints, 2]
    joint_limits = robot.data.soft_joint_pos_limits[:, :num_arm_joints].cpu().numpy()
    # Use env 0's limits as reference (should be same for all)
    limits_env0 = joint_limits[0]  # [7, 2]

    # Default joint states
    joint_pos_default = robot.data.default_joint_pos.clone()  # [num_envs, n_joints]
    joint_vel_default = robot.data.default_joint_vel.clone()

    # EE body index
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

    # Build per-joint grids in radians
    import math

    step_rad = math.radians(step_deg)
    joint_grids = []
    grid_sizes = []
    for j in range(num_arm_joints):
        lower = float(limits_env0[j, 0])
        upper = float(limits_env0[j, 1])
        if upper < lower:
            lower, upper = upper, lower

        num_steps = int(math.floor((upper - lower) / step_rad)) + 1
        values = lower + np.arange(num_steps, dtype=np.float32) * step_rad
        joint_grids.append(values)
        grid_sizes.append(len(values))
        print(f"[INFO] Joint {j+1}: range [{lower:.3f}, {upper:.3f}] rad, "
              f"step={step_rad:.3f}, points={len(values)}")

    # Total combinations in the grid
    total_combinations = 1
    for s in grid_sizes:
        total_combinations *= s
    print(f"[INFO] Theoretical total joint combinations: {total_combinations}")

    # Safety cap
    if max_samples is not None:
        max_global = min(total_combinations, max_samples)
        print(f"[INFO] max_samples={max_samples}, will generate at most {max_global} samples.")
    else:
        max_global = total_combinations

    # Helper: convert global index -> joint indices (mixed radix)
    def index_to_multi(idx, sizes):
        # sizes = [S0, S1, ..., S_{n-1}]
        indices = []
        for s in reversed(sizes):
            indices.append(idx % s)
            idx //= s
        indices.reverse()
        return indices  # list of length n

    # Number of "steps" needed (each step uses up to num_envs combinations)
    num_steps = (max_global + num_envs - 1) // num_envs  # ceil division

    rows = []
    joint_device = robot.data.joint_pos.device

    from tqdm import tqdm
    pbar = tqdm(total=max_global, desc="Collecting grid samples (multi-env)", ncols=100)

    sample_count = 0

    for step_idx in range(num_steps):
        if not simulation_app.is_running():
            print("[INFO] simulation_app closed; stopping early.")
            break

        # start from default joint states, modify per-env
        joint_pos = joint_pos_default.clone()

        # 1) assign a unique global index to each env
        valid_envs = []
        global_indices = []

        for env_id in range(num_envs):
            global_idx = step_idx * num_envs + env_id
            if global_idx >= max_global:
                break  # no more combinations
            valid_envs.append(env_id)
            global_indices.append(global_idx)

            # map global_idx -> per-joint indices
            idx_list = index_to_multi(global_idx, grid_sizes)
            q_vals = [joint_grids[j][idx_list[j]] for j in range(num_arm_joints)]
            q_np = np.array(q_vals, dtype=np.float32)

            joint_pos[env_id, :num_arm_joints] = torch.tensor(q_np, device=joint_device)

        if len(valid_envs) == 0:
            break  # nothing to do

        # 2) write joint positions to sim
        robot.write_joint_state_to_sim(joint_pos, joint_vel_default)
        scene.write_data_to_sim()

        # 3) step once
        sim.step()
        scene.update(sim_dt)

        # 4) read back data for each valid env
        for env_id, global_idx in zip(valid_envs, global_indices):
            q_now = robot.data.joint_pos[env_id, :num_arm_joints].cpu().numpy()
            ee_pos = robot.data.body_link_pos_w[env_id, ee_body_id].cpu().numpy()
            ee_quat = robot.data.body_link_quat_w[env_id, ee_body_id].cpu().numpy()

            row = np.concatenate(
                [
                    q_now,      # 7
                    ee_pos,     # 3
                    ee_quat,    # 4
                ]
            )
            rows.append(row)
            sample_count += 1
            pbar.update(1)

            if sample_count <= 3:
                print(
                    f"[DEBUG] sample {sample_count}: env={env_id}, global_idx={global_idx}, "
                    f"q[0]={q_now[0]:.3f}, pos=({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f})"
                )

        if sample_count >= max_global:
            break

    pbar.close()

    # Save CSV
    dirpath = os.path.dirname(output_csv)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    header = (
        [f"q{i+1}" for i in range(num_arm_joints)]
        + ["x", "y", "z"]
        + ["qw", "qx", "qy", "qz"]
    )

    import csv
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"[INFO] Saved {len(rows)} grid samples to '{output_csv}'")
# ---------------------------------------------------------------------#
# 5. Main
# ---------------------------------------------------------------------#

def main():
    # Create sim config on the requested device
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)

    # ------------------------------------------------------------------
    # Increase PhysX GPU buffers to avoid "patch buffer overflow" error
    # ------------------------------------------------------------------
    # Your error said: "please increase its size to at least 590600"
    # so we give it comfortable headroom.
    sim_cfg.physx.gpu_max_rigid_patch_count = max(
        sim_cfg.physx.gpu_max_rigid_patch_count, 1 << 22  # 4,194,304
    )
    # (optional but recommended alongside patch buffer)
    sim_cfg.physx.gpu_max_rigid_contact_count = max(
        sim_cfg.physx.gpu_max_rigid_contact_count, 1 << 23  # 8,388,608
    )

    sim = sim_utils.SimulationContext(sim_cfg)

    # Camera for non-headless runs
    sim.set_camera_view([3.5, 0.0, 3.0], [0.0, 0.0, 0.5])

    scene_cfg = IIWA14SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.reset()
    print("[INFO] Setup complete, starting grid data collection...")

    # Use grid generator instead of dynamic random motion
    generate_grid_dataset(
        sim=sim,
        scene=scene,
        step_deg=args_cli.step_deg,
        max_samples=args_cli.max_samples,
        output_csv=args_cli.output_csv,
    )



if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
