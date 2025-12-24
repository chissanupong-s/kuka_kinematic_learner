# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from isaaclab.app import AppLauncher
import math

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from original.asset.iiwa14 import KUKA_IIWA14_CFG

class IIWA14SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane",spawn=sim_utils.GroundPlaneCfg())

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0,color=(0.75, 0.75, 0.75))
    )

    IIWA14bot = KUKA_IIWA14_CFG.replace(prim_path="{ENV_REGEX_NS}/IIWA14bot")



def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    joint_target = torch.tensor([[
        math.radians(-85.0),
        math.radians(-33.7),
        math.radians(15.9),
        math.radians(90.2),
        math.radians(-14.8),
        math.radians(-69.3),
        math.radians(-92.0),
    ]], device=scene["IIWA14bot"].data.default_joint_pos.device)

    while simulation_app.is_running():
        # reset
        if count % 500 == 0:
            # reset counters
            count = 0
            # reset the scene entities to their initial positions offset by the environment origins
            root_state = scene["IIWA14bot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins

            # copy the default root state to the sim for the jetbot's orientation and velocity
            scene["IIWA14bot"].write_root_pose_to_sim(root_state[:, :7])
            scene["IIWA14bot"].write_root_velocity_to_sim(root_state[:, 7:])


            # copy the default joint states to the sim
            joint_pos, joint_vel = (
                scene["IIWA14bot"].data.default_joint_pos.clone(),
                scene["IIWA14bot"].data.default_joint_vel.clone(),
            )
            scene["IIWA14bot"].write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting IIWA14 Bot...")


        # wave
        # wave_action = scene["IIWA14bot"].data.default_joint_pos
        # wave_action[:, 0:4] = 0.25 * np.sin(2 * np.pi * 0.5 * sim_time)
        scene["IIWA14bot"].set_joint_position_target(joint_target)

        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # design scene
    scene_cfg = IIWA14SceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
