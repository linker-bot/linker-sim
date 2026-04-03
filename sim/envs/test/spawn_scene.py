from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# Ensure repo root is on sys.path when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Parse CLI first and launch app before importing Isaac Sim-dependent modules.
parser = argparse.ArgumentParser(description="Spawn test scene with AR5_L6 left robot and workspace table.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environment instances.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

from sim.envs.test.scene_cfg import TestSceneCfg


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    """Runs a basic simulation loop and periodically resets robot state."""
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()
    step_count = 0

    while simulation_app.is_running():
        if step_count % 600 == 0:
            step_count = 0
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])

            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            joint_pos += 0.02 * (torch.rand_like(joint_pos) - 0.5)
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            scene.reset()
            print("[INFO] Reset robot state.")

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        step_count += 1


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.8, -1.4, 1.2], target=[0.4, 0.0, 0.4])

    scene_cfg = TestSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO] Test scene setup complete.")
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
