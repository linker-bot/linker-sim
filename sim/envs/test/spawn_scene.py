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
parser = argparse.ArgumentParser(description="Spawn test scene with AR5_L6 robot(s) and workspace table.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environment instances.")
parser.add_argument("--robot_side", type=str, default="left", choices=["left", "right", "both"], help="Robot side to spawn.")
parser.add_argument(
    "--reset_interval",
    type=int,
    default=600,
    help="Simulation steps between periodic robot resets.",
)
parser.add_argument(
    "--reset_envs_per_event",
    type=int,
    default=0,
    help="How many envs to reset each event (0 means reset all).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

from sim.assets import make_ar5_l6_robot_cfg
from sim.envs.test.scene_cfg import TestDualSceneCfg, TestSceneCfg


def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    robot_names: list[str],
    reset_interval: int = 600,
    reset_envs_per_event: int = 0,
) -> None:
    """Runs a basic simulation loop and periodically resets robot state by env ids."""
    robots = [scene[name] for name in robot_names]
    sim_dt = sim.get_physics_dt()
    step_count = 0
    all_env_ids = torch.arange(scene.num_envs, device=robots[0].data.default_joint_pos.device, dtype=torch.long)
    reset_cursor = 0

    def reset_robot_envs(env_ids: torch.Tensor) -> None:
        """Reset selected environments without recreating scene/simulation views."""
        for robot in robots:
            root_state = robot.data.default_root_state[env_ids].clone()
            root_state[:, :3] += scene.env_origins[env_ids]
            robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
            robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=env_ids)

            joint_pos = robot.data.default_joint_pos[env_ids].clone()
            joint_vel = robot.data.default_joint_vel[env_ids].clone()
            joint_pos += 0.02 * (torch.rand_like(joint_pos) - 0.5)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Reset per-env internal state (actuator/sensor buffers) only for the selected envs.
        scene.reset(env_ids.tolist())

    while simulation_app.is_running():
        if step_count % reset_interval == 0:
            step_count = 0
            if reset_envs_per_event <= 0 or reset_envs_per_event >= scene.num_envs:
                env_ids = all_env_ids
            else:
                env_ids = (torch.arange(reset_envs_per_event, device=all_env_ids.device) + reset_cursor) % scene.num_envs
                reset_cursor = int((reset_cursor + reset_envs_per_event) % scene.num_envs)
            reset_robot_envs(env_ids)
            print(f"[INFO] Reset robot state for env ids: {env_ids.tolist()}")

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        step_count += 1


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.8, -1.4, 1.2], target=[0.4, 0.0, 0.4])

    if args_cli.robot_side == "both":
        scene_cfg = TestDualSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
        robot_names = ["robot_left", "robot_right"]
    else:
        scene_cfg = TestSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
        # Keep side selection outside the scene dataclass fields, so InteractiveScene only sees asset cfgs.
        scene_cfg.robot = make_ar5_l6_robot_cfg(side=args_cli.robot_side, prim_path="{ENV_REGEX_NS}/Robot")
        robot_names = ["robot"]
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO] Test scene setup complete.")
    run_simulator(
        sim,
        scene,
        robot_names=robot_names,
        reset_interval=max(1, args_cli.reset_interval),
        reset_envs_per_event=max(0, args_cli.reset_envs_per_event),
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
