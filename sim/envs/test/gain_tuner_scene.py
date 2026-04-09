"""Launch the test scene with Isaac Sim's Gain Tuner extension enabled.

This is the supported workflow for interactively tuning **stiffness** and **damping**
(PhysX joint drive gains) on the AR5 + L6 articulation.

It supports two tuning paths:
1. Isaac Sim native Gain Tuner UI (enabled via extension).
2. Live gain hot-reload from ``sim/envs/test/joint_gains.json`` (no app restart).

It also supports optional per-joint feed-forward torque offsets via ``offset`` in
the same JSON payload:
  tau_cmd = tau_pd(implicit) + offset

Usage (from repo root, with Isaac Lab / Isaac Sim on ``PYTHONPATH``)::

    python sim/envs/test/gain_tuner_scene.py --num_envs 1 --robot_side left

Do **not** pass ``--headless`` — the Gain Tuner is a GUI panel.

Then in Isaac Sim: open the Gain Tuner (e.g. **Tools → Robotics → Gain Tuner**, or
search **Window → Extensions** for ``isaacsim.robot_setup.gain_tuner``), select your
robot articulation on the stage, and tune drives. See also NVIDIA's tutorial:
https://docs.isaacsim.omniverse.nvidia.com/latest/robot_setup_tutorials/joint_tuning.html

Extension id may differ slightly across Isaac Sim versions; override with
``--gain_tuner_ext`` if needed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser(
    description=(
        "Spawn the AR5/L6 test scene with Isaac Sim Gain Tuner enabled. "
        "Run without --headless. After tuning, update AR5_L6_* PD constants in sim/assets/robots.py."
    )
)
parser.add_argument("--num_envs", type=int, default=1, help="Use 1 while tuning in the GUI.")
parser.add_argument("--robot_side", type=str, default="left", choices=["left", "right", "both"])
parser.add_argument(
    "--enable_gain_tuner",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Prepend Kit args to enable the Gain Tuner extension (default: true).",
)
parser.add_argument(
    "--gain_tuner_ext",
    type=str,
    default="isaacsim.robot_setup.gain_tuner",
    help="Extension id passed to Kit as `--enable <id>`.",
)
parser.add_argument(
    "--allow_headless",
    action="store_true",
    help="Skip the headless guard. The Gain Tuner UI will not be usable in headless mode.",
)
parser.add_argument(
    "--passive_command_stream",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "When true (default), do NOT push Isaac Lab articulation commands every frame. "
        "This lets Gain Tuner own joint targets. Use --no-passive_command_stream only "
        "for debugging."
    ),
)
parser.add_argument(
    "--gains_file",
    type=str,
    default=str((REPO_ROOT / "sim/envs/test/joint_gains.json").resolve()),
    help="JSON file containing per-joint stiffness/damping values.",
)
parser.add_argument(
    "--reload_gains",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Poll gains_file and apply updates at runtime (default: true).",
)
parser.add_argument(
    "--gains_reload_period_s",
    type=float,
    default=0.5,
    help="Polling period for gains hot-reload.",
)
parser.add_argument(
    "--print_joint_effort",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Print realtime per-joint tau_cmd_est to console.",
)
parser.add_argument(
    "--joint_effort_print_hz",
    type=float,
    default=5.0,
    help="Console print rate for joint efforts.",
)
parser.add_argument(
    "--joint_effort_env_id",
    type=int,
    default=0,
    help="Environment index to print joint effort from.",
)
parser.add_argument(
    "--joint_effort_arm_only",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Print only AR5 arm joints (default: true).",
)
parser.add_argument(
    "--joint_effort_joint_names",
    type=str,
    default="",
    help=(
        "Optional comma-separated joint names to print only those joints. "
        "Example: AR5_5_07L_W4C4A2_joint_1 or "
        "AR5_5_07L_W4C4A2_joint_1,AR5_5_07L_W4C4A2_joint_2"
    ),
)
parser.add_argument(
    "--disable_gravity",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Set world gravity to zero for quick testing.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.headless and not args_cli.allow_headless:
    parser.error(
        "Gain Tuner requires a GUI: run without --headless (it is off by default). "
        "If you only want a headless sim, use sim/envs/test/spawn_scene.py instead, "
        "or pass --allow_headless to suppress this error."
    )

if args_cli.enable_gain_tuner:
    enable_flag = f"--enable {args_cli.gain_tuner_ext}"
    existing = getattr(args_cli, "kit_args", None) or ""
    args_cli.kit_args = f"{enable_flag} {existing}".strip()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
import torch
from isaaclab.scene import InteractiveScene

from sim.assets import make_ar5_l6_robot_cfg
from sim.assets.robots import (
    AR5_L6_LEFT_ARM_DAMPING_MAP,
    AR5_L6_LEFT_ARM_STIFFNESS_MAP,
    AR5_L6_LEFT_HAND_DAMPING_MAP,
    AR5_L6_LEFT_HAND_STIFFNESS_MAP,
    AR5_L6_RIGHT_ARM_DAMPING_MAP,
    AR5_L6_RIGHT_ARM_STIFFNESS_MAP,
    AR5_L6_RIGHT_HAND_DAMPING_MAP,
    AR5_L6_RIGHT_HAND_STIFFNESS_MAP,
)
from sim.envs.test.scene_cfg import TestDualSceneCfg, TestSceneCfg


def _default_joint_gains() -> dict[str, dict[str, float]]:
    defaults: dict[str, dict[str, float]] = {}
    for joint, value in {
        **AR5_L6_LEFT_ARM_STIFFNESS_MAP,
        **AR5_L6_LEFT_HAND_STIFFNESS_MAP,
        **AR5_L6_RIGHT_ARM_STIFFNESS_MAP,
        **AR5_L6_RIGHT_HAND_STIFFNESS_MAP,
    }.items():
        defaults.setdefault(joint, {})["stiffness"] = float(value)
    for joint, value in {
        **AR5_L6_LEFT_ARM_DAMPING_MAP,
        **AR5_L6_LEFT_HAND_DAMPING_MAP,
        **AR5_L6_RIGHT_ARM_DAMPING_MAP,
        **AR5_L6_RIGHT_HAND_DAMPING_MAP,
    }.items():
        defaults.setdefault(joint, {})["damping"] = float(value)
    return defaults


def _ensure_gains_file_exists(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _default_joint_gains()
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote default gains file: {path}")


def _load_joint_gains(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level object in {path}, got: {type(payload)}")
    parsed: dict[str, dict[str, float]] = {}
    for joint_name, gains in payload.items():
        if not isinstance(gains, dict):
            raise ValueError(f"Expected object at key '{joint_name}', got: {type(gains)}")
        if "stiffness" not in gains or "damping" not in gains:
            raise ValueError(f"Joint '{joint_name}' must define both 'stiffness' and 'damping'")
        parsed[joint_name] = {
            "stiffness": float(gains["stiffness"]),
            "damping": float(gains["damping"]),
            # Optional additive feed-forward offset (torque/force depending on joint type).
            "offset": float(gains.get("offset", 0.0)),
        }
    return parsed


def _apply_joint_gains_from_map(robot, joint_gains: dict[str, dict[str, float]]) -> None:
    joint_names = list(robot.joint_names)
    num_envs, num_joints = robot.data.default_joint_pos.shape
    stiffness = torch.zeros((num_envs, num_joints), device=robot.data.default_joint_pos.device)
    damping = torch.zeros((num_envs, num_joints), device=robot.data.default_joint_pos.device)

    missing = [name for name in joint_names if name not in joint_gains]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(
            f"Gains file missing {len(missing)} joint(s) for this articulation: {missing_str}"
        )

    for idx, name in enumerate(joint_names):
        gains = joint_gains[name]
        stiffness[:, idx] = gains["stiffness"]
        damping[:, idx] = gains["damping"]

    if not hasattr(robot, "write_joint_stiffness_to_sim") or not hasattr(
        robot, "write_joint_damping_to_sim"
    ):
        raise RuntimeError(
            "Articulation does not expose write_joint_stiffness_to_sim/write_joint_damping_to_sim"
        )

    robot.write_joint_stiffness_to_sim(stiffness)
    robot.write_joint_damping_to_sim(damping)


def _build_joint_offset_efforts(robot, joint_gains: dict[str, dict[str, float]]) -> torch.Tensor:
    """Build per-env effort tensor from per-joint offset values."""
    joint_names = list(robot.joint_names)
    num_envs, num_joints = robot.data.default_joint_pos.shape
    efforts = torch.zeros((num_envs, num_joints), device=robot.data.default_joint_pos.device)
    for idx, name in enumerate(joint_names):
        efforts[:, idx] = float(joint_gains[name]["offset"])
    return efforts


def _build_joint_stiffness_tensor(robot, joint_gains: dict[str, dict[str, float]]) -> torch.Tensor:
    joint_names = list(robot.joint_names)
    num_envs, num_joints = robot.data.default_joint_pos.shape
    stiffness = torch.zeros((num_envs, num_joints), device=robot.data.default_joint_pos.device)
    for idx, name in enumerate(joint_names):
        stiffness[:, idx] = float(joint_gains[name]["stiffness"])
    return stiffness


def _build_joint_damping_tensor(robot, joint_gains: dict[str, dict[str, float]]) -> torch.Tensor:
    joint_names = list(robot.joint_names)
    num_envs, num_joints = robot.data.default_joint_pos.shape
    damping = torch.zeros((num_envs, num_joints), device=robot.data.default_joint_pos.device)
    for idx, name in enumerate(joint_names):
        damping[:, idx] = float(joint_gains[name]["damping"])
    return damping


def _apply_runtime_gains(
    robots: dict[str, object], gains_path: Path
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    joint_gains = _load_joint_gains(gains_path)
    stiffness_by_robot: dict[str, torch.Tensor] = {}
    damping_by_robot: dict[str, torch.Tensor] = {}
    offset_efforts_by_robot: dict[str, torch.Tensor] = {}
    for robot_name, robot in robots.items():
        _apply_joint_gains_from_map(robot, joint_gains)
        stiffness_by_robot[robot_name] = _build_joint_stiffness_tensor(robot, joint_gains)
        damping_by_robot[robot_name] = _build_joint_damping_tensor(robot, joint_gains)
        offset_efforts_by_robot[robot_name] = _build_joint_offset_efforts(robot, joint_gains)
        print(f"[INFO] Applied runtime gains to {robot_name}")
    return stiffness_by_robot, damping_by_robot, offset_efforts_by_robot


def _read_position_target_tensor(robot) -> torch.Tensor:
    # Prefer direct PhysX position targets if available.
    getter = getattr(robot.root_physx_view, "get_dof_position_targets", None)
    if callable(getter):
        targets = getter()
        if isinstance(targets, torch.Tensor):
            return targets
        return torch.tensor(targets, device=robot.data.joint_pos.device)
    # Fallback to IsaacLab command buffer.
    return robot.data.joint_pos_target


def _format_tau_cmd_est_line(
    robot_name: str,
    robot,
    stiffness: torch.Tensor,
    damping: torch.Tensor,
    offset_efforts: torch.Tensor,
    env_id: int,
    arm_only: bool,
    selected_joint_names: set[str] | None,
) -> str:
    joint_names = list(robot.joint_names)
    q = robot.data.joint_pos[env_id]
    qd = robot.data.joint_vel[env_id]
    q_target = _read_position_target_tensor(robot)[env_id]
    qd_target = torch.zeros_like(qd)
    tau_cmd_est = (
        stiffness[env_id] * (q_target - q)
        + damping[env_id] * (qd_target - qd)
        + offset_efforts[env_id]
    )
    tau_cmd_est = tau_cmd_est.detach().cpu()

    chunks: list[str] = []
    for idx, name in enumerate(joint_names):
        if selected_joint_names is not None and name not in selected_joint_names:
            continue
        if arm_only and "_joint_" not in name:
            continue
        chunks.append(f"{name}: tau_cmd_est={tau_cmd_est[idx]:+.3f}")
    return f"[TAU_CMD_EST][{robot_name}] " + " | ".join(chunks)


def main() -> None:
    if args_cli.disable_gravity:
        sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, gravity=(0.0, 0.0, 0.0))
    else:
        sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.8, -1.4, 1.2], target=[0.4, 0.0, 0.4])

    if args_cli.robot_side == "both":
        scene_cfg = TestDualSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
        robot_names = ["robot_left", "robot_right"]
    else:
        scene_cfg = TestSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
        scene_cfg.robot = make_ar5_l6_robot_cfg(
            side=args_cli.robot_side, prim_path="{ENV_REGEX_NS}/Robot"
        )
        robot_names = ["robot"]

    scene = InteractiveScene(scene_cfg)
    sim.reset()
    robots = {name: scene[name] for name in robot_names}

    gains_path = Path(args_cli.gains_file).expanduser().resolve()
    _ensure_gains_file_exists(gains_path)
    stiffness_by_robot, damping_by_robot, offset_efforts_by_robot = _apply_runtime_gains(
        robots=robots, gains_path=gains_path
    )
    gains_mtime = gains_path.stat().st_mtime

    print("[INFO] Scene ready for Gain Tuner.")
    if args_cli.disable_gravity:
        print("[INFO] Gravity is DISABLED for this run (gravity = [0, 0, 0]).")
    else:
        print("[INFO] Gravity is ENABLED (default Isaac Sim gravity).")
    print("[INFO] Select the robot articulation in the Gain Tuner, then run step/sine tests.")
    print("[INFO] Runtime gains file:", gains_path)
    if args_cli.passive_command_stream:
        print(
            "[INFO] Passive command stream is ON: skipping scene.write_data_to_sim() so Gain Tuner commands are not overwritten."
        )
    else:
        print(
            "[WARN] Passive command stream is OFF: scene.write_data_to_sim() may overwrite Gain Tuner commands."
        )
    if args_cli.reload_gains:
        print(
            f"[INFO] Gains hot-reload is ON (period={args_cli.gains_reload_period_s:.2f}s). "
            "Edit the gains file and save to apply without restarting."
        )
    else:
        print("[INFO] Gains hot-reload is OFF.")

    sim_dt = sim.get_physics_dt()
    next_reload_check = time.time() + max(0.05, args_cli.gains_reload_period_s)
    effort_period_s = 1.0 / max(0.1, args_cli.joint_effort_print_hz)
    next_effort_print = time.time() + effort_period_s
    selected_joint_names = {
        name.strip()
        for name in args_cli.joint_effort_joint_names.split(",")
        if name.strip()
    }
    selected_joint_names = selected_joint_names if len(selected_joint_names) > 0 else None
    if args_cli.print_joint_effort and selected_joint_names is not None:
        print(
            "[INFO] Printing selected joints only: "
            + ", ".join(sorted(selected_joint_names))
        )
    while simulation_app.is_running():
        if args_cli.reload_gains and time.time() >= next_reload_check:
            next_reload_check = time.time() + max(0.05, args_cli.gains_reload_period_s)
            try:
                new_mtime = gains_path.stat().st_mtime
                if new_mtime != gains_mtime:
                    stiffness_by_robot, damping_by_robot, offset_efforts_by_robot = _apply_runtime_gains(
                        robots=robots, gains_path=gains_path
                    )
                    gains_mtime = new_mtime
                    print("[INFO] Reloaded gains from file.")
            except Exception as exc:
                print(f"[WARN] Failed to reload gains: {exc}")

        if not args_cli.passive_command_stream:
            scene.write_data_to_sim()
        # Keep additive effort offsets active in both passive and non-passive modes.
        for robot_name, robot in robots.items():
            if offset_efforts_by_robot[robot_name].abs().sum().item() > 0.0:
                robot.root_physx_view.set_dof_actuation_forces(
                    offset_efforts_by_robot[robot_name], robot._ALL_INDICES
                )
        sim.step()
        scene.update(sim_dt)

        if args_cli.print_joint_effort and time.time() >= next_effort_print:
            next_effort_print = time.time() + effort_period_s
            env_id = args_cli.joint_effort_env_id
            try:
                for robot_name, robot in robots.items():
                    if env_id < 0 or env_id >= robot.data.joint_pos.shape[0]:
                        print(
                            f"[WARN] joint_effort_env_id={env_id} out of range for {robot_name}. "
                            f"Valid range: [0, {robot.data.joint_pos.shape[0]-1}]"
                        )
                        continue
                    print(
                        _format_tau_cmd_est_line(
                            robot_name=robot_name,
                            robot=robot,
                            stiffness=stiffness_by_robot[robot_name],
                            damping=damping_by_robot[robot_name],
                            offset_efforts=offset_efforts_by_robot[robot_name],
                            env_id=env_id,
                            arm_only=args_cli.joint_effort_arm_only,
                            selected_joint_names=selected_joint_names,
                        )
                    )
            except Exception as exc:
                print(f"[WARN] Failed to print joint efforts: {exc}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
