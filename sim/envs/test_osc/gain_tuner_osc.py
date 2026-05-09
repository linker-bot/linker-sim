"""Launch OSC tuning scene for AR5/L6 with runtime hot-reloadable gains.

Runs the workstation through the PR #2 runtime backbone
(`IsaacSimBackend` + `OscController` + `JointPDController`) and drives
a sine probe command through the OSC command input. Gains are
hot-reloaded from a JSON file — edit the file, save, and the
controller rebuilds without restarting Isaac.

Usage:
    python sim/envs/test_osc/gain_tuner_osc.py --num_envs 1 --robot_side left
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

_ROBOT_SIDE_TO_WORKSTATION = {
    "left": "ar5_l6_bench",
    "right": "ar5_l6_bench_right",
}


parser = argparse.ArgumentParser(description="OSC gain tuning with runtime hot-reload.")
parser.add_argument("--num_envs", type=int, default=1, help="Use 1 for interactive tuning.")
parser.add_argument("--robot_side", type=str, default="left", choices=["left", "right"])
parser.add_argument("--ee_frame", type=str, default="tcp", choices=["tcp", "wrist"])
parser.add_argument(
    "--gains_file",
    type=str,
    default=str((REPO_ROOT / "sim/envs/test_osc/osc_gains.json").resolve()),
)
parser.add_argument("--reload_gains", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--gains_reload_period_s", type=float, default=0.5)
parser.add_argument("--print_status", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--status_print_hz", type=float, default=5.0)
parser.add_argument("--status_env_id", type=int, default=0)
parser.add_argument("--disable_gravity", action=argparse.BooleanOptionalAction, default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
import torch  # noqa: E402

from sim.backends.isaac.backend import IsaacBackendCfg, IsaacSimBackend  # noqa: E402
from sim.controllers.joint_pd import JointPDController, JointPDControllerCfg  # noqa: E402
from sim.controllers.osc import OscController, OscControllerCfg  # noqa: E402


def _default_osc_gains() -> dict:
    return {
        "motion_stiffness_task": [150.0, 150.0, 150.0, 80.0, 80.0, 80.0],
        "motion_damping_ratio_task": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "nullspace_stiffness": 10.0,
        "actuator_stiffness": 150.0,
        "actuator_damping": 8.0,
        "arm_action_scale_pos": 0.05,
        "arm_action_scale_rot": 0.25,
        "hand_hold_default_pose": True,
        "probe_command": {
            "mode": "sine",
            "pos_axis": 0,
            "rot_axis": 3,
            "pos_amplitude": 0.25,
            "rot_amplitude": 0.0,
            "period_s": 4.0,
        },
    }


def _ensure_gains_file(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_default_osc_gains(), indent=2) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote default OSC gains file: {path}")


def _load_gains(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = ["motion_stiffness_task", "motion_damping_ratio_task", "nullspace_stiffness"]
    for key in required:
        if key not in payload:
            raise ValueError(f"missing required key {key!r} in {path}")
    payload.setdefault("actuator_stiffness", 150.0)
    payload.setdefault("actuator_damping", 8.0)
    payload.setdefault("arm_action_scale_pos", 0.05)
    payload.setdefault("arm_action_scale_rot", 0.25)
    payload.setdefault("hand_hold_default_pose", True)
    payload.setdefault(
        "probe_command",
        {"mode": "sine", "pos_axis": 0, "rot_axis": 3, "pos_amplitude": 0.25, "rot_amplitude": 0.0, "period_s": 4.0},
    )
    return payload


def _build_osc_from_gains(ee_frame: str, gains: dict) -> OscController:
    return OscController(
        OscControllerCfg(
            role="arm",
            frame=ee_frame,
            action_scale_pos=float(gains["arm_action_scale_pos"]),
            action_scale_rot=float(gains["arm_action_scale_rot"]),
            stiffness=tuple(float(v) for v in gains["motion_stiffness_task"]),
            damping_ratio=tuple(float(v) for v in gains["motion_damping_ratio_task"]),
            nullspace_stiffness=float(gains["nullspace_stiffness"]),
            actuator_stiffness=float(gains["actuator_stiffness"]),
            actuator_damping=float(gains["actuator_damping"]),
            gain_profile=None,  # override with JSON values literally
        )
    )


def _probe_command(gains: dict, t: float, num_envs: int, device: torch.device) -> torch.Tensor:
    probe = gains["probe_command"]
    cmd = torch.zeros((num_envs, 6), device=device)
    if str(probe.get("mode", "sine")).lower() == "zero":
        return cmd
    period = max(1e-3, float(probe.get("period_s", 4.0)))
    s = torch.sin(torch.tensor(2.0 * torch.pi * t / period, device=device))
    pos_axis = int(probe.get("pos_axis", 0))
    rot_axis = int(probe.get("rot_axis", 3))
    if 0 <= pos_axis <= 2:
        cmd[:, pos_axis] = float(probe.get("pos_amplitude", 0.25)) * s
    if 3 <= rot_axis <= 5:
        cmd[:, rot_axis] = float(probe.get("rot_amplitude", 0.0)) * s
    return cmd


def main() -> None:
    workstation_name = _ROBOT_SIDE_TO_WORKSTATION[args_cli.robot_side]

    gains_path = Path(args_cli.gains_file).expanduser().resolve()
    _ensure_gains_file(gains_path)
    gains = _load_gains(gains_path)
    gains_mtime = gains_path.stat().st_mtime

    backend = IsaacSimBackend(
        IsaacBackendCfg(
            workstations={"robot": workstation_name},
            num_envs=args_cli.num_envs,
            device=args_cli.device,
        )
    )
    # Apply --disable_gravity by poking the sim context after construction.
    if args_cli.disable_gravity:
        backend.sim.get_physics_context().set_gravity(0.0)

    robot = backend.robots["robot"]
    ee_frame = "arm:tool0" if args_cli.ee_frame == "tcp" else None

    osc = _build_osc_from_gains(ee_frame, gains)
    osc.attach(robot)

    hand = JointPDController(JointPDControllerCfg(role="hand", action_scale=0.0))
    hand.attach(robot)

    # Reset robots once so the default pose is realized.
    backend.reset()

    dt = backend.dt
    t = 0.0
    next_reload_check = time.time() + max(0.05, args_cli.gains_reload_period_s)
    status_period_s = 1.0 / max(0.1, args_cli.status_print_hz)
    next_status_print = time.time() + status_period_s
    print(f"[INFO] OSC tuner ready. Editing {gains_path} hot-reloads gains.")

    while simulation_app.is_running():
        # Hot-reload gains file.
        if args_cli.reload_gains and time.time() >= next_reload_check:
            next_reload_check = time.time() + max(0.05, args_cli.gains_reload_period_s)
            try:
                new_mtime = gains_path.stat().st_mtime
                if new_mtime != gains_mtime:
                    gains = _load_gains(gains_path)
                    gains_mtime = new_mtime
                    osc = _build_osc_from_gains(ee_frame, gains)
                    osc.attach(robot)
                    print("[INFO] Reloaded OSC gains from file.")
            except Exception as exc:
                print(f"[WARN] Failed to reload gains: {exc}")

        cmd = _probe_command(gains, t, args_cli.num_envs, backend.device)
        osc.set_command(cmd, robot)
        osc.apply(robot)
        if bool(gains.get("hand_hold_default_pose", True)):
            hand.set_command(torch.zeros((args_cli.num_envs, hand.command_dim), device=backend.device), robot)
            hand.apply(robot)

        backend.write_data()
        backend.step()
        t += dt

        if args_cli.print_status and time.time() >= next_status_print:
            next_status_print = time.time() + status_period_s
            env_id = args_cli.status_env_id
            if 0 <= env_id < backend.num_envs:
                ee_pose = robot.ee_pose_b(ee_frame)
                desired = osc._controller.desired_ee_pose_b if osc._controller is not None else None
                pos_err = 0.0 if desired is None else torch.norm(desired[env_id, :3] - ee_pose[env_id, :3]).item()
                print(f"[OSC] t={t:.2f}s env={env_id} ee_pos_err={pos_err:.5f}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
