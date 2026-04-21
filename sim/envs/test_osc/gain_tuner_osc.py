"""Launch OSC tuning scene for AR5/L6 with runtime hot-reloadable gains.

This script is OSC-oriented (end-effector pose control), unlike the legacy
PD gain tuner that edits joint-drive stiffness/damping.

Usage:
    python sim/envs/test_osc/gain_tuner_osc.py --num_envs 1 --robot_side left
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser(description="Spawn AR5/L6 OSC tuning scene with runtime gain hot-reload.")
parser.add_argument("--num_envs", type=int, default=1, help="Use 1 for interactive tuning.")
parser.add_argument("--robot_side", type=str, default="left", choices=["left", "right", "both"])
parser.add_argument("--ee_frame", type=str, default="tcp", choices=["tcp", "wrist"])
parser.add_argument(
    "--gains_file",
    type=str,
    default=str((REPO_ROOT / "sim/envs/test_osc/osc_gains.json").resolve()),
    help="JSON file containing OSC parameters and probe command settings.",
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
    "--print_status",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Print periodic OSC status including ee position error and arm effort norm.",
)
parser.add_argument(
    "--status_print_hz",
    type=float,
    default=5.0,
    help="Console print rate for status lines.",
)
parser.add_argument(
    "--status_env_id",
    type=int,
    default=0,
    help="Environment index to print status from.",
)
parser.add_argument(
    "--disable_gravity",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Set world gravity to zero for quick bench checks.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch
from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
from isaaclab.scene import InteractiveScene

from sim.assets import make_ar5_l6_robot_cfg
from sim.envs.test_osc.scene_cfg import TestOscDualSceneCfg, TestOscSceneCfg


def _default_osc_gains() -> dict:
    return {
        "motion_stiffness_task": [150.0, 150.0, 150.0, 80.0, 80.0, 80.0],
        "motion_damping_ratio_task": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "nullspace_stiffness": 10.0,
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


def _ensure_gains_file_exists(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_default_osc_gains(), indent=2) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote default OSC gains file: {path}")


def _load_osc_gains(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level object in {path}, got: {type(payload)}")

    required = ["motion_stiffness_task", "motion_damping_ratio_task", "nullspace_stiffness"]
    for key in required:
        if key not in payload:
            raise ValueError(f"Missing required key '{key}' in {path}")
    if len(payload["motion_stiffness_task"]) != 6 or len(payload["motion_damping_ratio_task"]) != 6:
        raise ValueError("motion_stiffness_task and motion_damping_ratio_task must each have 6 values.")

    payload.setdefault("arm_action_scale_pos", 0.05)
    payload.setdefault("arm_action_scale_rot", 0.25)
    payload.setdefault("hand_hold_default_pose", True)
    payload.setdefault(
        "probe_command",
        {
            "mode": "sine",
            "pos_axis": 0,
            "rot_axis": 3,
            "pos_amplitude": 0.25,
            "rot_amplitude": 0.0,
            "period_s": 4.0,
        },
    )
    return payload


@dataclass
class OscRobotCtx:
    robot_name: str
    robot: object
    arm_ids: torch.Tensor
    hand_ids: torch.Tensor
    ee_body_id: int
    jacobi_body_id: int
    osc: OperationalSpaceController
    side: str


def _resolve_side(robot) -> str:
    return "left" if any(name.startswith("AR5_5_07L_") for name in robot.joint_names) else "right"


def _resolve_ee_name(side: str, ee_frame: str) -> str:
    if side == "left":
        return "AR5_5_07L_W4C4A2_tcp" if ee_frame == "tcp" else "AR5_5_07L_W4C4A2_link7"
    return "AR5_5_07R_W4C4A2_tcp" if ee_frame == "tcp" else "AR5_5_07R_W4C4A2_link7"


def _build_osc_ctx(robot_name: str, robot, ee_frame: str, gains: dict, num_envs: int, device: str) -> OscRobotCtx:
    side = _resolve_side(robot)
    if side == "left":
        arm_ids, _ = robot.find_joints("AR5_5_07L_W4C4A2_joint_[1-7]")
        hand_ids, _ = robot.find_joints("lh_.*")
    else:
        arm_ids, _ = robot.find_joints("AR5_5_07R_W4C4A2_joint_[1-7]")
        hand_ids, _ = robot.find_joints("rh_.*")

    ee_name = _resolve_ee_name(side=side, ee_frame=ee_frame)
    body_ids, _ = robot.find_bodies(ee_name)
    if len(body_ids) != 1:
        raise ValueError(f"Expected one body for ee frame '{ee_name}', found {len(body_ids)}")
    ee_body_id = int(body_ids[0])
    jacobi_body_id = ee_body_id - 1 if robot.is_fixed_base else ee_body_id

    osc_cfg = OperationalSpaceControllerCfg(
        target_types=["pose_rel"],
        motion_control_axes_task=(1, 1, 1, 1, 1, 1),
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=False,
        gravity_compensation=True,
        impedance_mode="fixed",
        motion_stiffness_task=tuple(float(v) for v in gains["motion_stiffness_task"]),
        motion_damping_ratio_task=tuple(float(v) for v in gains["motion_damping_ratio_task"]),
        nullspace_control="position",
        nullspace_stiffness=float(gains["nullspace_stiffness"]),
        nullspace_damping_ratio=1.0,
    )
    osc = OperationalSpaceController(cfg=osc_cfg, num_envs=num_envs, device=device)

    return OscRobotCtx(
        robot_name=robot_name,
        robot=robot,
        arm_ids=torch.tensor(arm_ids, device=robot.data.joint_pos.device, dtype=torch.long),
        hand_ids=torch.tensor(hand_ids, device=robot.data.joint_pos.device, dtype=torch.long),
        ee_body_id=ee_body_id,
        jacobi_body_id=jacobi_body_id,
        osc=osc,
        side=side,
    )


def _compute_ee_pose_b(robot, ee_body_id: int) -> torch.Tensor:
    ee_pos_w = robot.data.body_pos_w[:, ee_body_id]
    ee_quat_w = robot.data.body_quat_w[:, ee_body_id]
    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
    return torch.cat([ee_pos_b, ee_quat_b], dim=-1)


def _compute_ee_velocity_b(robot, ee_body_id: int) -> torch.Tensor:
    ee_vel_w = robot.data.body_vel_w[:, ee_body_id, :]
    relative_vel_w = ee_vel_w - robot.data.root_vel_w
    ee_vel_b = torch.zeros_like(relative_vel_w)
    ee_vel_b[:, 0:3] = math_utils.quat_apply_inverse(robot.data.root_quat_w, relative_vel_w[:, 0:3])
    ee_vel_b[:, 3:6] = math_utils.quat_apply_inverse(robot.data.root_quat_w, relative_vel_w[:, 3:6])
    return ee_vel_b


def _build_probe_command(gains: dict, sim_time_s: float, num_envs: int, device: torch.device) -> torch.Tensor:
    probe = gains["probe_command"]
    cmd = torch.zeros((num_envs, 6), device=device)
    mode = str(probe.get("mode", "sine")).lower()
    if mode == "zero":
        return cmd

    pos_axis = int(probe.get("pos_axis", 0))
    rot_axis = int(probe.get("rot_axis", 3))
    period_s = max(1e-3, float(probe.get("period_s", 4.0)))
    phase = 2.0 * torch.pi * torch.tensor(sim_time_s / period_s, device=device)
    s = torch.sin(phase)

    pos_amp = float(probe.get("pos_amplitude", 0.25))
    rot_amp = float(probe.get("rot_amplitude", 0.0))
    if 0 <= pos_axis <= 2:
        cmd[:, pos_axis] = pos_amp * s
    if 3 <= rot_axis <= 5:
        cmd[:, rot_axis] = rot_amp * s
    return cmd


def _format_status(ctx: OscRobotCtx, env_id: int, arm_efforts: torch.Tensor) -> str:
    ee_pose_b = _compute_ee_pose_b(ctx.robot, ctx.ee_body_id)
    desired = ctx.osc.desired_ee_pose_b
    if desired is None:
        pos_err = 0.0
    else:
        pos_err = torch.norm(desired[env_id, :3] - ee_pose_b[env_id, :3], p=2).item()
    effort_norm = torch.norm(arm_efforts[env_id], p=2).item()
    return f"[OSC][{ctx.robot_name}] ee_pos_err={pos_err:.5f} arm_effort_l2={effort_norm:.4f}"


def main() -> None:
    if args_cli.disable_gravity:
        sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, gravity=(0.0, 0.0, 0.0))
    else:
        sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.8, -1.4, 1.2], target=[0.4, 0.0, 0.4])

    if args_cli.robot_side == "both":
        scene_cfg = TestOscDualSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
        robot_names = ["robot_left", "robot_right"]
    else:
        scene_cfg = TestOscSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
        scene_cfg.robot = make_ar5_l6_robot_cfg(
            side=args_cli.robot_side, prim_path="{ENV_REGEX_NS}/Robot", control_mode="osc"
        )
        robot_names = ["robot"]

    scene = InteractiveScene(scene_cfg)
    sim.reset()
    robots = {name: scene[name] for name in robot_names}

    gains_path = Path(args_cli.gains_file).expanduser().resolve()
    _ensure_gains_file_exists(gains_path)
    gains = _load_osc_gains(gains_path)
    gains_mtime = gains_path.stat().st_mtime

    osc_ctxs = [
        _build_osc_ctx(
            robot_name=name,
            robot=robot,
            ee_frame=args_cli.ee_frame,
            gains=gains,
            num_envs=args_cli.num_envs,
            device=str(robot.data.joint_pos.device),
        )
        for name, robot in robots.items()
    ]
    print("[INFO] OSC tuning scene ready.")
    print("[INFO] Runtime OSC gains file:", gains_path)

    sim_dt = sim.get_physics_dt()
    sim_time_s = 0.0
    next_reload_check = time.time() + max(0.05, args_cli.gains_reload_period_s)
    status_period_s = 1.0 / max(0.1, args_cli.status_print_hz)
    next_status_print = time.time() + status_period_s
    last_efforts: dict[str, torch.Tensor] = {
        ctx.robot_name: torch.zeros((args_cli.num_envs, len(ctx.arm_ids)), device=ctx.robot.data.joint_pos.device)
        for ctx in osc_ctxs
    }

    while simulation_app.is_running():
        if args_cli.reload_gains and time.time() >= next_reload_check:
            next_reload_check = time.time() + max(0.05, args_cli.gains_reload_period_s)
            try:
                new_mtime = gains_path.stat().st_mtime
                if new_mtime != gains_mtime:
                    gains = _load_osc_gains(gains_path)
                    gains_mtime = new_mtime
                    osc_ctxs = [
                        _build_osc_ctx(
                            robot_name=name,
                            robot=robot,
                            ee_frame=args_cli.ee_frame,
                            gains=gains,
                            num_envs=args_cli.num_envs,
                            device=str(robot.data.joint_pos.device),
                        )
                        for name, robot in robots.items()
                    ]
                    print("[INFO] Reloaded OSC gains from file.")
            except Exception as exc:
                print(f"[WARN] Failed to reload OSC gains: {exc}")

        cmd = _build_probe_command(gains=gains, sim_time_s=sim_time_s, num_envs=args_cli.num_envs, device=sim.device)
        cmd[:, 0:3] *= float(gains["arm_action_scale_pos"])
        cmd[:, 3:6] *= float(gains["arm_action_scale_rot"])

        for ctx in osc_ctxs:
            ee_pose_b = _compute_ee_pose_b(ctx.robot, ctx.ee_body_id)
            ee_vel_b = _compute_ee_velocity_b(ctx.robot, ctx.ee_body_id)
            ctx.osc.set_command(command=cmd, current_ee_pose_b=ee_pose_b)

            jacobian_b = ctx.robot.root_physx_view.get_jacobians()[:, ctx.jacobi_body_id, :, :][:, :, ctx.arm_ids]
            mass_matrix = ctx.robot.root_physx_view.get_generalized_mass_matrices()[:, ctx.arm_ids, :][:, :, ctx.arm_ids]
            gravity = ctx.robot.root_physx_view.get_gravity_compensation_forces()[:, ctx.arm_ids]
            joint_pos = ctx.robot.data.joint_pos[:, ctx.arm_ids]
            joint_vel = ctx.robot.data.joint_vel[:, ctx.arm_ids]
            null_target = ctx.robot.data.default_joint_pos[:, ctx.arm_ids]

            arm_efforts = ctx.osc.compute(
                jacobian_b=jacobian_b,
                current_ee_pose_b=ee_pose_b,
                current_ee_vel_b=ee_vel_b,
                mass_matrix=mass_matrix,
                gravity=gravity,
                current_joint_pos=joint_pos,
                current_joint_vel=joint_vel,
                nullspace_joint_pos_target=null_target,
            )
            last_efforts[ctx.robot_name] = arm_efforts
            ctx.robot.set_joint_effort_target(arm_efforts, joint_ids=ctx.arm_ids)

            if bool(gains.get("hand_hold_default_pose", True)) and len(ctx.hand_ids) > 0:
                hand_default = ctx.robot.data.default_joint_pos[:, ctx.hand_ids]
                ctx.robot.set_joint_position_target(hand_default, joint_ids=ctx.hand_ids)

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        sim_time_s += sim_dt

        if args_cli.print_status and time.time() >= next_status_print:
            next_status_print = time.time() + status_period_s
            env_id = args_cli.status_env_id
            for ctx in osc_ctxs:
                if env_id < 0 or env_id >= ctx.robot.data.joint_pos.shape[0]:
                    print(
                        f"[WARN] status_env_id={env_id} out of range for {ctx.robot_name}. "
                        f"Valid range: [0, {ctx.robot.data.joint_pos.shape[0]-1}]"
                    )
                    continue
                print(_format_status(ctx=ctx, env_id=env_id, arm_efforts=last_efforts[ctx.robot_name]))


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
