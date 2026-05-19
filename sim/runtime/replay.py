"""External-data replay loop.

Drives a backend + robot from a `ReplaySource`, bypassing the BaseEnv /
controller / task pipeline entirely. The MJCF/URDF position actuators
already implement joint PD with workstation-manifest gains, so we just
write target qpos and step physics.

Designed to work with both Mujoco and Isaac backends — the only
backend-touching call is `backend.step()`. Viewer integration is the
caller's job; pass `viewer` if you want `viewer.sync()` after each
replay frame and `viewer.is_running()` honored as a stop condition.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from sim.io.replay.sources import ReplaySource


def run_replay(
    backend: Any,
    robot: Any,
    source: ReplaySource,
    *,
    viewer: Any | None = None,
    realtime: bool = False,
    max_frames: int | None = None,
    stop_flag: list | None = None,
) -> int:
    """Replay `source` through `backend`/`robot`. Returns frames consumed.

    Args:
        backend: provides `.step()` (one physics dt) and `.dt`.
        robot: provides `actuated_joint_ids_of`, `set_joint_position_target`,
            `write_joint_state`, `joint_pos_default`, `joint_vel_default`,
            `device`, `num_envs`.
        source: a ReplaySource. `bind_robot(robot)` is called here.
        viewer: optional MuJoCo passive viewer; not touched if None.
        realtime: if True, sleep so each replay frame consumes
            `1/source.hz` wall-clock seconds.
        max_frames: clamp at this many frames (None = full source).
        stop_flag: optional `[bool]` checked each frame for early exit.
    """
    source.bind_robot(robot)

    sub_steps = max(1, int(round(1.0 / (float(source.hz) * float(backend.dt)))))
    period = 1.0 / float(source.hz)
    n_frames = source.num_frames if max_frames is None else min(source.num_frames, int(max_frames))

    print(
        f"[replay] {source.describe()} -> "
        f"{sub_steps} physics steps per frame "
        f"(backend.dt={backend.dt*1000:.2f} ms), realtime={realtime}"
    )

    _teleport(robot, source.joint_targets(0))

    deadline = time.perf_counter() + period
    for t in range(n_frames):
        if viewer is not None and not viewer.is_running():
            break
        if stop_flag is not None and stop_flag[0]:
            break

        for role, target in source.joint_targets(t).items():
            ids = robot.actuated_joint_ids_of(role)
            tgt = torch.from_numpy(target).to(robot.device).unsqueeze(0)
            robot.set_joint_position_target(tgt, ids)

        for _ in range(sub_steps):
            backend.step()
        if viewer is not None:
            viewer.sync()

        if realtime:
            now = time.perf_counter()
            sleep_for = deadline - now
            if sleep_for > 0:
                time.sleep(sleep_for)
            deadline += period

    print(f"[replay] done after {t + 1} frames.")
    return t + 1


def _teleport(robot: Any, first_frame: dict[str, np.ndarray]) -> None:
    """Snap the robot to the first replay frame so the PD doesn't whip."""
    jp = robot.joint_pos_default.clone()
    jv = robot.joint_vel_default.clone()
    for role, target in first_frame.items():
        ids = robot.actuated_joint_ids_of(role)
        tgt = torch.from_numpy(target).to(jp.device, dtype=jp.dtype)
        jp[:, ids] = tgt
    robot.write_joint_state(jp, jv)
