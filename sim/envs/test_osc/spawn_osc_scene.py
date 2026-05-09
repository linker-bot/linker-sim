"""OSC smoke-test entrypoint.

PR #2a rewires this to drive the composed workstation through the new
`IsaacSimBackend` directly — no `DirectRLEnv` subclass, no legacy
scene cfg. The original behavior (spawn workstation, let the implicit
PD hold it, periodically reset some envs) is preserved.

For a full OSC rollout against the new backbone, see
`sim/envs/test_osc/osc_rl_env.py` (shim over `BaseEnv` + `OscController`
+ `LegacyOscTask`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_ROBOT_SIDE_TO_WORKSTATION = {
    "left": "ar5_l6_bench",
    "right": "ar5_l6_bench_right",
    "both": "ar5_l6_bench_bimanual",
}


parser = argparse.ArgumentParser(description="Spawn OSC test scene over a composed workstation.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--workstation",
    type=str,
    default=None,
    help="Composed workstation name (e.g. 'ar5_l6_bench'). Overrides --robot_side.",
)
parser.add_argument(
    "--robot_side",
    type=str,
    default="left",
    choices=["left", "right", "both"],
    help="Convenience flag: 'left' -> ar5_l6_bench, 'right' -> ar5_l6_bench_right, "
    "'both' -> ar5_l6_bench_bimanual.",
)
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

import torch  # noqa: E402

from sim.backends.isaac.backend import IsaacBackendCfg, IsaacSimBackend  # noqa: E402


def _resolve_workstation(args) -> str:
    if args.workstation:
        return args.workstation
    return _ROBOT_SIDE_TO_WORKSTATION[args.robot_side]


def run(backend: IsaacSimBackend, reset_interval: int, reset_envs_per_event: int) -> None:
    num_envs = backend.num_envs
    device = backend.device
    all_env_ids = torch.arange(num_envs, device=device, dtype=torch.long)
    reset_cursor = 0
    step_count = 0

    while simulation_app.is_running():
        if step_count % reset_interval == 0:
            step_count = 0
            if reset_envs_per_event <= 0 or reset_envs_per_event >= num_envs:
                env_ids = all_env_ids
            else:
                env_ids = (torch.arange(reset_envs_per_event, device=device) + reset_cursor) % num_envs
                reset_cursor = int((reset_cursor + reset_envs_per_event) % num_envs)
            backend.reset(env_ids=env_ids)
            print(f"[INFO] Reset robot state for env ids: {env_ids.tolist()}")

        backend.write_data()
        backend.step()
        step_count += 1


def main() -> None:
    workstation_name = _resolve_workstation(args_cli)
    backend_cfg = IsaacBackendCfg(
        workstations={"robot": workstation_name},
        num_envs=args_cli.num_envs,
        env_spacing=2.5,
        device=args_cli.device,
    )
    backend = IsaacSimBackend(backend_cfg)
    print(f"[INFO] OSC test scene setup complete for workstation {workstation_name!r}.")
    run(
        backend,
        reset_interval=max(1, args_cli.reset_interval),
        reset_envs_per_event=max(0, args_cli.reset_envs_per_event),
    )


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
