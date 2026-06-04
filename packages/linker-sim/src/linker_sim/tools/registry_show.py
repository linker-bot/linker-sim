#!/usr/bin/env python3
"""Dump a workstation registry handle.

Read-only CLI for inspecting what `sim.registry.load(name)` returns for a
given workstation. Useful for verifying the composer's output before a
backend consumes it.

    python -m linker_sim.tools.registry_show                    # list workstations
    python -m linker_sim.tools.registry_show ar5_l6_bench_bimanual   # dump handle fields
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from linker_sim.registry import RegistryError, discover, load  # noqa: E402


def _dump(handle) -> None:
    print(f"workstation: {handle.name}")
    print(f"  dir:     {handle.dir}")
    print(f"  urdf:    {handle.urdf_path}")
    print(f"  mjcf:    {handle.mjcf_path if handle.mjcf_path else '(not available)'}")
    print(f"  ee_link:   {handle.ee_link}")
    print(f"  base_link: {handle.base_link}")
    print("  components:")
    for role, ref in handle.components.items():
        variant = f"@{ref.variant}" if ref.variant else ""
        print(f"    {role:<12} {ref.name}{variant}  (sha {ref.sha256[:12]}…)")
    print("  joints:")
    for role, js in handle.joints.items():
        if not js:
            print(f"    {role}: (none)")
            continue
        print(f"    {role}: {len(js)} actuated")
        for j in js:
            print(f"      - {j}")
        mimic = handle.mimic_joints.get(role, [])
        if mimic:
            print(f"    {role}: {len(mimic)} mimic")
            for j in mimic:
                print(f"      ~ {j}")
    print("  frames:")
    for fname, link in handle.frames.items():
        print(f"    {fname:<28} -> {link}")
    print("  default_gains:")
    for role, g in handle.default_gains.items():
        print(f"    {role:<6} kp={g.stiffness:<8} kd={g.damping}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("name", nargs="?", help="workstation name (omit to list)")
    args = p.parse_args(argv)

    if args.name is None:
        names = discover()
        if not names:
            print("no workstations found under assets/workstations/", file=sys.stderr)
            return 1
        for n in names:
            print(n)
        return 0

    try:
        handle = load(args.name)
    except RegistryError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    _dump(handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
