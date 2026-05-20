"""Session-only PD gain hot-reload helper.

Writes per-role `(stiffness, damping)` to a JSON file (seeded from the
workstation manifest's `default_gains`) and re-applies the values via
`robot.write_gains` whenever the file's mtime changes. Edit the file
while the sim is running to tune gains without restarting.

Discard the file when done — it isn't tracked by the manifest.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from sim.backends.base import Robot


class GainWatcher:
    def __init__(self, robot: Robot, path: Path, poll_period_s: float = 0.5):
        self._robot = robot
        self._path = Path(path)
        self._poll_period_s = float(poll_period_s)
        self._next_check = 0.0
        self._roles = list(robot.handle.joints.keys())
        if not self._path.exists():
            self._seed()
        self._mtime = self._path.stat().st_mtime
        print(f"[gain_tuner] watching {self._path}")

    def _seed(self) -> None:
        data: dict[str, dict[str, float]] = {}
        for role in self._roles:
            g = self._robot.handle.default_gains.get(role)
            if g is None:
                continue
            data[role] = {"stiffness": float(g.stiffness), "damping": float(g.damping)}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"[gain_tuner] seeded gains -> {self._path}")

    def tick(self) -> None:
        now = time.time()
        if now < self._next_check:
            return
        self._next_check = now + self._poll_period_s
        try:
            m = self._path.stat().st_mtime
        except FileNotFoundError:
            return
        if m == self._mtime:
            return
        self._mtime = m
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[gain_tuner] parse failed: {exc}")
            return
        for role, g in data.items():
            if role not in self._roles:
                continue
            try:
                self._robot.write_gains(role, float(g["stiffness"]), float(g["damping"]))
            except Exception as exc:
                print(f"[gain_tuner] apply failed for {role}: {exc}")
        print(f"[gain_tuner] applied: {data}")
