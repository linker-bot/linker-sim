"""OSC test env.

Scripts:

- `spawn_osc_scene.py` — manual smoke test; spawns a workstation +
  periodically resets envs.
- `gain_tuner_osc.py` — interactive OSC gain tuning with hot-reload.

For programmatic RL rollouts, use `sim.envs.base.BaseEnv` directly
(or `scripts/run.py` with a hydra config).
"""
