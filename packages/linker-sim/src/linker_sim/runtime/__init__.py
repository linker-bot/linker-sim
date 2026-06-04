"""Runtime helpers for `linker-sim`.

Right now this only hosts the replay loop. As more standalone runtime
modes are added (open-loop trajectories, scripted demos, ...), they
should land here too — separate from `sim/envs/base.py`, which is the
RL training loop.
"""
