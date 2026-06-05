"""Episode replayer.

Reads episodes produced by `sim.io.recorder.JsonlSink` and drives a
`BaseEnv` through them in one of two modes:

- `action_replay`: feed the recorded action each step. The env's
  physics + controller deterministically reproduce (up to RNG +
  float noise) the original rollout. Useful for reward/tuning diffs.
- `state_inject`: write the recorded joint state each step (via
  `robot.write_joint_state`) without applying actions. Useful as a
  determinism oracle — lets you visualize an exact trajectory that
  came from a different sim or from real-robot data.

Both modes bypass `Task.reset()` for the replayed env — the recorder
already captured the trajectory state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import torch

from linker_sim.envs.base import BaseEnv


ReplayMode = Literal["action_replay", "state_inject"]


@dataclass
class ReplayEpisode:
    """One recorded episode ready to be replayed."""

    episode_id: int
    frames: list[dict]

    @staticmethod
    def load_jsonl(path: str | Path) -> "ReplayEpisode":
        path = Path(path)
        frames: list[dict] = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                frames.append(json.loads(line))
        if not frames:
            raise ValueError(f"empty episode file: {path}")
        # Parse episode id out of the filename convention `episode_NNNNNN.jsonl`.
        try:
            eid = int(path.stem.split("_")[-1])
        except ValueError:
            eid = 0
        return ReplayEpisode(episode_id=eid, frames=frames)


class Replayer:
    """Drive a `BaseEnv` from a recorded episode."""

    def __init__(self, env: BaseEnv, mode: ReplayMode = "action_replay"):
        self.env = env
        self.mode = mode

    def replay(
        self,
        episode: ReplayEpisode,
        env_index: int = 0,
    ) -> Iterable[torch.Tensor]:
        """Yield observation tensors per replayed step.

        `env_index` selects which env in the batched env receives the
        replay. Other envs step with a zero action so the loop keeps
        running coherently.
        """
        num_envs = self.env.num_envs
        device = self.env.device

        obs, _ = self.env.reset()
        yield obs

        for frame in episode.frames:
            action = torch.zeros((num_envs, self.env.action_dim), device=device)
            recorded_action = torch.tensor(frame["action"], device=device, dtype=action.dtype)
            action[env_index] = recorded_action

            if self.mode == "state_inject":
                robot = self.env.robot
                # State-inject writes joint state before physics step so
                # the sim advances from the recorded pose. obs in the
                # recording was the POST-step observation; we write the
                # state that produced it.
                if "joint_pos" in frame and "joint_vel" in frame:
                    jp = torch.tensor(frame["joint_pos"], device=device)
                    jv = torch.tensor(frame["joint_vel"], device=device)
                    robot.write_joint_state(
                        jp.unsqueeze(0).expand(num_envs, -1).clone(),
                        jv.unsqueeze(0).expand(num_envs, -1).clone(),
                    )

            obs, reward, terminated, truncated, info = self.env.step(action)
            yield obs
