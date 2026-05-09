"""Recorder round-trip test.

Drives BaseEnv over fakes, records episodes to a temp dir with
JsonlSink, reloads via ReplayEpisode, and verifies the recorded data
round-trips.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from sim.envs.base import BaseEnv, BaseEnvCfg
from sim.io.recorder import JsonlSink, Recorder, RecorderCfg, null_sink
from sim.io.replayer import ReplayEpisode

# Reuse fakes from test_base_env.
import tests.test_base_env as _te
FakeBackend = _te.FakeBackend
_CountingController = _te._CountingController
_ZeroTask = _te._ZeroTask


def test_jsonl_sink_writes_parseable_lines(tmp_path: Path):
    sink = JsonlSink(tmp_path)
    for frame_idx in range(5):
        sink(
            episode_id=0,
            frame_idx=frame_idx,
            frame={"obs": [0.1, 0.2], "action": [0.0, 0.0], "reward": 1.5, "terminated": False, "truncated": False, "info": {}},
        )
    sink.close()

    path = tmp_path / "episode_000000.jsonl"
    assert path.is_file()
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 5
    for i, line in enumerate(lines):
        payload = json.loads(line)
        assert payload["frame_idx"] == i
        assert payload["reward"] == 1.5


def test_recorder_forwards_one_env_by_default():
    recorded = []
    def sink(ep, idx, frame): recorded.append((ep, idx, frame))

    rec = Recorder(sink, num_envs=3, cfg=RecorderCfg(enabled=True, record_every_env=False))
    obs = torch.zeros(3, 4)
    action = torch.zeros(3, 2)
    reward = torch.tensor([1.0, 2.0, 3.0])
    terminated = torch.tensor([False, False, False])
    truncated = torch.tensor([False, False, False])
    rec.record_step(obs, action, reward, terminated, truncated)

    # record_every_env=False -> only env 0 logged
    assert len(recorded) == 1
    _, _, frame = recorded[0]
    assert frame["reward"] == 1.0


def test_recorder_rolls_episode_on_done():
    recorded = []
    def sink(ep, idx, frame): recorded.append((ep, idx, frame))

    rec = Recorder(sink, num_envs=1, cfg=RecorderCfg(enabled=True, record_every_env=True))
    obs = torch.zeros(1, 2)
    action = torch.zeros(1, 1)
    reward = torch.tensor([0.5])

    # First step: episode 0
    rec.record_step(obs, action, reward, torch.tensor([False]), torch.tensor([False]))
    # Second step: truncated -> next step is episode 1
    rec.record_step(obs, action, reward, torch.tensor([False]), torch.tensor([True]))
    # Third step: episode 1, frame 0
    rec.record_step(obs, action, reward, torch.tensor([False]), torch.tensor([False]))

    eps = [ep for ep, _, _ in recorded]
    frames = [idx for _, idx, _ in recorded]
    assert eps == [0, 0, 1]
    assert frames == [0, 1, 0]


def test_replay_episode_loads_jsonl(tmp_path: Path):
    sink = JsonlSink(tmp_path)
    for i in range(3):
        sink(episode_id=7, frame_idx=i, frame={"action": [float(i), 0.0], "reward": 0.0})
    sink.close()

    path = tmp_path / "episode_000007.jsonl"
    ep = ReplayEpisode.load_jsonl(path)
    assert ep.episode_id == 7
    assert len(ep.frames) == 3
    assert ep.frames[2]["action"] == [2.0, 0.0]


def test_null_sink_is_a_noop():
    # Just checks that null_sink accepts the sink contract and returns None.
    assert null_sink(0, 0, {"anything": [1, 2]}) is None
