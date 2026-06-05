"""Episode recorder + sink adapters.

`Recorder` is called once per env `step` with batched `(obs, action,
reward, terminated, truncated)` tensors. It tracks per-env episode
ids and frame indices and forwards each env's frame to a user-chosen
`sink` callable. The sink owns storage — the recorder just emits
typed payloads.

Sink contract:

    sink(episode_id: int, frame_idx: int, frame: dict) -> None

`frame` keys:

    - "obs": numpy array, shape (observation_dim,)
    - "action": numpy array, shape (action_dim,)
    - "reward": float
    - "terminated": bool
    - "truncated": bool
    - "info": dict (empty by default; caller can pass extras)

Design note: `BaseEnv` does not know about the recorder. The driver
(`scripts/run.py` or a notebook) constructs the recorder, steps the
env, and forwards. This keeps `BaseEnv` clean and makes recording
optional.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import torch


Sink = Callable[[int, int, dict[str, Any]], None]
"""`(episode_id, frame_idx, frame_dict) -> None`."""


# ---------------------------------------------------------------------------- #
# Sink implementations
# ---------------------------------------------------------------------------- #


def null_sink(episode_id: int, frame_idx: int, frame: dict[str, Any]) -> None:
    """Discards everything. Useful when recording is toggled off."""
    return


class NullSink:
    """Class form of `null_sink` for hydra `_target_` instantiation.

    `instantiate({_target_: linker_sim.io.recorder.null_sink})` would *call*
    the function (hydra treats functions as factories). `NullSink`
    gives us something that instantiates to a callable object instead.
    """

    def __call__(self, episode_id: int, frame_idx: int, frame: dict[str, Any]) -> None:
        return

    def close(self) -> None:
        return


class JsonlSink:
    """Write one JSONL file per episode.

    File naming: `<out_dir>/episode_<episode_id:06d>.jsonl`. Each line
    is one frame dict serialized with `json.dumps`. Numpy arrays are
    converted to lists (lossy for float precision — use `LeRobotSink`
    for exact data).
    """

    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._open_files: dict[int, Any] = {}

    def __call__(self, episode_id: int, frame_idx: int, frame: dict[str, Any]) -> None:
        fh = self._open_files.get(episode_id)
        if fh is None:
            path = self.out_dir / f"episode_{episode_id:06d}.jsonl"
            fh = open(path, "w")
            self._open_files[episode_id] = fh
        payload = {"frame_idx": frame_idx, **_to_jsonable(frame)}
        fh.write(json.dumps(payload) + "\n")
        fh.flush()

    def close(self) -> None:
        for fh in self._open_files.values():
            fh.close()
        self._open_files.clear()


def _to_jsonable(frame: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in frame.items():
        out[k] = _coerce(v)
    return out


def _coerce(v: Any) -> Any:
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, (bool, int, float, str, type(None))):
        return v
    if isinstance(v, dict):
        return {k: _coerce(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_coerce(x) for x in v]
    return repr(v)  # fallback so serialization never crashes


class LeRobotSink:
    """Parquet writer compatible with the external LeRobot collector.

    Requires `pyarrow` (opt-in extra). Buffers frames in-memory per
    episode and flushes on `end_episode` — suitable for episodes that
    fit in RAM.

    Schema is intentionally minimal: `observation.state` (the flat obs
    vector), `action`, `reward`, `terminated`, `truncated`,
    `episode_index`, `frame_index`, `timestamp`. The LeRobot schema is
    richer (observation.images, task_index, …); map them in on the
    caller side if you need them.
    """

    def __init__(self, out_dir: str | Path, fps: float = 30.0):
        try:
            import pyarrow  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "LeRobotSink requires pyarrow. Install with "
                "`uv pip install 'linker-sim[lerobot]'`."
            ) from e
        self.out_dir = Path(out_dir).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self._buffers: dict[int, list[dict[str, Any]]] = {}

    def __call__(self, episode_id: int, frame_idx: int, frame: dict[str, Any]) -> None:
        self._buffers.setdefault(episode_id, []).append(
            {"frame_index": frame_idx, "timestamp": frame_idx / self.fps, **frame}
        )

    def end_episode(self, episode_id: int) -> Path | None:
        frames = self._buffers.pop(episode_id, None)
        if not frames:
            return None
        import pyarrow as pa
        import pyarrow.parquet as pq

        rows = [
            {
                "episode_index": episode_id,
                "frame_index": f["frame_index"],
                "timestamp": f["timestamp"],
                "observation.state": list(f.get("obs", [])),
                "action": list(f.get("action", [])),
                "reward": float(f.get("reward", 0.0)),
                "terminated": bool(f.get("terminated", False)),
                "truncated": bool(f.get("truncated", False)),
            }
            for f in frames
        ]
        table = pa.Table.from_pylist(rows)
        path = self.out_dir / f"episode_{episode_id:06d}.parquet"
        pq.write_table(table, path)
        return path

    def close(self) -> None:
        for ep in list(self._buffers.keys()):
            self.end_episode(ep)


# ---------------------------------------------------------------------------- #
# Recorder
# ---------------------------------------------------------------------------- #


@dataclass
class RecorderCfg:
    enabled: bool = True
    record_every_env: bool = False
    """If False, only env index 0 is recorded (useful when num_envs>>1)."""


class Recorder:
    """Batched env recorder. See module docstring."""

    def __init__(self, sink: Sink, num_envs: int, cfg: RecorderCfg | None = None):
        self.sink = sink
        self.num_envs = num_envs
        self.cfg = cfg or RecorderCfg()

        # Per-env bookkeeping. All on CPU — recorder is not in the hot path.
        self._episode_ids = torch.zeros(num_envs, dtype=torch.long)
        self._frame_idx = torch.zeros(num_envs, dtype=torch.long)
        self._global_episode_counter = 0

    def record_step(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        info: dict | None = None,
    ) -> None:
        if not self.cfg.enabled:
            return

        obs_np = obs.detach().cpu().numpy()
        action_np = action.detach().cpu().numpy()
        reward_np = reward.detach().cpu().numpy()
        term_np = terminated.detach().cpu().numpy()
        trunc_np = truncated.detach().cpu().numpy()

        env_iter = range(self.num_envs) if self.cfg.record_every_env else (0,)
        for env_idx in env_iter:
            ep_id = int(self._episode_ids[env_idx])
            frame_idx = int(self._frame_idx[env_idx])
            frame = {
                "obs": obs_np[env_idx],
                "action": action_np[env_idx],
                "reward": float(reward_np[env_idx]),
                "terminated": bool(term_np[env_idx]),
                "truncated": bool(trunc_np[env_idx]),
                "info": info or {},
            }
            self.sink(ep_id, frame_idx, frame)

        # Advance frame indices; bump episode counter on done.
        self._frame_idx += 1
        done = (terminated | truncated).detach().cpu()
        if bool(done.any()):
            done_idx = torch.nonzero(done, as_tuple=False).squeeze(-1)
            for idx in done_idx.tolist():
                self._global_episode_counter += 1
                self._episode_ids[idx] = self._global_episode_counter
                self._frame_idx[idx] = 0

    def close(self) -> None:
        if hasattr(self.sink, "close") and callable(self.sink.close):
            self.sink.close()


# ---------------------------------------------------------------------------- #
# Optional Protocol (for type-checker friendliness; not required at runtime)
# ---------------------------------------------------------------------------- #


class SinkProtocol(Protocol):
    def __call__(self, episode_id: int, frame_idx: int, frame: dict[str, Any]) -> None: ...
    def close(self) -> None: ...
