# `data_collection` synthetic fixture

Tiny `telemetry.npz` matching the layout expected by
[`sim/configs/source/data_collection.yaml`](../../../sim/configs/source/data_collection.yaml):

- 30 frames, 26 columns, `qpos` field
- cols 0-13: arm radians (all zero — robots hold default pose)
- cols 14-25: hand byte commands (random, deterministic seed)

Exists so `scripts/replay.py source=data_collection` runs end-to-end
against a fresh clone without external data.

For real recordings, override the source path on the CLI:

```bash
python scripts/replay.py source=data_collection source.path=/path/to/episode
```

To regenerate (e.g. if the layout schema changes), run from the repo root:

```python
import numpy as np
from pathlib import Path

out = Path("tests/fixtures/data_collection")
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
qpos = np.zeros((30, 26), dtype=np.float32)
qpos[:, 14:26] = rng.integers(0, 256, size=(30, 12)).astype(np.float32)
np.savez(out / "telemetry.npz", qpos=qpos)
```
