# linker-robot-assets

The shared robot-asset bundle for `linker-sim` and downstream consumers.

This package ships:

- the asset tree at [`src/linker_robot_assets/assets/`](src/linker_robot_assets/assets/) — URDF / MJCF / XRDF + meshes, organised by `components/` and `workstations/` (binaries are stored via git LFS);
- the **composer** that builds workstation URDFs from components ([`composer/`](src/linker_robot_assets/composer/));
- two validators ([`validate_workstation.py`](src/linker_robot_assets/validate_workstation.py), [`validate_component_mjcf.py`](src/linker_robot_assets/validate_component_mjcf.py));
- a thin **loader API** at the package root: `asset_root()`, `workstations()`, `load_manifest(name)`.

## Install profiles

| Profile | What you get | Pulls |
|---|---|---|
| `pip install linker-robot-assets` | Loader API + bundled assets. | `pyyaml` |
| `pip install linker-robot-assets[authoring]` | Plus composer + validators. | `mujoco`, `numpy` |

The runtime profile is intentionally light so consumers (e.g. `linker-sim`'s registry) only pay for `pyyaml`. Authoring tools are gated behind `[authoring]`.

## Usage

```python
from linker_robot_assets import asset_root, workstations, load_manifest

asset_root()                  # PosixPath('.../linker_robot_assets/assets')
workstations()                # ['a7_lite_l6_dc', 'ar5_l6_bench_bimanual', ...]
load_manifest('a7_lite_l6_dc')   # {'name': '...', 'joints': {...}, ...}
```

CLI (with `[authoring]`):

```bash
python -m linker_robot_assets.composer.compose <workstation_dir> [--check-drift]
python -m linker_robot_assets.validate_workstation <workstation_dir>
python -m linker_robot_assets.validate_component_mjcf <component_dir> [--variant NAME]
```
