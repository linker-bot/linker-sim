# Third-Party Notices

This project depends on third-party software with separate licenses.
End users and redistributors are responsible for ensuring full
compliance with all applicable third-party licenses when shipping
source, binaries, containers, or integrated products.

## MuJoCo

- Project: MuJoCo
- Upstream: [https://github.com/google-deepmind/mujoco](https://github.com/google-deepmind/mujoco)
- License: Apache License 2.0.
- Notes: Optional runtime dependency, installed via the `[mujoco]` extra.

## PyTorch

- Project: PyTorch
- Upstream: [https://github.com/pytorch/pytorch](https://github.com/pytorch/pytorch)
- License: BSD-3-Clause-style with additional terms.

## Hydra

- Project: Hydra
- Upstream: [https://github.com/facebookresearch/hydra](https://github.com/facebookresearch/hydra)
- License: MIT.

## PyYAML

- Project: PyYAML
- Upstream: [https://github.com/yaml/pyyaml](https://github.com/yaml/pyyaml)
- License: MIT.

## Apache Arrow / pyarrow

- Project: Apache Arrow (`pyarrow` Python bindings)
- Upstream: [https://github.com/apache/arrow](https://github.com/apache/arrow)
- License: Apache 2.0.
- Notes: Optional runtime dependency, installed via the `[lerobot]` extra.

## UMI-Dex

- Project: UMI-Dex (Linkerbot)
- Upstream: [https://github.com/Linkerbot/UMI-Dex](https://github.com/Linkerbot/UMI-Dex)
- License: Apache 2.0.
- Notes: Used by the UMI bag → replay pipeline. Currently consumed via
  a path-based import pending PyPI publication of `umi-dex`. Will move
  to a normal optional dependency under the `[umi-replay]` extra.

## Robot meshes

The 3D meshes shipped under `assets/components/` are derived from
manufacturer CAD released as open-source by their original authors:

- Rokae arm meshes (AR5 family) — Rokae, released under their
  open-source terms.
- Linkerhand meshes (L6, O6, L25, glove) — Linkerbot, released under
  their open-source terms.
- A7 lite arm meshes (A7 family) — Linkerbot.
- P7 arm meshes — Linkerbot.

Each mesh is included in this repository in good faith based on the
upstream open-source release. Redistributors should retain the upstream
attribution.

## Responsibility

End users and redistributors are responsible for ensuring full
compliance with all applicable third-party licenses when shipping
source, binaries, containers, or integrated products.
