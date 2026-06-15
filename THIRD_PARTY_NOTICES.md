# Third-Party Notices

This project depends on third-party software with separate licenses.
End users and redistributors are responsible for ensuring full
compliance with all applicable third-party licenses when shipping
source, binaries, containers, or integrated products.

## NVIDIA Isaac Sim

- Project: NVIDIA Isaac Sim
- Upstream: [https://developer.nvidia.com/isaac/sim](https://developer.nvidia.com/isaac/sim)
- License: Dual.
  - The Isaac Sim source-code wrapper is released under the Apache
    License 2.0.
  - The underlying NVIDIA Omniverse Kit SDK and proprietary engines
    (PhysX, Warp, renderers) are governed by the *NVIDIA Isaac Sim
    Additional Software and Materials License*, which is restrictive
    and not redistributable.
- Notes: Isaac Sim is an **optional** runtime dependency, installed by
  end users via the `[isaac]` extra. This project does not redistribute
  any NVIDIA binaries, USDs, or proprietary materials. Users installing
  the `[isaac]` extra agree to NVIDIA's terms.
- Suggested attribution:
  > NVIDIA Corporation. *NVIDIA Isaac Sim: Robotics Simulation and
  > Synthetic Data Generation Platform*. https://developer.nvidia.com/isaac/sim

## NVIDIA Isaac Lab

- Project: NVIDIA Isaac Lab (formerly Orbit)
- Upstream: [https://github.com/isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab)
- License: BSD 3-Clause License (core); Apache 2.0 for select extension
  packages such as `isaaclab_mimic`.
- Notes: Isaac Lab runs inside Isaac Sim. Installed separately by end
  users; not bundled here.
- Suggested citation (per upstream request):
  > Mittal, M., Yu, C., Yu, M., Liu, J., Chourmouzios, L., Kevadiya, A.,
  > et al. (2023). *Orbit: A Unified Simulation Framework for Interactive
  > Robot Learning Platforms*. arXiv:2301.04195.

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

## flatdict

- Project: flatdict
- Upstream: [https://github.com/gmr/flatdict](https://github.com/gmr/flatdict)
- License: BSD 3-Clause.

## Apache Arrow / pyarrow

- Project: Apache Arrow (`pyarrow` Python bindings)
- Upstream: [https://github.com/apache/arrow](https://github.com/apache/arrow)
- License: Apache 2.0.
- Notes: Optional runtime dependency, installed via the `[lerobot]` extra.

## Viser

- Project: Viser
- Upstream: [https://github.com/nerfstudio-project/viser](https://github.com/nerfstudio-project/viser)
- License: Apache 2.0.
- Notes: Optional runtime dependency, installed via the `[viser]` extra
  for the Viser browser-replay backend.

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
- LKLS73 arm meshes — Linkerbot.

Each mesh is included in this repository in good faith based on the
upstream open-source release. Redistributors should retain the upstream
attribution.

## Responsibility

End users and redistributors are responsible for ensuring full
compliance with all applicable third-party licenses when shipping
source, binaries, containers, or integrated products.
