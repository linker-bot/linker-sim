"""Sim-agnostic runtime protocols.

Every controller, task, and env in this repo is written against the
`Robot` and `SimBackend` protocols defined here. Concrete impls live
under `sim/backends/isaac/` and `sim/backends/mujoco/`. Adding a new
simulator means implementing these two protocols — nothing else.

Design notes:

- Tensors at the boundary (D10). Both backends expose torch `(B, ...)`
  tensors; the MuJoCo backend does numpy↔torch conversion internally.
- Role-centric access. Controllers ask for `robot.joint_ids_of("arm")`
  / `mass_matrix("arm")` — never raw global indices. The handle
  (`sim.registry.WorkstationHandle`) is the authoritative source for
  which joints belong to which role.
- `Robot` is a thin adapter, not an owner. The backend owns the
  underlying articulation/model; `Robot` just exposes it under a
  stable API.
- `SimBackend` owns the loop primitives (`step`, `reset`) and the set
  of robots. A backend with multiple articulations exposes them as
  `robots: dict[str, Robot]` keyed by role-name (not prim path).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

from sim.registry import WorkstationHandle


@runtime_checkable
class Robot(Protocol):
    """Sim-agnostic view of a single articulated robot.

    All tensors are `(num_envs, ...)` on `backend.device`. Role names
    match `handle.joints` keys (typically `"arm"`, `"hand"`, `"base"`).
    """

    # -- identification ----------------------------------------------------- #
    handle: WorkstationHandle
    num_envs: int
    device: torch.device

    # -- joint lookups ------------------------------------------------------ #
    def joint_ids_of(self, role: str) -> torch.Tensor:
        """Return `(N,)` long tensor of joint indices for `role`'s
        actuated joints (in manifest order). Includes mimic joints only
        if `role` has them and `include_mimic=True` was requested at
        backend construction."""
        ...

    def body_id_of(self, frame: str) -> int:
        """Return a single body index for the named frame. `frame` may be
        a bare link name or a `"role:frame_name"` lookup resolved via
        `handle.frames`."""
        ...

    def jacobi_body_id_of(self, frame: str) -> int:
        """Jacobian-row index for `frame`. On fixed-base articulations
        this is `body_id_of(frame) - 1`; on floating-base it's the
        same. Controllers using `jacobian()` must use this index."""
        ...

    # -- state readers ------------------------------------------------------ #
    @property
    def joint_pos(self) -> torch.Tensor:
        """`(B, n_joints)` — all joints, full articulation ordering."""
        ...

    @property
    def joint_vel(self) -> torch.Tensor: ...

    @property
    def joint_pos_default(self) -> torch.Tensor: ...

    @property
    def joint_vel_default(self) -> torch.Tensor: ...

    def ee_pose_b(self, frame: str | None = None) -> torch.Tensor:
        """`(B, 7)` — [xyz, quat(wxyz)] of `frame` in the root frame.
        Defaults to `handle.ee_link`."""
        ...

    def ee_vel_b(self, frame: str | None = None) -> torch.Tensor:
        """`(B, 6)` — [lin, ang] velocity of `frame` in the root frame."""
        ...

    def mass_matrix(self, role: str) -> torch.Tensor:
        """`(B, n_role, n_role)` — generalized mass matrix filtered to
        `role`'s actuated joints."""
        ...

    def jacobian(self, role: str, frame: str | None = None) -> torch.Tensor:
        """`(B, 6, n_role)` — geometric Jacobian of `frame` (defaulting
        to `handle.ee_link`) w.r.t. `role`'s actuated joints."""
        ...

    def gravity(self, role: str) -> torch.Tensor:
        """`(B, n_role)` — gravity-compensation torques for `role`'s
        actuated joints."""
        ...

    # -- command writers --------------------------------------------------- #
    def set_joint_effort(self, efforts: torch.Tensor, joint_ids: torch.Tensor) -> None: ...

    def set_joint_position_target(
        self, targets: torch.Tensor, joint_ids: torch.Tensor
    ) -> None: ...

    # -- reset primitives -------------------------------------------------- #
    def write_joint_state(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None: ...

    def write_root_state(
        self,
        root_pose: torch.Tensor,     # (B, 7)
        root_velocity: torch.Tensor, # (B, 6)
        env_ids: torch.Tensor | None = None,
    ) -> None: ...

    # -- actuator gains ---------------------------------------------------- #
    def write_gains(
        self,
        role: str,
        stiffness: float | torch.Tensor,
        damping: float | torch.Tensor,
    ) -> None:
        """Runtime override of actuator gains for `role`. Used by
        controllers that need to swap gain profiles (e.g. OSC's lower
        arm stiffness) after spawn."""
        ...


@runtime_checkable
class RigidBody(Protocol):
    """Sim-agnostic view of a single rigid body (e.g. task object).

    Minimal surface — tasks use it to read object pose/vel and to
    teleport the body on reset. No contact or attachment hooks here;
    those live on the backend when needed (stub for now).
    """

    name: str
    num_envs: int
    device: torch.device

    @property
    def root_pos_w(self) -> torch.Tensor:
        """`(B, 3)` world-frame position."""
        ...

    @property
    def root_quat_w(self) -> torch.Tensor:
        """`(B, 4)` world-frame orientation, wxyz."""
        ...

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        """`(B, 3)` world-frame linear velocity."""
        ...

    def write_root_pose(
        self,
        pose: torch.Tensor,                 # (B, 7)
        env_ids: torch.Tensor | None = None,
    ) -> None: ...

    def write_root_velocity(
        self,
        velocity: torch.Tensor,             # (B, 6)
        env_ids: torch.Tensor | None = None,
    ) -> None: ...


@runtime_checkable
class SimBackend(Protocol):
    """Sim-agnostic simulator frontend.

    Owns the physics loop, scene, and set of robots / rigid bodies.
    `BaseEnv` drives the backend; tasks/controllers go through it to
    read state and write commands.
    """

    num_envs: int
    device: torch.device
    dt: float                           # physics dt (seconds)
    robots: dict[str, Robot]            # role-name -> Robot
    rigid_bodies: dict[str, RigidBody]  # name -> RigidBody
    env_origins: torch.Tensor           # (num_envs, 3)

    def step(self) -> None:
        """Advance physics by one `dt`. Does NOT render unless the
        backend was constructed with a render interval that matches."""
        ...

    def write_data(self) -> None:
        """Flush any queued target/effort writes to the underlying sim
        (Isaac's `scene.write_data_to_sim`; no-op on MuJoCo where writes
        are immediate)."""
        ...

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset `env_ids` (or all envs if None) to the default pose.
        Controllers receive a separate `reset(env_ids)` call from the
        env — this method only touches the sim state."""
        ...

    def close(self) -> None:
        """Shut down the underlying sim context."""
        ...
