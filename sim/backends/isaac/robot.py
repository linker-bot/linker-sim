"""Isaac-side `Robot` implementation.

Wraps a single `isaaclab.assets.Articulation` and exposes it through
`sim.backends.base.Robot`. The adapter is intentionally thin: it caches
per-role joint-id tensors at construction and routes state reads /
command writes straight to the articulation.

The math for `ee_pose_b` / `ee_vel_b` mirrors the legacy
`TestOscRLEnv._compute_ee_pose_b` helpers so controllers see the same
numbers as before (parity-tested in `tests/test_osc_parity.py`).
"""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation

from sim.registry import WorkstationHandle


class IsaacRobot:
    """Isaac implementation of `sim.backends.base.Robot`."""

    def __init__(self, articulation: Articulation, handle: WorkstationHandle):
        self._art = articulation
        self.handle = handle
        self.num_envs = articulation.num_instances
        self.device = torch.device(articulation.device)

        # Per-role joint-id tensors. Actuated + mimic are both included
        # because Isaac's implicit actuator still needs a drive on mimic
        # joints — the URDF <mimic> tag enforces the coupling at solve.
        self._joint_ids_by_role: dict[str, torch.Tensor] = {}
        self._actuated_ids_by_role: dict[str, torch.Tensor] = {}
        for role, actuated in handle.joints.items():
            mimic = handle.mimic_joints.get(role, [])
            combined = list(actuated) + list(mimic)
            if not combined:
                continue
            # find_joints preserves the order of the input list when no
            # regex expansion happens, which is what we rely on.
            full_ids, _ = articulation.find_joints(combined)
            act_ids, _ = articulation.find_joints(list(actuated))
            self._joint_ids_by_role[role] = torch.as_tensor(
                full_ids, dtype=torch.long, device=self.device
            )
            self._actuated_ids_by_role[role] = torch.as_tensor(
                act_ids, dtype=torch.long, device=self.device
            )

        # Cache the EE body id for the default EE frame.
        self._ee_body_ids: dict[str, int] = {}
        self._jacobi_body_ids: dict[str, int] = {}
        self._resolve_body(self.handle.ee_link)  # warm the caches

    # ---------- frame resolution ----------------------------------------- #

    def _resolve_frame(self, frame: str | None) -> str:
        """Accept a link name, a `"role:frame_name"` reference, or None
        (defaults to `handle.ee_link`). Returns the prefixed link name."""
        if frame is None:
            return self.handle.ee_link
        if ":" in frame:
            resolved = self.handle.frames.get(frame)
            if resolved is None:
                raise KeyError(
                    f"frame {frame!r} not in handle.frames "
                    f"(available: {list(self.handle.frames)})"
                )
            return resolved
        return frame

    def _resolve_body(self, frame: str) -> tuple[int, int]:
        if frame in self._ee_body_ids:
            return self._ee_body_ids[frame], self._jacobi_body_ids[frame]
        body_ids, _ = self._art.find_bodies(frame)
        if len(body_ids) != 1:
            raise ValueError(
                f"expected exactly one body for frame {frame!r}, found {len(body_ids)}"
            )
        body_id = int(body_ids[0])
        jacobi_id = body_id - 1 if self._art.is_fixed_base else body_id
        self._ee_body_ids[frame] = body_id
        self._jacobi_body_ids[frame] = jacobi_id
        return body_id, jacobi_id

    def joint_ids_of(self, role: str) -> torch.Tensor:
        if role not in self._joint_ids_by_role:
            raise KeyError(
                f"role {role!r} not in handle.joints "
                f"(available: {list(self._joint_ids_by_role)})"
            )
        return self._joint_ids_by_role[role]

    def actuated_joint_ids_of(self, role: str) -> torch.Tensor:
        """Subset of `joint_ids_of(role)` excluding mimic joints. OSC
        needs this to dimension mass_matrix/jacobian correctly."""
        if role not in self._actuated_ids_by_role:
            raise KeyError(role)
        return self._actuated_ids_by_role[role]

    def body_id_of(self, frame: str) -> int:
        return self._resolve_body(self._resolve_frame(frame))[0]

    def jacobi_body_id_of(self, frame: str) -> int:
        return self._resolve_body(self._resolve_frame(frame))[1]

    # ---------- state readers -------------------------------------------- #

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._art.data.joint_pos

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._art.data.joint_vel

    @property
    def joint_pos_default(self) -> torch.Tensor:
        return self._art.data.default_joint_pos

    @property
    def joint_vel_default(self) -> torch.Tensor:
        return self._art.data.default_joint_vel

    def ee_pose_b(self, frame: str | None = None) -> torch.Tensor:
        body_id = self._resolve_body(self._resolve_frame(frame))[0]
        ee_pos_w = self._art.data.body_pos_w[:, body_id]
        ee_quat_w = self._art.data.body_quat_w[:, body_id]
        root_pos_w = self._art.data.root_pos_w
        root_quat_w = self._art.data.root_quat_w
        ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
        )
        return torch.cat([ee_pos_b, ee_quat_b], dim=-1)

    def ee_vel_b(self, frame: str | None = None) -> torch.Tensor:
        body_id = self._resolve_body(self._resolve_frame(frame))[0]
        ee_vel_w = self._art.data.body_vel_w[:, body_id, :]
        rel_vel_w = ee_vel_w - self._art.data.root_vel_w
        out = torch.zeros_like(rel_vel_w)
        root_quat_w = self._art.data.root_quat_w
        out[:, 0:3] = math_utils.quat_apply_inverse(root_quat_w, rel_vel_w[:, 0:3])
        out[:, 3:6] = math_utils.quat_apply_inverse(root_quat_w, rel_vel_w[:, 3:6])
        return out

    def mass_matrix(self, role: str) -> torch.Tensor:
        ids = self.actuated_joint_ids_of(role)
        mm = self._art.root_physx_view.get_generalized_mass_matrices()
        return mm[:, ids, :][:, :, ids]

    def jacobian(self, role: str, frame: str | None = None) -> torch.Tensor:
        ids = self.actuated_joint_ids_of(role)
        jacobi_id = self.jacobi_body_id_of(frame if frame is not None else self.handle.ee_link)
        J = self._art.root_physx_view.get_jacobians()
        return J[:, jacobi_id, :, :][:, :, ids]

    def gravity(self, role: str) -> torch.Tensor:
        ids = self.actuated_joint_ids_of(role)
        g = self._art.root_physx_view.get_gravity_compensation_forces()
        return g[:, ids]

    # ---------- command writers ------------------------------------------ #

    def set_joint_effort(self, efforts: torch.Tensor, joint_ids: torch.Tensor) -> None:
        self._art.set_joint_effort_target(efforts, joint_ids=joint_ids)

    def set_joint_position_target(
        self, targets: torch.Tensor, joint_ids: torch.Tensor
    ) -> None:
        self._art.set_joint_position_target(targets, joint_ids=joint_ids)

    # ---------- reset primitives ----------------------------------------- #

    def write_joint_state(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        self._art.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    def write_root_state(
        self,
        root_pose: torch.Tensor,
        root_velocity: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        self._art.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        self._art.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)

    # ---------- actuator gains ------------------------------------------- #

    def write_gains(
        self,
        role: str,
        stiffness: float | torch.Tensor,
        damping: float | torch.Tensor,
    ) -> None:
        ids = self.joint_ids_of(role)
        n = ids.shape[0]

        def _broadcast(x: float | torch.Tensor) -> torch.Tensor:
            if isinstance(x, torch.Tensor):
                if x.ndim == 0:
                    return x.expand(self.num_envs, n).to(self.device)
                if x.ndim == 1 and x.shape[0] == n:
                    return x.unsqueeze(0).expand(self.num_envs, n).to(self.device)
                if x.shape == (self.num_envs, n):
                    return x.to(self.device)
                raise ValueError(f"gain tensor shape {tuple(x.shape)} incompatible with ({self.num_envs}, {n})")
            return torch.full((self.num_envs, n), float(x), device=self.device)

        self._art.write_joint_stiffness_to_sim(_broadcast(stiffness), joint_ids=ids)
        self._art.write_joint_damping_to_sim(_broadcast(damping), joint_ids=ids)
