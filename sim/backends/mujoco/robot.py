"""MuJoCo `Robot` implementation.

Wraps `mujoco.MjModel` + `mujoco.MjData` for one composed workstation.
Joint columns are packed in manifest role order (actuated then mimic per
role). B=1; tensors are CPU torch.
"""

from __future__ import annotations

import numpy as np
import torch

from sim.registry import WorkstationHandle

try:
    import mujoco
except ImportError:
    mujoco = None  # type: ignore[assignment]


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[..., 1:4] = -out[..., 1:4]
    return out


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def _quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector `v` by the inverse of quaternion `q` (wxyz)."""
    q_conj = _quat_conjugate(q)
    vq = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
    return _quat_mul(_quat_mul(q_conj, vq), q)[..., 1:4]


def _subtract_frame_transforms(
    parent_pos: torch.Tensor,
    parent_quat: torch.Tensor,
    child_pos: torch.Tensor,
    child_quat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rel_pos = _quat_apply_inverse(parent_quat, child_pos - parent_pos)
    rel_quat = _quat_mul(_quat_conjugate(parent_quat), child_quat)
    return rel_pos, rel_quat


class MujocoRobot:
    """MuJoCo implementation of `sim.backends.base.Robot`."""

    def __init__(self, model: "mujoco.MjModel", data: "mujoco.MjData", handle: WorkstationHandle):
        if mujoco is None:
            raise ImportError("mujoco is not installed")

        self._model = model
        self._data = data
        self.handle = handle
        self.num_envs = 1
        self.device = torch.device("cpu")

        if model.njnt != model.nv:
            raise ValueError(
                f"expected njnt == nv (1-DOF joints only), got njnt={model.njnt} nv={model.nv}"
            )

        self._root_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, handle.base_link
        )
        if self._root_body_id < 0:
            raise ValueError(f"base_link {handle.base_link!r} not found in MJCF")

        # Pack joints: role order from manifest, actuated then mimic per role.
        self._joint_names: list[str] = []
        self._joint_mj_ids: list[int] = []
        self._qpos_adr: list[int] = []
        self._dof_adr: list[int] = []

        for role in handle.joints:
            actuated = list(handle.joints.get(role, []))
            mimic = list(handle.mimic_joints.get(role, []))
            for name in actuated + mimic:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid < 0:
                    raise ValueError(f"joint {name!r} (role {role}) not in MJCF")
                self._joint_names.append(name)
                self._joint_mj_ids.append(jid)
                self._qpos_adr.append(int(model.jnt_qposadr[jid]))
                self._dof_adr.append(int(model.jnt_dofadr[jid]))

        self._n_joints = len(self._joint_names)
        self._qpos_adr_np = np.asarray(self._qpos_adr, dtype=np.int32)
        self._dof_adr_np = np.asarray(self._dof_adr, dtype=np.int32)

        self._joint_ids_by_role: dict[str, torch.Tensor] = {}
        self._actuated_ids_by_role: dict[str, torch.Tensor] = {}
        self._dof_ids_by_role: dict[str, torch.Tensor] = {}
        self._actuated_dof_ids_by_role: dict[str, torch.Tensor] = {}

        col = 0
        for role in handle.joints:
            actuated = list(handle.joints.get(role, []))
            mimic = list(handle.mimic_joints.get(role, []))
            combined = actuated + mimic
            if not combined:
                continue
            n = len(combined)
            full_cols = torch.arange(col, col + n, dtype=torch.long, device=self.device)
            self._joint_ids_by_role[role] = full_cols
            n_act = len(actuated)
            self._actuated_ids_by_role[role] = full_cols[:n_act]
            dof_cols = torch.tensor(
                [self._dof_adr[col + i] for i in range(n)],
                dtype=torch.long,
                device=self.device,
            )
            self._dof_ids_by_role[role] = dof_cols
            self._actuated_dof_ids_by_role[role] = dof_cols[:n_act]
            col += n

        # joint_id (MuJoCo) -> packed column index
        self._col_by_mj_joint: dict[int, int] = {
            mj_id: i for i, mj_id in enumerate(self._joint_mj_ids)
        }

        # MuJoCo joint id -> actuator id (position actuators only)
        self._actuator_for_joint: dict[int, int] = {}
        for a in range(model.nu):
            jid = int(model.actuator_trnid[a, 0])
            if jid >= 0:
                self._actuator_for_joint[jid] = a

        self._default_qpos = torch.tensor(
            model.qpos0[self._qpos_adr_np], dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        self._default_qvel = torch.zeros(1, self._n_joints, device=self.device)

        self._body_ids: dict[str, int] = {}
        self._resolve_body(self.handle.ee_link)

        mujoco.mj_forward(model, data)
        root_pos = np.array(data.xpos[self._root_body_id])
        root_quat = np.array(data.xquat[self._root_body_id])
        if np.linalg.norm(root_pos) > 1e-4 or abs(root_quat[0] - 1.0) > 1e-3:
            raise ValueError(
                f"expected fixed base {handle.base_link!r} at identity at qpos=0, "
                f"got pos={root_pos} quat={root_quat}"
            )

    # ---------- helpers -------------------------------------------------- #

    def _pack_qpos(self) -> torch.Tensor:
        q = self._data.qpos[self._qpos_adr_np]
        return torch.from_numpy(np.asarray(q, dtype=np.float32)).unsqueeze(0)

    def _pack_qvel(self) -> torch.Tensor:
        v = self._data.qvel[self._dof_adr_np]
        return torch.from_numpy(np.asarray(v, dtype=np.float32)).unsqueeze(0)

    def _resolve_frame(self, frame: str | None) -> str:
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

    def _resolve_body(self, frame: str) -> int:
        if frame in self._body_ids:
            return self._body_ids[frame]
        link = self._resolve_frame(frame)
        bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, link)
        if bid < 0:
            raise ValueError(f"body {link!r} not found in MJCF")
        self._body_ids[frame] = bid
        return bid

    def _body_pose_w(self, body_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        pos = torch.from_numpy(
            np.array(self._data.xpos[body_id], dtype=np.float32)
        ).unsqueeze(0)
        quat = torch.from_numpy(
            np.array(self._data.xquat[body_id], dtype=np.float32)
        ).unsqueeze(0)
        return pos, quat

    def _body_velocity_w(self, body_id: int) -> torch.Tensor:
        vel = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self._model, self._data, mujoco.mjtObj.mjOBJ_BODY, body_id, vel, 0
        )
        return torch.from_numpy(vel.astype(np.float32)).unsqueeze(0)

    # ---------- Robot interface ------------------------------------------ #

    def joint_ids_of(self, role: str) -> torch.Tensor:
        if role not in self._joint_ids_by_role:
            raise KeyError(
                f"role {role!r} not in handle.joints "
                f"(available: {list(self._joint_ids_by_role)})"
            )
        return self._joint_ids_by_role[role]

    def actuated_joint_ids_of(self, role: str) -> torch.Tensor:
        if role not in self._actuated_ids_by_role:
            raise KeyError(role)
        return self._actuated_ids_by_role[role]

    def body_id_of(self, frame: str) -> int:
        return self._resolve_body(self._resolve_frame(frame))

    def jacobi_body_id_of(self, frame: str) -> int:
        return self.body_id_of(frame)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._pack_qpos()

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._pack_qvel()

    @property
    def joint_pos_default(self) -> torch.Tensor:
        return self._default_qpos.clone()

    @property
    def joint_vel_default(self) -> torch.Tensor:
        return self._default_qvel.clone()

    def ee_pose_b(self, frame: str | None = None) -> torch.Tensor:
        body_id = self._resolve_body(self._resolve_frame(frame))
        ee_pos, ee_quat = self._body_pose_w(body_id)
        root_pos, root_quat = self._body_pose_w(self._root_body_id)
        pos_b, quat_b = _subtract_frame_transforms(root_pos, root_quat, ee_pos, ee_quat)
        return torch.cat([pos_b, quat_b], dim=-1)

    def ee_vel_b(self, frame: str | None = None) -> torch.Tensor:
        body_id = self._resolve_body(self._resolve_frame(frame))
        ee_vel = self._body_velocity_w(body_id)
        root_vel = self._body_velocity_w(self._root_body_id)
        rel_vel = ee_vel - root_vel
        _, root_quat = self._body_pose_w(self._root_body_id)
        out = torch.zeros_like(rel_vel)
        out[:, 0:3] = _quat_apply_inverse(root_quat, rel_vel[:, 0:3])
        out[:, 3:6] = _quat_apply_inverse(root_quat, rel_vel[:, 3:6])
        return out

    def mass_matrix(self, role: str) -> torch.Tensor:
        dofs = self._actuated_dof_ids_by_role[role]
        dof_list = dofs.detach().cpu().numpy().tolist()
        full = np.zeros((self._model.nv, self._model.nv), dtype=np.float64)
        mujoco.mj_fullM(self._model, full, self._data.qM)
        sub = full[np.ix_(dof_list, dof_list)]
        return torch.from_numpy(sub.astype(np.float32)).unsqueeze(0)

    def jacobian(self, role: str, frame: str | None = None) -> torch.Tensor:
        body_id = self._resolve_body(
            self._resolve_frame(frame if frame is not None else self.handle.ee_link)
        )
        dofs = self._actuated_dof_ids_by_role[role]
        dof_list = dofs.detach().cpu().numpy().tolist()
        jacp = np.zeros((3, self._model.nv), dtype=np.float64)
        jacr = np.zeros((3, self._model.nv), dtype=np.float64)
        mujoco.mj_jacBody(self._model, self._data, jacp, jacr, body_id)
        J = np.vstack([jacp, jacr])[:, dof_list]
        return torch.from_numpy(J.astype(np.float32)).unsqueeze(0)

    def gravity(self, role: str) -> torch.Tensor:
        dofs = self._actuated_dof_ids_by_role[role]
        dof_list = dofs.detach().cpu().numpy().tolist()
        qvel_saved = self._data.qvel.copy()
        qacc_saved = self._data.qacc.copy()
        self._data.qvel[:] = 0.0
        self._data.qacc[:] = 0.0
        result = np.zeros(self._model.nv, dtype=np.float64)
        mujoco.mj_rne(self._model, self._data, 0, result)
        self._data.qvel[:] = qvel_saved
        self._data.qacc[:] = qacc_saved
        g = result[dof_list]
        return torch.from_numpy(g.astype(np.float32)).unsqueeze(0)

    def set_joint_effort(self, efforts: torch.Tensor, joint_ids: torch.Tensor) -> None:
        eff = efforts.detach().cpu().numpy().reshape(-1)
        cols = joint_ids.detach().cpu().numpy().tolist()
        for i, col in enumerate(cols):
            dof = self._dof_adr[col]
            self._data.qfrc_applied[dof] = float(eff[i])

    def set_joint_position_target(
        self, targets: torch.Tensor, joint_ids: torch.Tensor
    ) -> None:
        tgt = targets.detach().cpu().numpy().reshape(-1)
        cols = joint_ids.detach().cpu().numpy().tolist()
        for i, col in enumerate(cols):
            mj_jid = self._joint_mj_ids[col]
            act_id = self._actuator_for_joint.get(mj_jid)
            if act_id is None:
                raise ValueError(
                    f"no position actuator for joint {self._joint_names[col]!r}"
                )
            self._data.ctrl[act_id] = float(tgt[i])

    def write_joint_state(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        del env_ids  # B=1 only
        jp = joint_pos[0].detach().cpu().numpy()
        jv = joint_vel[0].detach().cpu().numpy()
        for i, col in enumerate(range(self._n_joints)):
            self._data.qpos[self._qpos_adr[col]] = float(jp[col])
            self._data.qvel[self._dof_adr[col]] = float(jv[col])
        mujoco.mj_forward(self._model, self._data)

    def write_root_state(
        self,
        root_pose: torch.Tensor,
        root_velocity: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        del env_ids
        pos = root_pose[0, :3].detach().cpu().numpy()
        quat = root_pose[0, 3:7].detach().cpu().numpy()
        if np.linalg.norm(pos) > 1e-4 or abs(quat[0] - 1.0) > 1e-3:
            raise NotImplementedError(
                "floating-base root writes are not supported on the MuJoCo backend"
            )

    def write_gains(
        self,
        role: str,
        stiffness: float | torch.Tensor,
        damping: float | torch.Tensor,
    ) -> None:
        ids = self.actuated_joint_ids_of(role)
        n = int(ids.shape[0])

        def _scalar_values(x: float | torch.Tensor) -> list[float]:
            if isinstance(x, torch.Tensor):
                if x.ndim == 0:
                    return [float(x)] * n
                if x.ndim == 1 and x.shape[0] == n:
                    return [float(v) for v in x.detach().cpu()]
                if x.shape == (1, n):
                    return [float(v) for v in x[0].detach().cpu()]
                raise ValueError(
                    f"gain tensor shape {tuple(x.shape)} incompatible with n={n}"
                )
            return [float(x)] * n

        kps = _scalar_values(stiffness)
        kds = _scalar_values(damping)
        cols = ids.detach().cpu().numpy().tolist()
        for col, kp, kd in zip(cols, kps, kds, strict=True):
            mj_jid = self._joint_mj_ids[col]
            act_id = self._actuator_for_joint.get(mj_jid)
            if act_id is None:
                continue
            self._model.actuator_gainprm[act_id, 0] = kp
            self._model.actuator_biasprm[act_id, 1] = -kp
            dof = self._dof_adr[col]
            self._model.dof_damping[dof] = kd
