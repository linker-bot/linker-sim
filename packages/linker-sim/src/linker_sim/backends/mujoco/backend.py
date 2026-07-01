"""MuJoCo `SimBackend` implementation.

Loads a composed `workstation.mjcf` via `sim.registry`, runs B=1 physics
on CPU, and exposes one `MujocoRobot` per cfg.workstations entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from linker_sim.backends.mujoco.robot import MujocoRobot
from linker_sim.registry import load as load_workstation

try:
    import mujoco
except ImportError:
    mujoco = None  # type: ignore[assignment]


@dataclass
class MujocoBackendCfg:
    workstations: dict[str, str] = field(default_factory=lambda: {"robot": "ar5_o6_bench_bimanual"})
    num_envs: int = 1
    dt: float = 1.0 / 500.0
    device: str = "cpu"


class MujocoSimBackend:
    """Concrete MuJoCo backend. See module docstring."""

    def __init__(self, cfg: MujocoBackendCfg):
        if mujoco is None:
            raise ImportError(
                "mujoco is not installed. Install with "
                "`pip install 'linker-sim[mujoco]'`."
            )
        if cfg.num_envs != 1:
            raise NotImplementedError(
                f"MuJoCo backend supports num_envs=1 only (got {cfg.num_envs}). "
                "Parallel rollouts use one process per env (D8)."
            )
        if str(cfg.device) != "cpu":
            raise ValueError(
                f"MuJoCo backend requires device='cpu' (got {cfg.device!r})"
            )
        if len(cfg.workstations) != 1:
            raise NotImplementedError(
                "Multi-articulation scenes are not supported. Use one composed "
                "workstation per scene, e.g. workstations={'robot': 'ar5_o6_bench_bimanual'}."
            )

        self.cfg = cfg
        (robot_name, ws_name), = cfg.workstations.items()

        handle = load_workstation(ws_name)
        if handle.mjcf_path is None:
            raise FileNotFoundError(
                f"{ws_name}: workstation.mjcf missing — run "
                f"`python -m linker_robot_assets.composer.compose assets/workstations/{ws_name}`"
            )

        self._model = mujoco.MjModel.from_xml_path(str(handle.mjcf_path))
        self._model.opt.timestep = float(cfg.dt)
        self._mimic_couplings = self._extract_and_disable_mimic_equalities()
        self._data = mujoco.MjData(self._model)
        mujoco.mj_forward(self._model, self._data)

        self.num_envs = 1
        self.device = torch.device("cpu")
        self.dt = float(cfg.dt)
        self.env_origins = torch.zeros(1, 3)

        robot = MujocoRobot(self._model, self._data, handle)
        robot._mimic_snap = self._apply_mimic_snap  # write_joint_state uses this
        self.robots: dict[str, MujocoRobot] = {robot_name: robot}
        self.rigid_bodies: dict = {}

        self._default_qpos = self._model.qpos0.copy()
        self._default_qvel = np.zeros(self._model.nv, dtype=np.float64)

    def _extract_and_disable_mimic_equalities(self) -> list[tuple[int, int, int, int, np.ndarray, float, float]]:
        """Read <equality joint> constraints, disable them, return coupling tuples.

        MuJoCo's soft equality solver cannot stably enforce joint-to-joint
        coupling when one side has near-zero inertia (e.g. Linker hand DIP
        joints): under-damped mimic joints explode, over-damped ones don't
        track. Since `<mimic>` in URDF is a kinematic relation anyway, we
        disable the physics constraint and snap qpos/qvel of the mimic
        joint after every step (see `_apply_mimic_snap`).

        The tuple also carries the mimic joint's own [lo, hi] range so the
        snap can clip. The o6 left thumb mimic ratio (1.8788) can drive
        the IP joint past its authored 1.08 rad range when cmc_pitch is
        at its own upper limit (0.58 rad) — without clipping, MuJoCo's
        joint-limit constraint fights the snap every step and produces a
        visible wiggle.
        """
        eq_joint = int(mujoco.mjtEq.mjEQ_JOINT)
        couplings: list[tuple[int, int, int, int, np.ndarray, float, float]] = []
        for i in range(self._model.neq):
            if int(self._model.eq_type[i]) != eq_joint:
                continue
            j_mim = int(self._model.eq_obj1id[i])
            j_act = int(self._model.eq_obj2id[i])
            poly = np.asarray(self._model.eq_data[i][:5], dtype=np.float64).copy()
            lo, hi = self._model.jnt_range[j_mim]
            couplings.append((
                int(self._model.jnt_qposadr[j_mim]),
                int(self._model.jnt_qposadr[j_act]),
                int(self._model.jnt_dofadr[j_mim]),
                int(self._model.jnt_dofadr[j_act]),
                poly,
                float(lo),
                float(hi),
            ))
            self._model.eq_active0[i] = 0
        return couplings

    def _apply_mimic_snap(self) -> None:
        """Snap each mimic joint to `polycoef(q_actuated)` in qpos.

        `polycoef = [c0, c1, c2, c3, c4]` and the MuJoCo convention is
        `q_mim - r_mim = sum_k c_k * (q_act - r_act)^k` (reference offsets
        are zero for hinge joints on this workstation). We clip to the
        mimic joint's [lo, hi] range — otherwise the joint-limit
        constraint fights the snap and the joint visibly wiggles.

        qvel_mim is zeroed rather than set to the analytic derivative:
        because the equality constraint is disabled, propagating a
        synthetic velocity into the mimic DOF leaks kinetic energy back
        through the mass matrix into the actuated joint (which is very
        low-inertia here) and drives it into high-frequency limit-cycle
        oscillation. Zero velocity is safe — qpos is snapped every step
        anyway, so the mimic tracks perfectly regardless.
        """
        if not self._mimic_couplings:
            return
        qpos = self._data.qpos
        qvel = self._data.qvel
        for m_qa, a_qa, m_dof, a_dof, poly, lo, hi in self._mimic_couplings:
            q = qpos[a_qa]
            q_target = poly[0] + q * (poly[1] + q * (poly[2] + q * (poly[3] + q * poly[4])))
            if q_target > hi:
                qpos[m_qa] = hi
            elif q_target < lo:
                qpos[m_qa] = lo
            else:
                qpos[m_qa] = q_target
            qvel[m_dof] = 0.0
        mujoco.mj_kinematics(self._model, self._data)

    def step(self) -> None:
        mujoco.mj_step(self._model, self._data)
        self._apply_mimic_snap()

    def write_data(self) -> None:
        pass

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        del env_ids
        self._data.qpos[:] = self._default_qpos
        self._data.qvel[:] = self._default_qvel
        self._data.ctrl[:] = 0.0
        self._data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self._model, self._data)
        self._apply_mimic_snap()

    def close(self) -> None:
        pass
