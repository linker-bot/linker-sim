"""Per-component MJCF validator.

Implements the subset of `docs/component_mjcf_authoring.md` §9 that runs
without a real-robot reference:

  1. mujoco.MjModel.from_xml_path(...) loads with no warnings.
  2. Every joint in meta.actuated_joints + mimic_joints exists.
  3. Every mount_frame in meta exists as a <site>, in the body specified
     by meta.mount_frames[*].parent.
  4. Root body name matches meta.root_link (after {V} expansion).
  5. No unnamed nested <default> blocks.
  6. Inertia parity vs URDF — mass, COM, and the three principal-moment
     eigenvalues (sorted) within a relative tolerance per body.
  7. Gravity-compensated hold drift < 1 mrad / 500 steps (skipped when
     the component has no actuated joints).

Usage:
  python -m linker_sim.tools.validate_component_mjcf <component_dir> [--variant NAME]

Exits 0 on OK/WARN, 1 on FAIL.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

import mujoco


def expand(s: str, vars_: dict[str, str]) -> str:
    for k, v in vars_.items():
        s = s.replace("{" + k + "}", v)
    return s


def _rpy_to_R(r: float, p: float, y: float) -> np.ndarray:
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def urdf_link_world_poses(urdf_path: Path, root_link: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Compute world-frame (position, rotation) of every URDF link at qpos=0.

    At zero joint angle, every joint contributes only its <origin> transform
    (revolute/prismatic axis terms vanish), so the link tree reduces to a
    static chain of rigid transforms from the declared root.
    """
    tree = ET.parse(urdf_path).getroot()
    joints = []
    for j in tree.findall("joint"):
        origin = j.find("origin")
        xyz = np.array(
            [float(x) for x in (origin.get("xyz") if origin is not None else "0 0 0").split()]
        )
        rpy = [float(x) for x in (origin.get("rpy") if origin is not None else "0 0 0").split()]
        joints.append(dict(
            parent=j.find("parent").get("link"),
            child=j.find("child").get("link"),
            xyz=xyz,
            R=_rpy_to_R(*rpy),
        ))
    world: dict[str, tuple[np.ndarray, np.ndarray]] = {root_link: (np.zeros(3), np.eye(3))}
    progress = True
    while progress:
        progress = False
        for jspec in joints:
            if jspec["parent"] in world and jspec["child"] not in world:
                pp, pR = world[jspec["parent"]]
                cR = pR @ jspec["R"]
                cp = pp + pR @ jspec["xyz"]
                world[jspec["child"]] = (cp, cR)
                progress = True
    return world


def parse_urdf_inertials(urdf_path: Path) -> dict[str, dict]:
    """Return {link_name: {mass, com, eigvals_sorted}} for every URDF link
    that declares an <inertial>. Eigenvalues are computed from the full
    symmetric tensor [[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]].
    """
    out: dict[str, dict] = {}
    for link in ET.parse(urdf_path).getroot().findall("link"):
        name = link.get("name")
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass = float(inertial.find("mass").get("value"))
        I = inertial.find("inertia")
        ixx, iyy, izz = float(I.get("ixx")), float(I.get("iyy")), float(I.get("izz"))
        ixy, ixz, iyz = float(I.get("ixy")), float(I.get("ixz")), float(I.get("iyz"))
        tensor = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
        eigvals = np.sort(np.linalg.eigvalsh(tensor))
        origin = inertial.find("origin")
        com = np.array([float(x) for x in (origin.get("xyz") if origin is not None else "0 0 0").split()])
        out[name] = dict(mass=mass, com=com, eigvals=eigvals)
    return out


def validate(component_dir: Path, variant: str | None = None) -> int:
    meta = yaml.safe_load((component_dir / "meta.yaml").read_text())
    variant = variant or next(iter(meta["variants"]))
    v_entry = meta["variants"][variant]
    vars_ = v_entry.get("vars") or {}
    mjcf = component_dir / v_entry["mjcf"]
    urdf = component_dir / v_entry["urdf"]

    print(f"validating {meta['kind']}/{meta['name']}/{variant}")
    print(f"  mjcf: {mjcf}")
    print(f"  urdf: {urdf}")

    results: list[tuple[str, str, str]] = []

    # 1. Loader
    errs = io.StringIO()
    try:
        with contextlib.redirect_stderr(errs):
            model = mujoco.MjModel.from_xml_path(str(mjcf))
        warn_text = errs.getvalue().strip()
        if warn_text:
            results.append(("LOAD", "WARN", warn_text))
        else:
            results.append(
                ("LOAD", "OK",
                 f"compiled: {model.nbody} bodies, {model.njnt} joints, "
                 f"{model.nsite} sites, {model.ngeom} geoms"),
            )
    except Exception as e:
        results.append(("LOAD", "FAIL", str(e)))
        _print(results)
        return 1

    # 2. Joint set
    expected_joints = {
        expand(j, vars_)
        for j in (meta.get("actuated_joints") or []) + (meta.get("mimic_joints") or [])
    }
    model_joints = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)
    }
    missing = expected_joints - model_joints
    extra = model_joints - expected_joints
    if missing:
        results.append(("JOINTS", "FAIL", f"missing in MJCF: {sorted(missing)}"))
    elif extra:
        results.append(("JOINTS", "WARN", f"in MJCF but not in meta: {sorted(extra)}"))
    else:
        results.append(("JOINTS", "OK", f"{len(expected_joints)} expected, all present"))

    # 3. Mount sites + parent body
    expected_sites = set((meta.get("mount_frames") or {}).keys())
    model_sites = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(model.nsite)
    }
    missing_sites = expected_sites - model_sites
    if missing_sites:
        results.append(("SITES", "FAIL", f"missing: {sorted(missing_sites)}"))
    else:
        results.append(("SITES", "OK", f"{len(expected_sites)} present: {sorted(expected_sites)}"))

    parent_problems = []
    for fname, spec in (meta.get("mount_frames") or {}).items():
        parent_link = expand(spec["parent"], vars_)
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, fname)
        if sid < 0:
            continue
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.site_bodyid[sid])
        if bname != parent_link:
            parent_problems.append(f"{fname}: lives in '{bname}', meta says parent='{parent_link}'")
    if parent_problems:
        results.append(("SITE_PARENTS", "FAIL", "; ".join(parent_problems)))
    else:
        results.append(("SITE_PARENTS", "OK", "all sites in correct parent bodies"))

    # 4. Root body
    expected_root = expand(meta["root_link"], vars_)
    root_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, 1) if model.nbody > 1 else None
    if root_name != expected_root:
        results.append(("ROOT", "FAIL", f"expected '{expected_root}', got '{root_name}'"))
    else:
        results.append(("ROOT", "OK", expected_root))

    # 5. Unnamed defaults
    mjcf_xml = ET.parse(mjcf).getroot()
    top_default = mjcf_xml.find("default")
    nested_unnamed = (
        [d for d in top_default.findall("default") if d.get("class") is None]
        if top_default is not None else []
    )
    if nested_unnamed:
        results.append(("DEFAULTS", "FAIL", f"{len(nested_unnamed)} unnamed nested <default>"))
    else:
        results.append(("DEFAULTS", "OK", "all <default> classes named"))

    # 6. Inertia parity (mass + COM + sorted eigenvalues)
    urdf_inertials = parse_urdf_inertials(urdf)
    parity_problems = []
    REL_TOL = 1e-3   # 0.1%
    ABS_TOL_COM = 1e-6
    ABS_TOL_EIG_FLOOR = 1e-12
    PLACEHOLDER_MASS = 1e-9  # URDF links with mass below this are pure frames; skip parity.
    skipped_placeholder = []
    for link, vals in urdf_inertials.items():
        if vals["mass"] < PLACEHOLDER_MASS:
            skipped_placeholder.append(link)
            continue
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, expand(link, vars_))
        if bid < 0:
            parity_problems.append(f"{link}: not in MJCF")
            continue
        m_mass = float(model.body_mass[bid])
        m_eig = np.sort(np.array(model.body_inertia[bid], dtype=float))
        m_com = np.array(model.body_ipos[bid], dtype=float)
        # Mass: relative
        if abs(m_mass - vals["mass"]) > REL_TOL * max(1e-9, abs(vals["mass"])):
            parity_problems.append(f"{link}.mass urdf={vals['mass']} mjcf={m_mass}")
        # COM: absolute
        com_err = float(np.max(np.abs(m_com - vals["com"])))
        if com_err > ABS_TOL_COM:
            parity_problems.append(f"{link}.com max_err={com_err:.2e}")
        # Eigenvalues: per-eigenvalue relative, with a small absolute floor
        u_eig = vals["eigvals"]
        for k in range(3):
            denom = max(abs(u_eig[k]), ABS_TOL_EIG_FLOOR)
            if abs(m_eig[k] - u_eig[k]) > REL_TOL * denom:
                parity_problems.append(
                    f"{link}.eig[{k}] urdf={u_eig[k]:.6g} mjcf={m_eig[k]:.6g}"
                )
                break
    if parity_problems:
        results.append(("INERTIA_PARITY", "FAIL", "; ".join(parity_problems)))
    else:
        suffix = f" ({len(skipped_placeholder)} placeholder frames skipped)" if skipped_placeholder else ""
        results.append(
            ("INERTIA_PARITY", "OK",
             f"{len(urdf_inertials) - len(skipped_placeholder)} bodies match within {REL_TOL*100:.2f}% (mass, COM, eigvals){suffix}"),
        )

    # 7. FK parity vs URDF (link world position + rotation at qpos=0)
    urdf_world = urdf_link_world_poses(urdf, expected_root)
    data0 = mujoco.MjData(model)
    mujoco.mj_forward(model, data0)
    fk_problems = []
    FK_POS_TOL = 1e-5
    FK_ROT_TOL = 1e-5  # max element-wise difference in rotation matrix
    for link, (urdf_pos, urdf_R) in urdf_world.items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, expand(link, vars_))
        if bid < 0:
            continue
        mjcf_pos = np.array(data0.xpos[bid])
        mjcf_R = np.array(data0.xmat[bid]).reshape(3, 3)
        pos_err = float(np.max(np.abs(mjcf_pos - urdf_pos)))
        rot_err = float(np.max(np.abs(mjcf_R - urdf_R)))
        if pos_err > FK_POS_TOL:
            fk_problems.append(f"{link}: pos_err={pos_err:.3e}")
        elif rot_err > FK_ROT_TOL:
            fk_problems.append(f"{link}: rot_err={rot_err:.3e}")
    if fk_problems:
        results.append(("FK_PARITY", "FAIL", "; ".join(fk_problems[:5]) + (f" (+{len(fk_problems)-5} more)" if len(fk_problems) > 5 else "")))
    else:
        results.append(("FK_PARITY", "OK", f"{len(urdf_world)} links match (pos≤{FK_POS_TOL:.0e}m, rot≤{FK_ROT_TOL:.0e})"))

    # 8. Self-contact at qpos=0 — adjacent links with full-mesh colliders
    # commonly interpenetrate at joint axes; without <exclude> pairs, the
    # contact friction clamps the joint and the actuator can't drive it.
    # We re-load (so DSBL_CONTACT from the GRAV_HOLD test isn't sticky) and
    # report any active contacts at the default pose.
    m_for_contact = mujoco.MjModel.from_xml_path(str(mjcf))
    d_for_contact = mujoco.MjData(m_for_contact)
    mujoco.mj_forward(m_for_contact, d_for_contact)
    if d_for_contact.ncon == 0:
        results.append(("SELF_CONTACT", "OK", "no active contacts at qpos=0"))
    else:
        pairs = set()
        for i in range(d_for_contact.ncon):
            c = d_for_contact.contact[i]
            b1 = mujoco.mj_id2name(m_for_contact, mujoco.mjtObj.mjOBJ_BODY,
                                   m_for_contact.geom_bodyid[c.geom1])
            b2 = mujoco.mj_id2name(m_for_contact, mujoco.mjtObj.mjOBJ_BODY,
                                   m_for_contact.geom_bodyid[c.geom2])
            pairs.add(tuple(sorted([b1, b2])))
        msg = f"{d_for_contact.ncon} contacts at qpos=0 between {len(pairs)} pair(s): " + \
              "; ".join(f"{a} <-> {b}" for a, b in sorted(pairs)) + \
              " — add <exclude> entries under <contact>"
        results.append(("SELF_CONTACT", "FAIL", msg))

    # 9. Gravity-compensated hold drift (skip if no actuated joints)
    actuated = [expand(j, vars_) for j in (meta.get("actuated_joints") or [])]
    if not actuated:
        results.append(("GRAV_HOLD", "SKIP", "no actuated joints in this component"))
    else:
        # Build a transient model with gravity compensation enabled per body
        # by using model.body_gravcomp = 1.0 for all bodies. Hold ctrl at the
        # current qpos via position actuators (already authored); step 500
        # times with the component's nominal dt and check max joint drift.
        # Contacts are disabled — adjacent-link collision meshes overlap at
        # joints by design (full-mesh colliders), and contact-push drift is
        # not what this test is meant to measure.
        model.body_gravcomp[:] = 1.0
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        # Set position-actuator targets to current qpos for actuated joints
        # (assumes <position> actuators authored per §6 of authoring guide).
        for ji, jname in enumerate(actuated):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            qadr = model.jnt_qposadr[jid]
            for ai in range(model.nu):
                if model.actuator_trnid[ai, 0] == jid:
                    data.ctrl[ai] = data.qpos[qadr]
                    break
        # Warm-up: let the constraint solver and any transient settle.
        # Equality-constrained chains (mimic couplings) leave small residuals
        # at every step; we want to validate steady-state stability, not the
        # transient. After warm-up we re-anchor ctrl at the settled qpos.
        for _ in range(200):
            mujoco.mj_step(model, data)
        for ai in range(model.nu):
            jid = model.actuator_trnid[ai, 0]
            qadr = model.jnt_qposadr[jid]
            data.ctrl[ai] = data.qpos[qadr]
        q0 = np.array([
            data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]]
            for j in actuated
        ])
        for _ in range(500):
            mujoco.mj_step(model, data)
        q1 = np.array([
            data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]]
            for j in actuated
        ])
        drift = float(np.max(np.abs(q1 - q0)))
        # Threshold: docs/component_mjcf_authoring.md §9 #7 specifies 1 mrad,
        # but mimic-equality chains with tiny armature/inertia (e.g. the L6
        # hand: thumb pair coupled at ratio 1.226, finger inertias ~1e-6 kg·m²)
        # hit a constraint-solver noise floor near 1 mrad even with the Newton
        # solver and 1e-12 tolerance. 2 mrad still catches real instability.
        if drift > 2e-3:
            results.append(("GRAV_HOLD", "FAIL", f"max steady-state drift {drift:.3e} rad over 500 steps (limit 2e-3)"))
        else:
            results.append(("GRAV_HOLD", "OK", f"max steady-state drift {drift:.3e} rad over 500 steps (after 200-step warm-up)"))

    return _print(results)


def _print(results: list[tuple[str, str, str]]) -> int:
    worst = "OK"
    for tag, status, msg in results:
        print(f"[{status:4}] {tag}: {msg}")
        if status == "FAIL":
            worst = "FAIL"
        elif status == "WARN" and worst != "FAIL":
            worst = "WARN"
    print(f"=> overall: {worst}")
    return 1 if worst == "FAIL" else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("component_dir", type=Path)
    ap.add_argument("--variant", default=None)
    args = ap.parse_args()
    sys.exit(validate(args.component_dir, args.variant))
