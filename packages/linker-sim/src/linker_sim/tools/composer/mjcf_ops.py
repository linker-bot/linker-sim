"""MJCF composition operations.

Reads hand-authored component MJCFs, role-prefixes every name + cross-
reference + default-class scope, rewrites mesh paths so they resolve from
the composed workstation's directory, and merges each component's
top-level sections (`<asset>`, `<default>`, `<worldbody>`, `<actuator>`,
`<equality>`, `<contact>`) into one `<mujoco>` document. The mount tree
is built by reparenting each non-base component's root body under its
mount-frame parent body in the composed `<worldbody>`.

Why this can't reuse urdf_ops.py: MJCF organizes elements very differently
— meshes live under `<asset>` (not inline on visuals), joints nest inside
`<body>` (not flat under `<robot>`), drives sit under `<actuator>`, and
class scoping is governed by `<default>`. None of those map onto URDF's
flat structure, so this is its own composer pass.
"""

from __future__ import annotations

import os.path
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .determinism import fmt_vec
from .schemas import (
    ComponentMeta,
    Mount,
    SchemaError,
    Variant,
)
from .urdf_ops import NameMap

if TYPE_CHECKING:
    from .urdf_ops import CompiledComponent


# --------------------------- Availability check --------------------------- #


@dataclass
class MjcfAvailability:
    """Whether every component declares + ships an MJCF file."""

    all_present: bool
    missing: list[str]  # roles with no MJCF declared or file missing


def check_mjcf_availability(
    components_with_variant: list[tuple[str, ComponentMeta, Variant]],
) -> MjcfAvailability:
    """Inspect each component variant for a usable MJCF source."""
    missing: list[str] = []
    for role, meta, variant in components_with_variant:
        if not variant.mjcf:
            missing.append(f"{role} ({meta.name}/{variant.name}): no mjcf declared")
            continue
        path = meta.source_dir / variant.mjcf
        if not path.is_file():
            missing.append(f"{role} ({meta.name}/{variant.name}): {path} not found")
    return MjcfAvailability(all_present=not missing, missing=missing)


# --------------------------- Name prefixing ------------------------------- #

# Elements that carry a `name=` attribute that we role-prefix.
PREFIXED_NAME_TAGS = frozenset({
    # kinematic + visual
    "body", "joint", "site", "geom", "camera", "light",
    # assets
    "mesh", "material", "texture", "skin", "hfield",
    # tendons, sensors
    "tendon", "sensor",
    # actuators (any of these may appear under <actuator>)
    "general", "motor", "position", "velocity", "intvelocity",
    "damper", "cylinder", "muscle", "adhesion",
})

# Cross-reference attributes — values are names in the same component's
# namespace, so they get the same role prefix.
REF_ATTRS = (
    "joint",
    "joint1",
    "joint2",
    "body",
    "body1",
    "body2",
    "geom",
    "geom1",
    "geom2",
    "site",
    "site1",
    "site2",
    "mesh",
    "material",
    "tendon",
    "tendon1",
    "tendon2",
)


def prefix_mjcf(root: ET.Element, nm: NameMap) -> None:
    """Role-prefix every nameable element, every cross-reference, and every
    default-class scope name. Mutates `root` in place.
    """
    # Pass 1: rename `name=` on elements that own a per-component identity.
    for elem in root.iter():
        if elem.tag in PREFIXED_NAME_TAGS:
            n = elem.attrib.get("name")
            if n:
                elem.attrib["name"] = nm.prefix(n)

    # Pass 2: rewrite cross-references.
    for elem in root.iter():
        for attr in REF_ATTRS:
            v = elem.attrib.get(attr)
            if v:
                elem.attrib[attr] = nm.prefix(v)

    # Pass 3: rewrite default-class scope names. `class=` appears on every
    # element that selects a default; `childclass=` propagates a default to
    # nested elements. The `<default class=...>` definition is also caught
    # here — its `class` attribute is rewritten to match.
    for elem in root.iter():
        for attr in ("class", "childclass"):
            v = elem.attrib.get(attr)
            if v:
                elem.attrib[attr] = nm.prefix(v)


# --------------------------- Mesh path rewriting -------------------------- #


def rewrite_mesh_paths_mjcf(
    asset_elem: ET.Element,
    *,
    source_mjcf_dir: Path,
    source_meshdir: str,
    workstation_dir: Path,
) -> None:
    """Make every `<mesh file=...>` path relative to `workstation_dir`.

    The composed workstation drops `<compiler meshdir=...>`, so file paths
    must be self-resolving from the MJCF's own directory.
    """
    src_meshdir_path = (source_mjcf_dir / source_meshdir).resolve()
    for mesh in asset_elem.findall("mesh"):
        f = mesh.attrib.get("file")
        if not f:
            continue
        if f.startswith(("/", "package://", "http://", "https://")):
            continue
        abs_target = (src_meshdir_path / f).resolve()
        try:
            rel = os.path.relpath(str(abs_target), start=str(workstation_dir))
        except ValueError:
            rel = str(abs_target)
        mesh.attrib["file"] = rel.replace("\\", "/")


# --------------------------- Composition helpers -------------------------- #


def _find_body_by_name(parent: ET.Element, name: str) -> ET.Element | None:
    """DFS for a `<body name=...>` descendant."""
    if parent.tag == "body" and parent.attrib.get("name") == name:
        return parent
    for child in parent:
        result = _find_body_by_name(child, name)
        if result is not None:
            return result
    return None


def _collision_body_names(body_root: ET.Element) -> list[str]:
    """Names of bodies in `body_root`'s subtree that own at least one
    collision geom. By component-MJCF convention (§5), a collision geom is
    one whose `class=` ends in `_collision`. Names are returned in DFS
    document order so emitted `<exclude>` lists are deterministic.
    """
    out: list[str] = []
    for body in body_root.iter("body"):
        name = body.attrib.get("name", "")
        if not name:
            continue
        for geom in body.findall("geom"):
            cls = geom.attrib.get("class", "")
            if cls.endswith("_collision"):
                out.append(name)
                break
    return out


def _mount_ancestor_pairs(
    roles: list[str],
    mounts: list[Mount],
    freeze_base_role: str,
) -> list[tuple[str, str]]:
    """Return (ancestor_role, descendant_role) pairs along the mount chain.

    The mount graph is a tree rooted at `freeze_base_role`; for each role,
    we walk up to the root and emit a pair for every ancestor. Sibling
    components (e.g. arm_left vs arm_right) share a common ancestor but
    are not in each other's chain, so they are not paired.
    """
    parent_of: dict[str, str | None] = {role: None for role in roles}
    for m in mounts:
        child_role = m.child.split(":", 1)[0]
        parent_role = m.parent.split(":", 1)[0]
        parent_of[child_role] = parent_role

    pairs: list[tuple[str, str]] = []
    for role in roles:
        if role == freeze_base_role:
            continue
        cursor = parent_of.get(role)
        while cursor is not None:
            pairs.append((cursor, role))
            cursor = parent_of.get(cursor)
    return pairs


# --------------------------- Composition ---------------------------------- #


def compose_mjcf(
    *,
    workstation_name: str,
    compiled: list["CompiledComponent"],
    mounts: list[Mount],
    freeze_base_role: str | None,
    workstation_dir: Path,
) -> ET.Element:
    """Compose component MJCFs into one workstation `<mujoco>` document.

    Pipeline:
      1. Parse each component MJCF, role-prefix names + class scopes, and
         rewrite mesh `file=` paths to be relative to `workstation_dir`.
      2. Merge top-level sections: `<compiler>` (taken from one component;
         all components must declare the canonical attributes per
         docs/component_mjcf_authoring.md §2 and §3 #8 — `meshdir` is
         dropped because composed file paths are workstation-relative),
         `<option/>`, `<default>` (concatenated), `<asset>` (concatenated),
         `<actuator>`, `<equality>`, `<contact>`.
      3. Build `<worldbody>`: place `freeze_base`'s root body at the top
         level; for each mount in recipe order, attach the child's root
         body as a sub-body of the parent role's mount-frame body, with
         pos/euler from the recipe.

    The base body's `<freejoint/>` decision lives upstream (only freeze_base
    is supported today; floating-base workstations would need a top-level
    `<freejoint/>` here).
    """
    if freeze_base_role is None:
        raise SchemaError(
            "MJCF composition currently requires freeze_base; floating "
            "workstations are not yet supported"
        )

    # 1. Parse + prefix per component.
    parsed: dict[str, dict] = {}
    for c in compiled:
        if not c.variant.mjcf:
            raise SchemaError(
                f"role {c.role}: variant '{c.variant.name}' has no MJCF declared"
            )
        mjcf_path = c.meta.source_dir / c.variant.mjcf
        if not mjcf_path.is_file():
            raise SchemaError(f"role {c.role}: MJCF file not found: {mjcf_path}")
        try:
            comp_root = ET.parse(str(mjcf_path)).getroot()
        except ET.ParseError as e:
            raise SchemaError(f"role {c.role}: MJCF parse error at {mjcf_path}: {e}") from e

        prefix_mjcf(comp_root, c.name_map)

        compiler = comp_root.find("compiler")
        meshdir = compiler.attrib.get("meshdir", "") if compiler is not None else ""
        asset = comp_root.find("asset")
        if asset is not None:
            rewrite_mesh_paths_mjcf(
                asset,
                source_mjcf_dir=mjcf_path.parent,
                source_meshdir=meshdir,
                workstation_dir=workstation_dir,
            )
        parsed[c.role] = {"root": comp_root, "comp": c, "mjcf_path": mjcf_path}

    comp_by_role = {c.role: c for c in compiled}

    # 2. Build composed root.
    out = ET.Element("mujoco", {"model": workstation_name})

    # `<compiler>` — take from the first component, drop meshdir (paths are
    # now workstation-relative). Other components' compilers must agree on
    # the surviving attributes.
    iter_parsed = iter(parsed.values())
    first = next(iter_parsed)
    first_compiler = first["root"].find("compiler")
    if first_compiler is None:
        raise SchemaError(
            f"role {first['comp'].role}: component MJCF has no <compiler>"
        )
    canonical = {k: v for k, v in first_compiler.attrib.items() if k != "meshdir"}
    for entry in parsed.values():
        c = entry["root"].find("compiler")
        if c is None:
            continue
        attribs = {k: v for k, v in c.attrib.items() if k != "meshdir"}
        if attribs != canonical:
            raise SchemaError(
                f"role {entry['comp'].role}: <compiler> attributes "
                f"{attribs} differ from canonical {canonical}"
            )
    ET.SubElement(out, "compiler", canonical)

    ET.SubElement(out, "option")

    # `<default>` — concat each component's nested classes (already
    # role-prefixed). The composed top-level `<default>` stays unnamed; its
    # immediate children are all named per §3 #3.
    out_default = ET.SubElement(out, "default")
    for entry in parsed.values():
        d = entry["root"].find("default")
        if d is None:
            continue
        for child in list(d):
            out_default.append(deepcopy(child))

    # `<asset>` — concat all meshes (paths already rewritten in pass 1).
    out_asset = ET.SubElement(out, "asset")
    for entry in parsed.values():
        a = entry["root"].find("asset")
        if a is None:
            continue
        for child in list(a):
            out_asset.append(deepcopy(child))

    # 3. Build `<worldbody>` mount tree.
    out_wb = ET.SubElement(out, "worldbody")

    # Each component MJCF must contain exactly one top-level `<body>` per
    # docs §2 — the component's root link.
    component_roots: dict[str, ET.Element] = {}
    role_collision_bodies: dict[str, list[str]] = {}
    for role, entry in parsed.items():
        wb = entry["root"].find("worldbody")
        if wb is None:
            raise SchemaError(f"role {role}: component MJCF has no <worldbody>")
        bodies = wb.findall("body")
        if len(bodies) != 1:
            raise SchemaError(
                f"role {role}: component <worldbody> must contain exactly one "
                f"top-level <body>, found {len(bodies)}"
            )
        root_body = deepcopy(bodies[0])
        component_roots[role] = root_body
        # Collect collision-bearing bodies *before* mount reparenting — once
        # we attach role X under role Y, walking Y's subtree would also pick
        # up X's bodies and mis-attribute them.
        role_collision_bodies[role] = _collision_body_names(root_body)

    # Place the freeze_base component as the top-level body. Its authored
    # `pos`/`euler` (if any) is left as-is — the workstation has no parent.
    out_wb.append(component_roots[freeze_base_role])

    # Apply mounts in recipe order. Each mount's parent body must already
    # be in the composed tree (either freeze_base's tree or attached by an
    # earlier mount).
    for i, m in enumerate(mounts):
        child_role, child_frame = m.child.split(":", 1)
        parent_role, parent_frame = m.parent.split(":", 1)

        if child_role == freeze_base_role:
            raise SchemaError(
                f"mounts[{i}]: cannot mount freeze_base role "
                f"'{freeze_base_role}' as a child"
            )
        if child_role not in comp_by_role:
            raise SchemaError(
                f"mounts[{i}].child='{m.child}': role '{child_role}' "
                f"not in components"
            )
        if parent_role not in comp_by_role:
            raise SchemaError(
                f"mounts[{i}].parent='{m.parent}': role '{parent_role}' "
                f"not in components"
            )

        parent_comp = comp_by_role[parent_role]
        if parent_frame not in parent_comp.mount_frames:
            raise SchemaError(
                f"mounts[{i}].parent='{m.parent}': frame '{parent_frame}' "
                f"not declared in {parent_role} meta.mount_frames"
            )
        parent_body_name = parent_comp.mount_frames[parent_frame]
        parent_body_elem = _find_body_by_name(out_wb, parent_body_name)
        if parent_body_elem is None:
            raise SchemaError(
                f"mounts[{i}]: parent body '{parent_body_name}' not in "
                f"composed worldbody — recipe must order mounts so parents "
                f"are attached first"
            )

        child_body = component_roots[child_role]
        child_body.attrib["pos"] = fmt_vec(m.xyz)
        # Drop any authored quat — the mount overrides root pose. We use
        # `euler` to match the component MJCF convention (`eulerseq="XYZ"`).
        child_body.attrib.pop("quat", None)
        if any(abs(v) > 0 for v in m.rpy):
            child_body.attrib["euler"] = fmt_vec(m.rpy)
        else:
            child_body.attrib.pop("euler", None)
        parent_body_elem.append(child_body)

    # `<actuator>`, `<equality>`, `<contact>` — concat in role iteration order.
    # `<contact>` also receives auto-generated mount-seam excludes (see below).
    for tag in ("actuator", "equality"):
        merged: list[ET.Element] = []
        for entry in parsed.values():
            sec = entry["root"].find(tag)
            if sec is None:
                continue
            merged.extend(deepcopy(child) for child in list(sec))
        if merged:
            out_sec = ET.SubElement(out, tag)
            for child in merged:
                out_sec.append(child)

    # `<contact>` — concat per-component excludes, then append mount-seam
    # excludes between every (ancestor_component, descendant_component) pair
    # along the mount chain. Without these, the parent's collision mesh
    # interpenetrates the child's links at qpos=0; contact friction then
    # clamps the joint and the actuator can't drive it (same failure mode
    # as adjacent intra-component pairs in §8 of the authoring guide).
    contact_children: list[ET.Element] = []
    for entry in parsed.values():
        sec = entry["root"].find("contact")
        if sec is None:
            continue
        contact_children.extend(deepcopy(child) for child in list(sec))

    role_order = [c.role for c in compiled]
    seam_pairs = _mount_ancestor_pairs(role_order, mounts, freeze_base_role)
    for ancestor_role, descendant_role in seam_pairs:
        for a in role_collision_bodies[ancestor_role]:
            for d in role_collision_bodies[descendant_role]:
                contact_children.append(
                    ET.Element("exclude", {"body1": a, "body2": d})
                )

    if contact_children:
        out_contact = ET.SubElement(out, "contact")
        for child in contact_children:
            out_contact.append(child)

    return out
