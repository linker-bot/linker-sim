"""URDF composition operations.

Reads flat component URDFs, prefixes names, merges them under a single
`<robot>` root, inserts mount joints, and rewrites mesh paths so the
composed file resolves from its own directory.

Why prefixing (not runtime splicing or xacro): MuJoCo requires a single-
model flat XML with no name collisions, and we want URDF to match for
cross-stack parity (real-robot driver, LeRobot collector).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from .determinism import fmt_vec
from .schemas import (
    ComponentMeta,
    Mount,
    SchemaError,
    Variant,
    expand_vars,
)

# URDF tags that may appear under <robot> and that we carry through.
PASSTHROUGH_TAGS = ("link", "joint", "material", "transmission")

# Tags to drop during composition. Gazebo is a different simulator entirely;
# <mujoco> is handled specially (collected and merged into one block).
DROP_TAGS = ("gazebo",)


# --------------------------- Name remapping ------------------------------- #


@dataclass
class NameMap:
    """Tracks how a role's raw names map to prefixed composed names.

    Pre-expansion (template) names like `AR5_5_07{V}_W4C4A2_base` are expanded
    against the variant vars first, then prefixed with `<role>_`.
    """

    role: str
    variant_vars: dict[str, str]

    def prefix(self, raw: str) -> str:
        return f"{self.role}_{raw}"

    def expand_and_prefix(self, templated: str) -> str:
        return self.prefix(expand_vars(templated, self.variant_vars))


# --------------------------- URDF parsing --------------------------------- #


def load_urdf(path: Path) -> ET.Element:
    """Parse a URDF file into an ElementTree root. Raises on parse errors."""
    if not path.is_file():
        raise SchemaError(f"URDF not found: {path}")
    try:
        return ET.parse(str(path)).getroot()
    except ET.ParseError as e:
        raise SchemaError(f"URDF parse error at {path}: {e}") from e


def collect_joint_names(root: ET.Element) -> tuple[list[str], list[str]]:
    """Return (actuated_joint_names, mimic_joint_names) in document order.

    Actuated = non-fixed and non-mimic. Mimic joints are counted separately
    because they're not independent DOFs.
    """
    actuated: list[str] = []
    mimic: list[str] = []
    for j in root.findall("joint"):
        jtype = j.attrib.get("type", "")
        if jtype == "fixed":
            continue
        name = j.attrib.get("name")
        if not name:
            continue
        if j.find("mimic") is not None:
            mimic.append(name)
        else:
            actuated.append(name)
    return actuated, mimic


# --------------------------- Prefixing ------------------------------------ #


def prefix_urdf(root: ET.Element, nm: NameMap) -> None:
    """Rewrite link/joint/material/transmission names (and their references).

    Mutates `root` in place. After this call every <link name=>, <joint name=>,
    <material name=>, and all references to those via `link="..."`,
    `joint="..."`, parent/child children etc. carry the role prefix.
    """
    # Pass 1: rename any element whose tag matches a renamable kind and
    # that carries a `name` attribute. `iter()` recurses, so this also
    # catches <joint name=> references inside <transmission> and
    # <material name=> references inside <visual>/<collision>.
    renamable_tags = {"link", "joint", "material", "transmission"}
    for elem in root.iter():
        if elem.tag in renamable_tags:
            n = elem.attrib.get("name")
            if n:
                elem.attrib["name"] = nm.prefix(n)

    # Pass 2: rewrite non-`name` references. These elements have other
    # attributes (link, joint, joint1, joint2) that point at renamed
    # definitions from pass 1.
    for elem in root.iter():
        # <parent link="..."/> <child link="..."/>
        if elem.tag in ("parent", "child"):
            link = elem.attrib.get("link")
            if link:
                elem.attrib["link"] = nm.prefix(link)
        # <mimic joint="..."/>
        elif elem.tag == "mimic":
            j = elem.attrib.get("joint")
            if j:
                elem.attrib["joint"] = nm.prefix(j)
        # <mujoco><equality><joint joint1="..." joint2="..."/></equality>
        # Equality joints inside the URDF-embedded <mujoco> block reference
        # the hand's revolute joints by raw source-name; rename those too
        # or the composed URDF's equality constraints dangle.
        elif elem.tag == "joint":
            for a in ("joint1", "joint2"):
                v = elem.attrib.get(a)
                if v:
                    elem.attrib[a] = nm.prefix(v)


# --------------------------- Mesh path rewriting -------------------------- #


def rewrite_mesh_paths(
    root: ET.Element,
    *,
    source_urdf_dir: Path,
    workstation_dir: Path,
) -> None:
    """Rewrite `<mesh filename="...">` to resolve from `workstation_dir`.

    Mesh filenames in a URDF are relative to the URDF file's own directory
    (`source_urdf_dir`). After composition the composed workstation.urdf
    sits at a different location, so we compute an absolute target and
    re-relativize it against `workstation_dir`. Component meshes stay where
    they live — no copies.

    Absolute paths and `package://` / `http(s)://` scheme URIs are left
    alone — the real-robot loader handles those.
    """
    for mesh in root.iter("mesh"):
        filename = mesh.attrib.get("filename")
        if not filename:
            continue
        if filename.startswith(("package://", "http://", "https://", "/")):
            continue
        abs_target = (source_urdf_dir / filename).resolve()
        try:
            new_rel = _relpath(abs_target, workstation_dir)
        except ValueError:
            new_rel = str(abs_target)
        mesh.attrib["filename"] = new_rel


def _relpath(target: Path, start: Path) -> str:
    """Relative path with forward slashes, portable across OSes."""
    import os.path

    rel = os.path.relpath(str(target), start=str(start))
    return rel.replace("\\", "/")


# --------------------------- Mount joint emission ------------------------- #


def make_fixed_joint(
    *,
    name: str,
    parent_link: str,
    child_link: str,
    xyz: tuple[float, float, float],
    rpy: tuple[float, float, float],
) -> ET.Element:
    j = ET.Element("joint", {"name": name, "type": "fixed"})
    ET.SubElement(j, "origin", {"xyz": fmt_vec(xyz), "rpy": fmt_vec(rpy)})
    ET.SubElement(j, "parent", {"link": parent_link})
    ET.SubElement(j, "child", {"link": child_link})
    return j


def make_world_link() -> ET.Element:
    """The conventional `world` link used for freezing the base to world."""
    link = ET.Element("link", {"name": "world"})
    # No geometry; Isaac + MuJoCo both accept an empty link as the fixed root.
    return link


# --------------------------- Component → URDF pass ------------------------ #


@dataclass
class CompiledComponent:
    """One component after prefixing + mesh-path rewrite. Ready to merge."""

    role: str
    meta: ComponentMeta
    variant: Variant
    name_map: NameMap
    root: ET.Element  # the component's <robot> element, mutated
    actuated_joints: list[str]  # prefixed, in document order
    mimic_joints: list[str]  # prefixed, in document order
    root_link: str  # prefixed
    mount_frames: dict[str, str]  # frame_name -> prefixed link name


def compile_component(
    *,
    role: str,
    meta: ComponentMeta,
    variant: Variant,
    workstation_dir: Path,
) -> CompiledComponent:
    urdf_path = meta.source_dir / variant.urdf
    root = load_urdf(urdf_path)
    nm = NameMap(role=role, variant_vars=dict(variant.vars))

    prefix_urdf(root, nm)
    rewrite_mesh_paths(
        root,
        source_urdf_dir=urdf_path.parent,
        workstation_dir=workstation_dir,
    )

    actuated, mimic = collect_joint_names(root)
    root_link_prefixed = nm.expand_and_prefix(meta.root_link)
    mount_frames = {
        fname: nm.expand_and_prefix(frame.parent)
        for fname, frame in meta.mount_frames.items()
    }

    return CompiledComponent(
        role=role,
        meta=meta,
        variant=variant,
        name_map=nm,
        root=root,
        actuated_joints=actuated,
        mimic_joints=mimic,
        root_link=root_link_prefixed,
        mount_frames=mount_frames,
    )


# --------------------------- Composition ---------------------------------- #


def compose_urdf(
    *,
    workstation_name: str,
    compiled: list[CompiledComponent],
    mounts: list[Mount],
    freeze_base_role: str | None,
) -> ET.Element:
    """Merge compiled components into one <robot> and add mount joints.

    Element order:
      1. optional `world` link
      2. merged materials (deduped by name)
      3. merged links (component document order, then role order)
      4. merged joints
      5. merged transmissions
      6. optional <mujoco> blocks (concatenated)
      7. mount joints (in recipe order)
      8. optional world→base fixed joint
    """
    out = ET.Element("robot", {"name": workstation_name})

    # 1. world link (if any base is to be welded to world)
    if freeze_base_role is not None:
        out.append(make_world_link())

    # 2-6. Collect by tag so ordering within `out` is predictable.
    materials: dict[str, ET.Element] = {}
    links: list[ET.Element] = []
    joints: list[ET.Element] = []
    transmissions: list[ET.Element] = []
    mujoco_blocks: list[ET.Element] = []

    for comp in compiled:
        for child in list(comp.root):
            tag = child.tag
            if tag in DROP_TAGS:
                continue
            if tag == "material":
                # References without color/texture are inline references
                # inside <visual>; those stay where they are. A top-level
                # <material> with a color child is a definition — dedup by
                # name (names are role-prefixed so no cross-component
                # collisions are expected, but within a role a duplicate
                # would be a source bug).
                name = child.attrib.get("name", "")
                has_def_child = (
                    child.find("color") is not None or child.find("texture") is not None
                )
                if has_def_child:
                    if name in materials:
                        continue
                    materials[name] = child
                # Material references without def children are always
                # nested inside a visual; they don't appear at this level.
                continue
            if tag == "link":
                links.append(child)
            elif tag == "joint":
                joints.append(child)
            elif tag == "transmission":
                transmissions.append(child)
            elif tag == "mujoco":
                mujoco_blocks.append(child)
            # Any other tag is silently dropped. (We enumerate expected
            # ones explicitly so unknown content surfaces as missing.)

    for mat in materials.values():
        out.append(mat)
    for link in links:
        out.append(link)
    for joint in joints:
        out.append(joint)
    for trans in transmissions:
        out.append(trans)

    # 7. Mount joints from recipe.
    comp_by_role = {c.role: c for c in compiled}
    for i, m in enumerate(mounts):
        child_role, child_frame = m.child.split(":", 1)
        parent_role, parent_frame = m.parent.split(":", 1)
        child_comp = comp_by_role[child_role]
        parent_comp = comp_by_role[parent_role]

        if child_frame not in child_comp.mount_frames:
            raise SchemaError(
                f"mounts[{i}].child='{m.child}': frame '{child_frame}' "
                f"not declared in {child_role} meta.mount_frames "
                f"({list(child_comp.mount_frames)})"
            )
        if parent_frame not in parent_comp.mount_frames:
            raise SchemaError(
                f"mounts[{i}].parent='{m.parent}': frame '{parent_frame}' "
                f"not declared in {parent_role} meta.mount_frames "
                f"({list(parent_comp.mount_frames)})"
            )

        out.append(
            make_fixed_joint(
                name=f"mount_{child_role}_to_{parent_role}_{child_frame}",
                parent_link=parent_comp.mount_frames[parent_frame],
                child_link=child_comp.mount_frames[child_frame],
                xyz=m.xyz,
                rpy=m.rpy,
            )
        )

    # 8. Freeze base to world if requested. Attaches the component's root
    # link (declared in meta.root_link) to the `world` link we added in
    # step 1. Recipe mounts must also ensure every non-base component has
    # an inbound mount, otherwise we'll have disconnected kinematic trees.
    if freeze_base_role is not None:
        base_comp = comp_by_role[freeze_base_role]
        out.append(
            make_fixed_joint(
                name=f"mount_world_to_{freeze_base_role}",
                parent_link="world",
                child_link=base_comp.root_link,
                xyz=(0.0, 0.0, 0.0),
                rpy=(0.0, 0.0, 0.0),
            )
        )

    # <mujoco> blocks carried through so URDF→MuJoCo loading preserves
    # equality constraints (mimic joints) and compiler hints. If multiple
    # components ship <mujoco> blocks we concatenate them; a real merge of
    # `<compiler>` attributes across components is deferred (flagged as an
    # error below to avoid silent bad merges).
    if len(mujoco_blocks) > 1:
        compilers = [b.find("compiler") for b in mujoco_blocks]
        if sum(1 for c in compilers if c is not None) > 1:
            raise SchemaError(
                "multiple components declare <mujoco><compiler>; merging is "
                "not implemented — drop all but one or hand-author MJCF"
            )
    for b in mujoco_blocks:
        out.append(b)

    return out
