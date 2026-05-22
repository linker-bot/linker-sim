"""`compose_workstation` CLI and orchestration.

Usage:
    python -m tools.composer.compose <workstation_dir> [--check-drift]
    python -m tools.composer.compose assets/workstations/ar5_l6_bench_bimanual

Reads `recipe.yaml` from the workstation directory, resolves every referenced
component under `assets/components/`, emits `workstation.urdf` and (when
component MJCFs are present) `workstation.mjcf`, and writes a generated
`manifest.yaml` describing the result.

With `--check-drift`, the composer runs the full pipeline in memory and
exits non-zero if the committed artifacts do not match freshly composed
output. Used by CI to catch recipe/component edits that forgot to
re-commit the generated files.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import COMPOSER_VERSION
from .determinism import serialize
from .mjcf_ops import check_mjcf_availability, compose_mjcf
from .schemas import (
    Artifacts,
    ComponentMeta,
    ComponentProvenance,
    DefaultGains,
    Manifest,
    Recipe,
    SchemaError,
    Variant,
    resolve_variant,
)
from .urdf_ops import CompiledComponent, compile_component, compose_urdf


# --------------------------- Path resolution ------------------------------ #


@dataclass
class Paths:
    workstation_dir: Path
    assets_root: Path
    components_root: Path
    recipe: Path
    out_urdf: Path
    out_mjcf: Path
    out_manifest: Path


def resolve_paths(workstation_dir: Path, assets_root: Path | None) -> Paths:
    workstation_dir = workstation_dir.resolve()
    if not workstation_dir.is_dir():
        raise SchemaError(f"workstation directory not found: {workstation_dir}")
    recipe = workstation_dir / "recipe.yaml"

    # By convention: assets/workstations/<ws>/recipe.yaml. Walk up two levels
    # unless --assets-root overrides.
    if assets_root is None:
        assets_root = workstation_dir.parent.parent
    assets_root = assets_root.resolve()
    components_root = assets_root / "components"
    if not components_root.is_dir():
        raise SchemaError(f"components directory not found: {components_root}")

    return Paths(
        workstation_dir=workstation_dir,
        assets_root=assets_root,
        components_root=components_root,
        recipe=recipe,
        out_urdf=workstation_dir / "workstation.urdf",
        out_mjcf=workstation_dir / "workstation.mjcf",
        out_manifest=workstation_dir / "manifest.yaml",
    )


# --------------------------- Hashing -------------------------------------- #


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        return sha256_bytes(f.read())


def sha256_component_sources(meta: ComponentMeta, variant: Variant) -> str:
    """Hash of the files that define this component variant.

    Covers meta.yaml + the variant's URDF + MJCF + XRDF (if any). Does NOT
    hash meshes because they're binary LFS-tracked and the URDF's mesh
    references already fingerprint mesh identity.
    """
    h = hashlib.sha256()
    h.update(Path(meta.source_dir / "meta.yaml").read_bytes())
    h.update(b"\n\x00")
    h.update((meta.source_dir / variant.urdf).read_bytes())
    if variant.mjcf:
        mjcf_path = meta.source_dir / variant.mjcf
        if mjcf_path.is_file():
            h.update(b"\n\x00")
            h.update(mjcf_path.read_bytes())
    if variant.xrdf:
        xrdf_path = meta.source_dir / variant.xrdf
        if xrdf_path.is_file():
            h.update(b"\n\x00")
            h.update(xrdf_path.read_bytes())
    return h.hexdigest()


# --------------------------- Orchestration -------------------------------- #


@dataclass
class ComposeResult:
    urdf_text: str
    urdf_sha256: str
    mjcf_text: str | None
    mjcf_sha256: str | None
    manifest: Manifest
    manifest_yaml: str


def compose(paths: Paths) -> ComposeResult:
    recipe = Recipe.load(paths.recipe)
    recipe_sha = sha256_file(paths.recipe)

    # Load component metas and pick variants.
    components_with_variant: list[tuple[str, ComponentMeta, Variant]] = []
    for role, cref in recipe.components.items():
        component_dir = paths.components_root / cref.component
        meta_path = component_dir / "meta.yaml"
        meta = ComponentMeta.load(meta_path)
        variant = resolve_variant(meta, cref.variant)
        components_with_variant.append((role, meta, variant))

    # Compile each component (load URDF, prefix, rewrite mesh paths).
    compiled: list[CompiledComponent] = []
    for role, meta, variant in components_with_variant:
        compiled.append(
            compile_component(
                role=role,
                meta=meta,
                variant=variant,
                workstation_dir=paths.workstation_dir,
            )
        )

    # Compose URDF.
    urdf_root = compose_urdf(
        workstation_name=recipe.name,
        compiled=compiled,
        mounts=recipe.mounts,
        freeze_base_role=recipe.freeze_base,
    )
    urdf_text = _add_header_comment(
        serialize(urdf_root, indent="  ", xml_declaration=True),
        recipe_name=recipe.name,
        recipe_sha=recipe_sha,
    )
    urdf_sha = sha256_bytes(urdf_text.encode("utf-8"))

    # MJCF: compose only when every component ships an authored MJCF.
    mjcf_avail = check_mjcf_availability(components_with_variant)
    mjcf_text: str | None = None
    mjcf_sha: str | None = None
    if mjcf_avail.all_present:
        mjcf_root = compose_mjcf(
            workstation_name=recipe.name,
            compiled=compiled,
            mounts=recipe.mounts,
            freeze_base_role=recipe.freeze_base,
            workstation_dir=paths.workstation_dir,
        )
        mjcf_text = _add_header_comment(
            serialize(mjcf_root, indent="  ", xml_declaration=True),
            recipe_name=recipe.name,
            recipe_sha=recipe_sha,
        )
        mjcf_sha = sha256_bytes(mjcf_text.encode("utf-8"))
    else:
        print(
            "[compose] skipping workstation.mjcf (component MJCFs not "
            "yet authored):",
            file=sys.stderr,
        )
        for m in mjcf_avail.missing:
            print(f"  - {m}", file=sys.stderr)

    # Build manifest.
    provenance = {
        role: ComponentProvenance(
            name=_rel_component_path(paths.components_root, meta.source_dir),
            variant=variant.name,
            sha256=sha256_component_sources(meta, variant),
        )
        for (role, meta, variant) in components_with_variant
    }

    joints = {c.role: list(c.actuated_joints) for c in compiled}
    mimic_joints = {c.role: list(c.mimic_joints) for c in compiled}

    # Resolve the end-effector frame from the first component that declares
    # ee_frame (conventionally the arm role). For multi-arm workstations we
    # ALSO surface a per-role map so tasks can address either arm's ee.
    ee_link = ""
    ee_links: dict[str, str] = {}
    for c in compiled:
        if c.meta.ee_frame:
            link = c.mount_frames[c.meta.ee_frame]
            ee_links[c.role] = link
            if not ee_link:
                ee_link = link

    base_link = ""
    if recipe.freeze_base:
        base_link = next(c for c in compiled if c.role == recipe.freeze_base).root_link

    # Merge default gains per role (component default -> recipe override).
    merged_gains: dict[str, DefaultGains] = {}
    for c in compiled:
        if c.meta.default_gains:
            merged_gains[c.role] = c.meta.default_gains
    for role, override in recipe.physics_overrides.items():
        base = merged_gains.get(role)
        if base is None:
            continue
        merged_gains[role] = DefaultGains(
            stiffness=(override.stiffness if override.stiffness is not None else base.stiffness),
            damping=(override.damping if override.damping is not None else base.damping),
        )

    # Per-role named gain profiles. Components declare them under
    # `gain_profiles:` in meta.yaml; runtime controllers select one via
    # `handle.gain_profiles[role][name]`. Includes an implicit `default`
    # profile mirroring `default_gains` so callers always have a fallback.
    gain_profiles: dict[str, dict[str, DefaultGains]] = {}
    for c in compiled:
        role_profiles: dict[str, DefaultGains] = {}
        if c.meta.default_gains:
            role_profiles["default"] = c.meta.default_gains
        for pname, pgains in c.meta.gain_profiles.items():
            role_profiles[pname] = pgains
        if role_profiles:
            gain_profiles[c.role] = role_profiles

    # Frames we surface in the manifest: every mount frame from every
    # component, namespaced role:frame so multiple components' "mount"
    # frames don't collide.
    frames: dict[str, str] = {}
    for c in compiled:
        for fname, linkname in c.mount_frames.items():
            frames[f"{c.role}:{fname}"] = linkname

    # XRDF paths: record relative path from workstation dir back to
    # component source for each role that ships an XRDF.
    xrdf_paths: dict[str, str] = {}
    for role, meta, variant in components_with_variant:
        if variant.xrdf:
            xrdf_abs = (meta.source_dir / variant.xrdf).resolve()
            if xrdf_abs.is_file():
                import os.path
                xrdf_paths[role] = os.path.relpath(
                    str(xrdf_abs), start=str(paths.workstation_dir)
                ).replace("\\", "/")

    manifest = Manifest(
        schema_version=1,
        name=recipe.name,
        composer_version=COMPOSER_VERSION,
        recipe_sha256=recipe_sha,
        components=provenance,
        artifacts=Artifacts(
            urdf="workstation.urdf",
            mjcf=("workstation.mjcf" if mjcf_text is not None else None),
            urdf_sha256=urdf_sha,
            mjcf_sha256=mjcf_sha,
        ),
        joints=joints,
        mimic_joints=mimic_joints,
        frames=frames,
        ee_link=ee_link,
        ee_links=ee_links,
        base_link=base_link,
        default_gains=merged_gains,
        gain_profiles=gain_profiles,
        xrdf_paths=xrdf_paths,
    )
    manifest_yaml = yaml.safe_dump(
        manifest.to_dict(),
        sort_keys=False,
        default_flow_style=False,
    )

    return ComposeResult(
        urdf_text=urdf_text,
        urdf_sha256=urdf_sha,
        mjcf_text=mjcf_text,
        mjcf_sha256=mjcf_sha,
        manifest=manifest,
        manifest_yaml=manifest_yaml,
    )


def _rel_component_path(components_root: Path, source_dir: Path) -> str:
    import os.path

    return os.path.relpath(str(source_dir), start=str(components_root)).replace("\\", "/")


def _add_header_comment(xml_text: str, *, recipe_name: str, recipe_sha: str) -> str:
    """Prepend a generated-by comment. Goes between <?xml ...?> and <robot>."""
    header = (
        f"<!-- Generated by tools/composer (version {COMPOSER_VERSION}). "
        f"Workstation: {recipe_name}. Recipe sha256: {recipe_sha}. "
        f"Do not edit by hand; re-run `python -m tools.composer.compose "
        f"<workstation>`. -->\n"
    )
    if xml_text.startswith("<?xml"):
        first_newline = xml_text.index("\n") + 1
        return xml_text[:first_newline] + header + xml_text[first_newline:]
    return header + xml_text


# --------------------------- IO / CLI ------------------------------------- #


def write_if_changed(path: Path, new_text: str) -> bool:
    """Write only if contents differ. Returns True if written."""
    if path.is_file():
        if path.read_text() == new_text:
            return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new_text)
    return True


def check_drift(paths: Paths, result: ComposeResult) -> list[str]:
    """Return a list of drift messages (empty if committed == fresh)."""
    errors: list[str] = []

    if not paths.out_urdf.is_file():
        errors.append(f"missing committed artifact: {paths.out_urdf}")
    elif paths.out_urdf.read_text() != result.urdf_text:
        errors.append(
            f"drift: {paths.out_urdf} differs from fresh compose "
            f"(recipe or components changed; re-run compose)"
        )

    if result.mjcf_text is not None:
        if not paths.out_mjcf.is_file():
            errors.append(f"missing committed artifact: {paths.out_mjcf}")
        elif paths.out_mjcf.read_text() != result.mjcf_text:
            errors.append(f"drift: {paths.out_mjcf} differs from fresh compose")

    if not paths.out_manifest.is_file():
        errors.append(f"missing committed artifact: {paths.out_manifest}")
    elif paths.out_manifest.read_text() != result.manifest_yaml:
        errors.append(
            f"drift: {paths.out_manifest} differs from fresh compose "
            f"(manifest is generated; re-run compose)"
        )

    return errors


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Compose a workstation URDF/MJCF from components.",
    )
    p.add_argument(
        "workstation_dir",
        type=Path,
        help="path to the workstation directory (contains recipe.yaml)",
    )
    p.add_argument(
        "--assets-root",
        type=Path,
        default=None,
        help="override assets root (default: workstation_dir/../..)",
    )
    p.add_argument(
        "--check-drift",
        action="store_true",
        help="do not write; exit non-zero if committed files differ from fresh",
    )
    args = p.parse_args(argv)

    try:
        paths = resolve_paths(args.workstation_dir, args.assets_root)
        result = compose(paths)
    except SchemaError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.check_drift:
        errors = check_drift(paths, result)
        if errors:
            for msg in errors:
                print(f"drift: {msg}", file=sys.stderr)
            return 1
        print(f"[compose] {paths.workstation_dir.name}: no drift")
        return 0

    wrote_urdf = write_if_changed(paths.out_urdf, result.urdf_text)
    wrote_manifest = write_if_changed(paths.out_manifest, result.manifest_yaml)
    wrote_mjcf = False
    if result.mjcf_text is not None:
        wrote_mjcf = write_if_changed(paths.out_mjcf, result.mjcf_text)

    print(
        f"[compose] {paths.workstation_dir.name}: "
        f"urdf={'updated' if wrote_urdf else 'unchanged'}, "
        f"mjcf={'updated' if wrote_mjcf else ('skipped' if result.mjcf_text is None else 'unchanged')}, "
        f"manifest={'updated' if wrote_manifest else 'unchanged'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
