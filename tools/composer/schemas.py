"""Schemas for component meta.yaml, recipe.yaml, and workstation manifest.yaml.

Uses dataclasses + a small YAML loader so we avoid a pydantic dep. Error
messages are explicit — a broken meta file should say which field is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


# ----------------------------- Exceptions ---------------------------------- #


class SchemaError(ValueError):
    """Raised when a YAML file fails validation."""


def _require(d: dict, key: str, where: str) -> Any:
    if key not in d:
        raise SchemaError(f"{where}: missing required field '{key}'")
    return d[key]


def _as_xyz(v: Any, where: str) -> tuple[float, float, float]:
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        raise SchemaError(f"{where}: expected 3-element list, got {v!r}")
    return (float(v[0]), float(v[1]), float(v[2]))


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise SchemaError(f"file not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise SchemaError(f"{path}: root must be a mapping, got {type(data).__name__}")
    return data


# ----------------------------- Component meta ------------------------------ #


ComponentKind = Literal["arm", "hand", "base", "sensor"]
DriveMode = Literal["position", "velocity", "effort"]


@dataclass(frozen=True)
class MountFrame:
    """A named frame a component exposes for attachment.

    `parent` may contain `{V}` placeholders resolved by variant vars.
    """

    parent: str
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @staticmethod
    def from_dict(d: dict, where: str) -> "MountFrame":
        parent = _require(d, "parent", where)
        xyz = _as_xyz(d.get("xyz", [0.0, 0.0, 0.0]), f"{where}.xyz")
        rpy = _as_xyz(d.get("rpy", [0.0, 0.0, 0.0]), f"{where}.rpy")
        return MountFrame(parent=str(parent), xyz=xyz, rpy=rpy)


@dataclass(frozen=True)
class Variant:
    """One variant of a component (e.g. left / right).

    `vars` holds template substitutions applied to link/joint names and the
    mount-frame `parent` strings (e.g. `{V: L}` expands `{V}` to `L`).
    """

    name: str
    vars: dict[str, str]
    urdf: str
    mjcf: str | None
    meshdir: str

    @staticmethod
    def from_dict(name: str, d: dict, where: str) -> "Variant":
        vars_ = d.get("vars", {})
        if not isinstance(vars_, dict):
            raise SchemaError(f"{where}.vars: must be a mapping")
        return Variant(
            name=name,
            vars={str(k): str(v) for k, v in vars_.items()},
            urdf=str(_require(d, "urdf", where)),
            mjcf=(str(d["mjcf"]) if d.get("mjcf") else None),
            meshdir=str(_require(d, "meshdir", where)),
        )


@dataclass(frozen=True)
class DefaultGains:
    stiffness: float
    damping: float

    @staticmethod
    def from_dict(d: dict, where: str) -> "DefaultGains":
        return DefaultGains(
            stiffness=float(_require(d, "stiffness", where)),
            damping=float(_require(d, "damping", where)),
        )


@dataclass(frozen=True)
class ComponentMeta:
    """Contract a component declares to the composer.

    Link/joint name strings may embed `{V}` placeholders that are expanded
    per variant. The composer reads the component's URDF to extract the
    actual kinematic tree; meta is the *declared* contract used for
    validation and for filling the workstation manifest.
    """

    schema_version: int
    kind: ComponentKind
    name: str
    variants: dict[str, Variant]
    root_link: str
    mount_frames: dict[str, MountFrame]
    ee_frame: str | None
    actuated_joints: list[str]
    mimic_joints: list[str]
    drive: DriveMode
    default_gains: DefaultGains | None
    gain_profiles: dict[str, DefaultGains]
    """Optional named PD profiles (e.g. `joint`, `osc`). Controllers
    pick one at runtime via `handle.gain_profiles[role][profile]`.
    `default_gains` is preserved as a backward-compat alias for the
    single-profile case."""
    source_dir: Path  # directory containing this meta.yaml

    @staticmethod
    def load(path: Path) -> "ComponentMeta":
        d = _load_yaml(path)
        where = str(path)
        kind = _require(d, "kind", where)
        if kind not in ("arm", "hand", "base", "sensor"):
            raise SchemaError(f"{where}.kind: must be one of arm|hand|base|sensor, got {kind!r}")

        variants_raw = _require(d, "variants", where)
        if not isinstance(variants_raw, dict) or not variants_raw:
            raise SchemaError(f"{where}.variants: must be a non-empty mapping")
        variants: dict[str, Variant] = {}
        for vname, vdata in variants_raw.items():
            if not isinstance(vdata, dict):
                raise SchemaError(f"{where}.variants.{vname}: must be a mapping")
            variants[vname] = Variant.from_dict(vname, vdata, f"{where}.variants.{vname}")

        mount_frames_raw = d.get("mount_frames", {})
        if not isinstance(mount_frames_raw, dict):
            raise SchemaError(f"{where}.mount_frames: must be a mapping")
        mount_frames = {
            str(k): MountFrame.from_dict(v, f"{where}.mount_frames.{k}")
            for k, v in mount_frames_raw.items()
        }

        ee_frame = d.get("ee_frame")
        if kind == "arm" and not ee_frame:
            raise SchemaError(f"{where}: arm components must declare ee_frame")
        if ee_frame is not None and ee_frame not in mount_frames:
            raise SchemaError(
                f"{where}.ee_frame: '{ee_frame}' is not declared in mount_frames"
            )

        drive = d.get("drive", "position")
        if drive not in ("position", "velocity", "effort"):
            raise SchemaError(f"{where}.drive: must be position|velocity|effort")

        dg_raw = d.get("default_gains")
        default_gains = DefaultGains.from_dict(dg_raw, f"{where}.default_gains") if dg_raw else None

        profiles_raw = d.get("gain_profiles", {})
        if not isinstance(profiles_raw, dict):
            raise SchemaError(f"{where}.gain_profiles: must be a mapping")
        gain_profiles: dict[str, DefaultGains] = {}
        for pname, pdata in profiles_raw.items():
            if not isinstance(pdata, dict):
                raise SchemaError(f"{where}.gain_profiles.{pname}: must be a mapping")
            gain_profiles[str(pname)] = DefaultGains.from_dict(
                pdata, f"{where}.gain_profiles.{pname}"
            )

        return ComponentMeta(
            schema_version=int(d.get("schema_version", 1)),
            kind=kind,
            name=str(_require(d, "name", where)),
            variants=variants,
            root_link=str(_require(d, "root_link", where)),
            mount_frames=mount_frames,
            ee_frame=ee_frame,
            actuated_joints=[str(j) for j in d.get("actuated_joints", [])],
            mimic_joints=[str(j) for j in d.get("mimic_joints", [])],
            drive=drive,
            default_gains=default_gains,
            gain_profiles=gain_profiles,
            source_dir=path.parent.resolve(),
        )


# ----------------------------- Recipe -------------------------------------- #


@dataclass(frozen=True)
class ComponentRef:
    """Recipe reference to a component + variant."""

    component: str  # relative path under assets/components, e.g. "arms/ar5"
    variant: str | None

    @staticmethod
    def from_dict(d: dict, where: str) -> "ComponentRef":
        return ComponentRef(
            component=str(_require(d, "component", where)),
            variant=(str(d["variant"]) if d.get("variant") else None),
        )


@dataclass(frozen=True)
class Mount:
    """A fixed-joint attachment between two component frames.

    `child` and `parent` use `role:frame` syntax (e.g. `arm:tool0`).
    """

    child: str
    parent: str
    xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @staticmethod
    def from_dict(d: dict, where: str) -> "Mount":
        child = str(_require(d, "child", where))
        parent = str(_require(d, "parent", where))
        for name, ref in [("child", child), ("parent", parent)]:
            if ":" not in ref:
                raise SchemaError(
                    f"{where}.{name}: expected 'role:frame' syntax, got {ref!r}"
                )
        xyz = _as_xyz(d.get("xyz", [0.0, 0.0, 0.0]), f"{where}.xyz")
        rpy = _as_xyz(d.get("rpy", [0.0, 0.0, 0.0]), f"{where}.rpy")
        return Mount(child=child, parent=parent, xyz=xyz, rpy=rpy)


@dataclass(frozen=True)
class PhysicsOverride:
    stiffness: float | None = None
    damping: float | None = None

    @staticmethod
    def from_dict(d: dict, where: str) -> "PhysicsOverride":
        return PhysicsOverride(
            stiffness=(float(d["stiffness"]) if "stiffness" in d else None),
            damping=(float(d["damping"]) if "damping" in d else None),
        )


@dataclass(frozen=True)
class Recipe:
    """Workstation composition spec."""

    schema_version: int
    name: str
    description: str | None
    components: dict[str, ComponentRef]  # role -> component ref
    mounts: list[Mount]
    physics_overrides: dict[str, PhysicsOverride]  # role -> override
    freeze_base: str | None  # role to weld to world, or None for floating root
    source_path: Path

    @staticmethod
    def load(path: Path) -> "Recipe":
        d = _load_yaml(path)
        where = str(path)

        components_raw = _require(d, "components", where)
        if not isinstance(components_raw, dict) or not components_raw:
            raise SchemaError(f"{where}.components: must be a non-empty mapping")
        components: dict[str, ComponentRef] = {}
        for role, ref in components_raw.items():
            if not isinstance(ref, dict):
                raise SchemaError(f"{where}.components.{role}: must be a mapping")
            _validate_role(role, where)
            components[str(role)] = ComponentRef.from_dict(
                ref, f"{where}.components.{role}"
            )

        mounts_raw = d.get("mounts", [])
        if not isinstance(mounts_raw, list):
            raise SchemaError(f"{where}.mounts: must be a list")
        mounts = [
            Mount.from_dict(m, f"{where}.mounts[{i}]")
            for i, m in enumerate(mounts_raw)
        ]

        for m in mounts:
            for field_name, ref in [("child", m.child), ("parent", m.parent)]:
                role = ref.split(":", 1)[0]
                if role not in components:
                    raise SchemaError(
                        f"{where}.mounts.{field_name}='{ref}': role '{role}' "
                        f"not in components {list(components)}"
                    )

        physics_overrides_raw = d.get("physics_overrides", {})
        if not isinstance(physics_overrides_raw, dict):
            raise SchemaError(f"{where}.physics_overrides: must be a mapping")
        physics_overrides = {
            str(k): PhysicsOverride.from_dict(v, f"{where}.physics_overrides.{k}")
            for k, v in physics_overrides_raw.items()
        }
        for role in physics_overrides:
            if role not in components:
                raise SchemaError(
                    f"{where}.physics_overrides: role '{role}' not in components"
                )

        freeze_base = d.get("freeze_base")
        if freeze_base is not None and freeze_base not in components:
            raise SchemaError(
                f"{where}.freeze_base: role '{freeze_base}' not in components"
            )

        return Recipe(
            schema_version=int(d.get("schema_version", 1)),
            name=str(_require(d, "name", where)),
            description=(str(d["description"]) if d.get("description") else None),
            components=components,
            mounts=mounts,
            physics_overrides=physics_overrides,
            freeze_base=(str(freeze_base) if freeze_base else None),
            source_path=path.resolve(),
        )


def _validate_role(role: str, where: str) -> None:
    if not role or not role.replace("_", "").isalnum():
        raise SchemaError(
            f"{where}.components: role '{role}' must be alphanumeric + underscores"
        )
    # Role becomes a prefix so we reject things that would produce ambiguous names.
    if role.startswith("_") or role.endswith("_"):
        raise SchemaError(
            f"{where}.components: role '{role}' must not start or end with '_'"
        )


# ----------------------------- Manifest (generated) ------------------------ #


@dataclass(frozen=True)
class ComponentProvenance:
    name: str
    variant: str | None
    sha256: str


@dataclass(frozen=True)
class Artifacts:
    urdf: str
    mjcf: str | None
    urdf_sha256: str
    mjcf_sha256: str | None


@dataclass(frozen=True)
class Manifest:
    """Generated by the composer. What `registry.load()` reads at runtime."""

    schema_version: int
    name: str
    composer_version: str
    recipe_sha256: str
    components: dict[str, ComponentProvenance]
    artifacts: Artifacts
    joints: dict[str, list[str]]  # role -> ordered joint names (prefixed)
    mimic_joints: dict[str, list[str]]
    frames: dict[str, str]  # frame_name -> prefixed link name (e.g. "ee": "arm_...")
    ee_link: str
    base_link: str
    default_gains: dict[str, DefaultGains]  # role -> merged gains
    gain_profiles: dict[str, dict[str, DefaultGains]]  # role -> profile_name -> gains

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "composer_version": self.composer_version,
            "recipe_sha256": self.recipe_sha256,
            "components": {
                role: {
                    "name": p.name,
                    "variant": p.variant,
                    "sha256": p.sha256,
                }
                for role, p in self.components.items()
            },
            "artifacts": {
                "urdf": self.artifacts.urdf,
                "mjcf": self.artifacts.mjcf,
                "urdf_sha256": self.artifacts.urdf_sha256,
                "mjcf_sha256": self.artifacts.mjcf_sha256,
            },
            "joints": {role: list(js) for role, js in self.joints.items()},
            "mimic_joints": {role: list(js) for role, js in self.mimic_joints.items()},
            "frames": dict(self.frames),
            "ee_link": self.ee_link,
            "base_link": self.base_link,
            "default_gains": {
                role: {"stiffness": g.stiffness, "damping": g.damping}
                for role, g in self.default_gains.items()
            },
            "gain_profiles": {
                role: {
                    pname: {"stiffness": g.stiffness, "damping": g.damping}
                    for pname, g in profiles.items()
                }
                for role, profiles in self.gain_profiles.items()
            },
        }


# ------------------------- Variant template expansion ---------------------- #


def expand_vars(template: str, vars: dict[str, str]) -> str:
    """Substitute {K} placeholders from `vars`. Error if unresolved."""
    out = template
    for k, v in vars.items():
        out = out.replace("{" + k + "}", v)
    if "{" in out and "}" in out[out.index("{") :]:
        raise SchemaError(f"unresolved placeholder in {template!r} after vars={vars}")
    return out


def resolve_variant(meta: ComponentMeta, variant_name: str | None) -> Variant:
    if variant_name is None:
        if len(meta.variants) == 1:
            return next(iter(meta.variants.values()))
        raise SchemaError(
            f"component {meta.name!r} has variants {list(meta.variants)}; "
            f"recipe must specify one"
        )
    if variant_name not in meta.variants:
        raise SchemaError(
            f"component {meta.name!r}: variant {variant_name!r} not in "
            f"{list(meta.variants)}"
        )
    return meta.variants[variant_name]
