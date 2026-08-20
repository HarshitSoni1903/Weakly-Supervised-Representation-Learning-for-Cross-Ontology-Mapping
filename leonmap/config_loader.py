"""
Load user config from YAML and patch the module-level defaults in config.py.

Usage:
    from leonmap.config_loader import load_user_config
    load_user_config("my_config.yaml")
    cfg = BuildConfig()  # now has overrides applied

YAML structure (include only what you want to override):
    work_dir: "/path/to/root"   # root for db/, data/, models/, logs/, mapper_results/
    build:
      db_dir: "my_db"
      threshold: 0.85
    collections:
      my_onto:
        source: owl
        model: ft
        owl_path: my_onto.owl
        id_prefixes: ["MY_"]
    ablations: { ... }
    mappings: { ... }
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

import leonmap.config as _cfg

_BUILD_OVERRIDES: Dict[str, Any] = {}
_SUFFIX_KEYS = {".owl": "owl_path", ".tsv": "csv_path", ".csv": "csv_path"}


class ConfigError(Exception):
    pass


def collection_name(path: str | Path) -> str:
    """Collection name inferred from an ontology filename."""
    return Path(path).stem.split("_")[0].lower()


def collection_spec(
    path: str | Path,
    model: str = "ft",
    id_prefixes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Collection spec for an ontology file. Source type comes from the suffix."""
    p = Path(path)
    key = _SUFFIX_KEYS.get(p.suffix.lower())
    if key is None:
        raise ConfigError(f"Cannot read {p.suffix!r}: expected one of {sorted(_SUFFIX_KEYS)}")
    spec: Dict[str, Any] = {"source": "owl" if key == "owl_path" else "csv", "model": model, key: str(p)}
    if id_prefixes:
        spec["id_prefixes"] = list(id_prefixes)
    return spec


def _validate(raw: Dict[str, Any]) -> None:
    """
    Validate only what would cause silent failures or wasted compute.
    Type errors and bad values will surface naturally at runtime.
    """
    errors: List[str] = []

    # check top-level keys
    valid_top = {"work_dir", "build", "collections", "ablations", "mappings"}
    unknown = set(raw.keys()) - valid_top
    if unknown:
        errors.append(f"Unknown top-level keys: {sorted(unknown)}")

    if "work_dir" in raw and not isinstance(raw["work_dir"], (str, Path)):
        errors.append("work_dir: must be a path string")

    # check build fields exist on BuildConfig
    for key in raw.get("build", {}):
        if key not in {f.name for f in dataclasses.fields(_cfg.BuildConfig)}:
            errors.append(f"build.{key}: not a BuildConfig field")

    # merged collection view for cross-referencing
    all_collections = dict(_cfg.COLLECTIONS)
    for name, spec in raw.get("collections", {}).items():
        if not isinstance(spec, dict):
            errors.append(f"collections.{name}: must be a dict")
            continue
        # source type needs the right path field
        source = spec.get("source")
        if source == "owl" and "owl_path" not in spec:
            errors.append(f"collections.{name}: source=owl requires 'owl_path'")
        if source == "csv" and "csv_path" not in spec:
            errors.append(f"collections.{name}: source=csv requires 'csv_path'")
        all_collections[name] = spec

    # ablations/mappings reference valid collections
    for section in ("ablations", "mappings"):
        for name, spec in raw.get(section, {}).items():
            if not isinstance(spec, dict):
                errors.append(f"{section}.{name}: must be a dict")
                continue
            for col_key in ("src_collection", "tgt_collection"):
                col = spec.get(col_key)
                if col and col not in all_collections:
                    errors.append(f"{section}.{name}.{col_key}: collection {col!r} not defined")

    # mapping model consistency (would produce garbage results silently)
    for name, spec in raw.get("mappings", {}).items():
        if not isinstance(spec, dict):
            continue
        src, tgt = spec.get("src_collection"), spec.get("tgt_collection")
        if src and tgt and src in all_collections and tgt in all_collections:
            sm = all_collections[src].get("model")
            tm = all_collections[tgt].get("model")
            if sm and tm and sm != tm:
                errors.append(
                    f"mappings.{name}: model mismatch: {src} uses '{sm}', {tgt} uses '{tm}'"
                )

    if errors:
        raise ConfigError("Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))


def _apply_build_overrides(overrides: Dict[str, Any]) -> None:
    """Accumulate build overrides across loads. Arguments passed to BuildConfig() still win."""
    _BUILD_OVERRIDES.update(overrides)
    if getattr(_cfg.BuildConfig.__init__, "_patched", False):
        return
    original = _cfg.BuildConfig.__init__

    def patched(self, _orig=original, **kwargs):
        merged = dict(_BUILD_OVERRIDES)
        merged.update(kwargs)
        _orig(self, **merged)

    patched._patched = True
    _cfg.BuildConfig.__init__ = patched


def load_user_config(path: str | Path) -> List[str]:
    """
    Load YAML config and patch config.py in place. Returns the collection names it declares.
    Validates cross-references before applying anything.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ConfigError(f"Config must be a YAML dict, got {type(raw).__name__}")

    _validate(raw)

    if "work_dir" in raw:
        _cfg.set_work_dir(raw["work_dir"])

    if raw.get("build"):
        _apply_build_overrides(raw["build"])

    # merge dicts
    if "collections" in raw:
        _cfg.COLLECTIONS.update(raw["collections"])
    if "ablations" in raw:
        _cfg.ABLATIONS.update(raw["ablations"])
    if "mappings" in raw:
        _cfg.MAPPINGS.update(raw["mappings"])

    return list(raw.get("collections", {}))
