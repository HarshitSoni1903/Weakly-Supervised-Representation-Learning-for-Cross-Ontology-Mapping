"""
Build FAISS vector databases for ontology collections.

Usage:
    python build_vdb.py                                          # build all
    python build_vdb.py --collections hp mp mesh mondo           # specific ones
    python build_vdb.py --collections hp mp --rebuild            # overwrite
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from leonmap.config import BuildConfig, COLLECTIONS, resolve_path
from leonmap.utils import (
    get_logger,
    resolve_device,
    load_encoder,
    free_encoder,
    load_owl_concepts,
    load_csv_concepts,
    write_collection,
    run_all_sanity_checks,
    model_name_for,
)


def _resolve_owl_path(cfg: BuildConfig, owl_path: str) -> Path:
    p = Path(owl_path)
    if p.is_absolute() and p.exists():
        return p
    candidate = resolve_path(cfg.data_dir) / owl_path
    if candidate.exists():
        return candidate
    if p.exists():
        return p
    raise FileNotFoundError(f"OWL file not found: tried {p} and {candidate}")


def _load_concepts(cfg: BuildConfig, spec: dict) -> list:
    source = spec.get("source", "owl")
    id_prefixes = spec.get("id_prefixes", [])

    if source == "owl":
        owl_path = _resolve_owl_path(cfg, spec["owl_path"])
        return load_owl_concepts(str(owl_path), id_prefixes=id_prefixes or None)
    elif source == "csv":
        csv_path = resolve_path(cfg.data_dir) / spec["csv_path"]
        return load_csv_concepts(str(csv_path))
    else:
        raise ValueError(f"Unknown source type: {source}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build FAISS collections from ontology files.")
    ap.add_argument("--collections", nargs="*", default=None, help="Which collections to build (default: all)")
    ap.add_argument("--rebuild", action="store_true", help="Overwrite existing collections")
    ap.add_argument("--monitor", type=int, default=None, help="Show N samples per ontology (overrides config)")
    args = ap.parse_args()

    cfg = BuildConfig()
    if args.rebuild:
        cfg.rebuild = True
    monitor_n = args.monitor if args.monitor is not None else cfg.monitor_samples

    logger = get_logger("build_vdb", cfg.log_dir)
    device = resolve_device(cfg.device)

    cols = args.collections or list(COLLECTIONS.keys())
    for c in cols:
        if c not in COLLECTIONS:
            raise SystemExit(f"Unknown collection: {c}. Available: {sorted(COLLECTIONS.keys())}")

    logger.info(f"Collections to build: {cols}")
    logger.info(f"Device: {device}, rebuild={cfg.rebuild}, monitor_samples={monitor_n}")

    # group by model so we load each model only once
    by_model: dict[str, list[str]] = {}
    for c in cols:
        m = COLLECTIONS[c]["model"]
        by_model.setdefault(m, []).append(c)

    # figure out which collections actually need building
    to_build = []
    for c in cols:
        cdir = resolve_path(cfg.db_dir) / c
        if cdir.exists() and not cfg.rebuild:
            logger.info(f"[SKIP] {c} already exists (use --rebuild to overwrite)")
        else:
            to_build.append(c)

    if not to_build:
        logger.info("Nothing to build.")
        return

    # preview: load each unique source, show samples, then clear memory
    if monitor_n > 0:
        unique_sources: dict[str, int] = {}  # cache_key -> concept count
        print(f"\n{'='*60}")
        print(f"[MONITOR] Preview of source files, {monitor_n} samples each:")
        for c in to_build:
            spec = COLLECTIONS[c]
            cache_key = spec.get("owl_path", "") or spec.get("csv_path", "")
            if cache_key in unique_sources:
                continue
            concepts = _load_concepts(cfg, spec)
            unique_sources[cache_key] = len(concepts)
            samples = random.sample(concepts, min(monitor_n, len(concepts)))
            print(f"\n--- {cache_key} ({len(concepts)} concepts) ---")
            for s in samples:
                print(f"  id={s['id']}  label={s['label']}")
                if s.get('definition'):
                    print(f"    def={s['definition'][:120]}...")
                if s.get('synonyms'):
                    print(f"    syns={s['synonyms'][:5]}")
            del concepts  # free immediately
        print(f"\n{'='*60}")
        resp = input("Proceed with building all? [y/n]: ").strip().lower()
        if resp not in ("y", "yes", ""):
            logger.info("User declined. Exiting.")
            return

    # build: reload source files as needed, cache within each model group
    for model_key, col_names in by_model.items():
        build_cols = [c for c in col_names if c in to_build]
        if not build_cols:
            continue

        mname = model_name_for(cfg, model_key)
        logger.info(f"Loading model [{model_key}]: {mname}")
        tok, mdl = load_encoder(mname, device)

        owl_cache: dict[str, list] = {}
        for cname in build_cols:
            spec = COLLECTIONS[cname]
            cdir = resolve_path(cfg.db_dir) / cname

            if cdir.exists() and cfg.rebuild:
                logger.info(f"[REBUILD] removing {cdir}")
                shutil.rmtree(cdir)

            cache_key = spec.get("owl_path", "") or spec.get("csv_path", "")
            if cache_key not in owl_cache:
                owl_cache[cache_key] = _load_concepts(cfg, spec)
            concepts = list(owl_cache[cache_key])
            logger.info(f"[LOAD] {cname}: {len(concepts)} concepts from {cache_key}")

            write_collection(cfg, cname, concepts, tok, mdl, device, logger)
            run_all_sanity_checks(cfg, cname, logger)

        free_encoder(tok, mdl, device)
        owl_cache.clear()

    logger.info("Build complete.")


if __name__ == "__main__":
    main()
