"""
Build FAISS vector databases for ontology collections.
Usage:
    python build_vdb.py                          # build all collections
    python build_vdb.py --collections hp mp       # build specific ones
    python build_vdb.py --collections hp --rebuild # overwrite existing
    python build_vdb.py --monitor                  # preview 2 samples before building each
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from config import BuildConfig, COLLECTIONS, resolve_path
from utils import (
    get_logger,
    resolve_device,
    load_encoder,
    free_encoder,
    load_owl_concepts,
    load_csv_concepts,
    write_collection,
    run_all_sanity_checks,
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


def _load_concepts(cfg: BuildConfig, spec: dict, collection: str) -> list:
    source = spec.get("source", "owl")
    id_prefixes = spec.get("id_prefixes", [])

    if source == "owl":
        owl_path = _resolve_owl_path(cfg, spec["owl_path"])
        concepts = load_owl_concepts(str(owl_path), id_prefixes=id_prefixes or None)
    elif source == "csv":
        csv_path = resolve_path(cfg.data_dir) / spec["csv_path"]
        concepts = load_csv_concepts(str(csv_path))
    else:
        raise ValueError(f"Unknown source type: {source}")

    return concepts


def _preview_concepts(concepts: list, collection: str, n: int = 2) -> bool:
    """Print n random samples and ask y/n."""
    samples = random.sample(concepts, min(n, len(concepts)))
    print(f"\n{'='*60}")
    print(f"[MONITOR] {collection}: {len(concepts)} concepts loaded. Showing {len(samples)} samples:")
    for s in samples:
        print(f"  id={s['id']}  label={s['label']}")
        if s.get('definition'):
            print(f"  def={s['definition'][:120]}...")
        if s.get('synonyms'):
            print(f"  syns={s['synonyms'][:5]}")
        print()
    resp = input("Proceed with building? [y/n]: ").strip().lower()
    return resp in ("y", "yes", "")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build FAISS collections from ontology files.")
    ap.add_argument("--collections", nargs="*", default=None, help="Which collections to build (default: all)")
    ap.add_argument("--rebuild", action="store_true", help="Overwrite existing collections")
    ap.add_argument("--monitor", action="store_true", help="Preview samples before building each collection")
    args = ap.parse_args()

    cfg = BuildConfig()
    if args.rebuild:
        cfg.rebuild = True
    if args.monitor:
        cfg.monitor_mode = True

    logger = get_logger("build_vdb", cfg.log_dir)
    device = resolve_device(cfg.device)

    cols = args.collections or list(COLLECTIONS.keys())
    for c in cols:
        if c not in COLLECTIONS:
            raise SystemExit(f"Unknown collection: {c}. Available: {sorted(COLLECTIONS.keys())}")

    logger.info(f"Collections to build: {cols}")
    logger.info(f"Device: {device}, rebuild={cfg.rebuild}, monitor={cfg.monitor_mode}")

    # group by model so we load each model only once
    by_model: dict[str, list[str]] = {}
    for c in cols:
        m = COLLECTIONS[c]["model"]
        by_model.setdefault(m, []).append(c)

    # cache parsed OWL files so we don't re-parse the same file
    owl_cache: dict[str, list] = {}

    for model_key, col_names in by_model.items():
        from utils import model_name_for
        mname = model_name_for(cfg, model_key)
        logger.info(f"Loading model [{model_key}]: {mname}")
        tok, mdl = load_encoder(mname, device)

        for cname in col_names:
            spec = COLLECTIONS[cname]
            cdir = resolve_path(cfg.db_dir) / cname

            if cdir.exists() and not cfg.rebuild:
                logger.info(f"[SKIP] {cname} already exists (use --rebuild to overwrite)")
                continue

            if cdir.exists() and cfg.rebuild:
                logger.info(f"[REBUILD] removing {cdir}")
                shutil.rmtree(cdir)

            # load concepts (with caching by owl_path)
            cache_key = spec.get("owl_path", "") or spec.get("csv_path", "")
            if cache_key not in owl_cache:
                owl_cache[cache_key] = _load_concepts(cfg, spec, cname)
            concepts = list(owl_cache[cache_key])  # copy so filtering doesn't affect cache

            logger.info(f"[LOAD] {cname}: {len(concepts)} concepts from {cache_key}")

            if cfg.monitor_mode:
                if not _preview_concepts(concepts, cname):
                    logger.info(f"[SKIP] {cname}: user declined")
                    continue

            write_collection(cfg, cname, concepts, tok, mdl, device, logger)
            run_all_sanity_checks(cfg, cname, logger)

        free_encoder(tok, mdl, device)
        # clear owl cache between models to free memory
        owl_cache.clear()

    logger.info("Build complete.")


if __name__ == "__main__":
    main()
