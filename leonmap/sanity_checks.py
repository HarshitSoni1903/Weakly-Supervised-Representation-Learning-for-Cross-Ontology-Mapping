"""
Run sanity checks on all built collections (or specific ones).

Usage:
    python sanity_checks.py                    # check all in db/
    python sanity_checks.py --collections hp mp
"""
from __future__ import annotations

import argparse
from pathlib import Path

from leonmap.config import BuildConfig, resolve_path
from leonmap.utils import get_logger, run_all_sanity_checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collections", nargs="*", default=None)
    ap.add_argument("--config", default=None, help="Path to YAML config override")
    args = ap.parse_args()
    if args.config:
        from leonmap.config_loader import load_user_config
        load_user_config(args.config)

    cfg = BuildConfig()
    logger = get_logger("sanity", cfg.log_dir)

    db = resolve_path(cfg.db_dir)
    if not db.exists():
        raise SystemExit(f"db_dir not found: {db}")

    if args.collections:
        cols = args.collections
    else:
        cols = sorted([p.name for p in db.iterdir() if p.is_dir()])

    for c in cols:
        logger.info(f"Checking {c}...")
        run_all_sanity_checks(cfg, c, logger)

    logger.info("All checks passed.")


if __name__ == "__main__":
    main()
