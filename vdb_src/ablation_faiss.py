from __future__ import annotations

# ---- IMPORTANT: set OpenMP env BEFORE importing torch ----
import os
import platform

if platform.system() == "Darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from config import BuildConfig, COLLECTIONS
from retrieve import (
    fetch_top_k,
    resolve_device,
    load_encoder,
    load_collection,
)

# -------------------------
# Canonicalization
# -------------------------

def canonicalize_id(x: str) -> str:
    return str(x).strip().replace(":", "_")


# -------------------------
# Gold parsing (auto)
# -------------------------

def _detect_gold_file(data_dir: Path) -> Optional[Path]:
    for cand in [
        data_dir / "hp_mp_gold.tsv",
        data_dir / "gold.tsv",
        data_dir / "gold_mappings.tsv",
        data_dir / "mappings.tsv",
    ]:
        if cand.exists():
            return cand

    for p in sorted(data_dir.glob("*.tsv")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                row = next(csv.reader(f, delimiter="\t"))
            if len(row) >= 2:
                return p
        except Exception:
            continue
    return None


def load_gold_pairs(path: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        rows = list(reader)

    if not rows:
        return pairs

    header_l = [c.strip().lower() for c in rows[0]]
    has_header = any(("hp" in c or "mp" in c) for c in header_l)

    def add(a: str, b: str):
        a2 = canonicalize_id(a)
        b2 = canonicalize_id(b)
        if a2 and b2:
            pairs.append((a2, b2))

    if has_header:
        hp_i = None
        mp_i = None
        for i, h in enumerate(header_l):
            if hp_i is None and "hp" in h:
                hp_i = i
            if mp_i is None and "mp" in h:
                mp_i = i

        if hp_i is None or mp_i is None:
            hp_i, mp_i = 0, 1

        for r in rows[1:]:
            if len(r) <= max(hp_i, mp_i):
                continue
            add(r[hp_i], r[mp_i])
    else:
        for r in rows:
            if len(r) >= 2:
                add(r[0], r[1])

    return pairs


# -------------------------
# Ablation helpers
# -------------------------

def model_name_for(cfg: BuildConfig, model_key: str) -> str:
    return cfg.base_model_name if model_key == "base" else cfg.ft_model_path


def collections_for(model_key: str, direction: str) -> Tuple[str, str]:
    if direction == "hp_to_mp":
        return (f"hp_{model_key}", f"mp_{model_key}")
    if direction == "mp_to_hp":
        return (f"mp_{model_key}", f"hp_{model_key}")
    raise ValueError(direction)


# -------------------------
# Evaluation
# -------------------------

def evaluate_condition(
    cfg: BuildConfig,
    gold_pairs: List[Tuple[str, str]],
    direction: str,
    model_key: str,
    query_mode: str,
    ks: List[int],
    out_dir: Path,
    project_root: Path,
    device: torch.device,
    tok,
    mdl,
) -> Dict:

    src_collection, tgt_collection = collections_for(model_key, direction)

    src_db = load_collection(cfg, src_collection, project_root)
    tgt_db = load_collection(cfg, tgt_collection, project_root)

    max_k = max(ks)

    total = 0
    hits_at = {k: 0 for k in ks}
    skipped_src_missing = 0
    skipped_tgt_missing = 0

    pbar = tqdm(gold_pairs, desc=f"{direction} | {model_key} | {query_mode}", unit="pair")

    for hp_id, mp_id in pbar:
        src_id, tgt_id = (hp_id, mp_id) if direction == "hp_to_mp" else (mp_id, hp_id)

        if tgt_id not in tgt_db.id2pos:
            skipped_tgt_missing += 1
            continue

        payload = src_db.get_payload_by_id(src_id)
        if payload is None:
            skipped_src_missing += 1
            continue

        results = fetch_top_k(
            cfg=cfg,
            src_payload=payload,
            tgt_db=tgt_db,
            model=mdl,
            tokenizer=tok,
            top_k=max_k,
            query_mode=query_mode,
        )
        filtered = [tid for tid, _ in results]

        total += 1

        for k in ks:
            if tgt_id in set(filtered[:k]):
                hits_at[k] += 1

        if total and 50 in hits_at:
            pbar.set_postfix({"eval": total, "r@50": hits_at[50] / total})

    result = {
        "direction": direction,
        "model_key": model_key,
        "query_mode": query_mode,
        "src_collection": src_collection,
        "tgt_collection": tgt_collection,
        "evaluated": total,
        "skipped_src_missing": skipped_src_missing,
        "skipped_tgt_missing": skipped_tgt_missing,
    }

    for k in ks:
        result[f"recall@{k}"] = (hits_at[k] / total) if total else 0.0

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


# -------------------------
# MAIN
# -------------------------

def main() -> None:

    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=str, default=None)
    ap.add_argument("--ks", nargs="*", type=int, default=[1, 50, 100, 200])
    ap.add_argument("--models", nargs="*", type=str, default=["base", "ft"], choices=["base", "ft"])
    ap.add_argument("--modes", nargs="*", type=str, default=["label_only", "full_src"], choices=["label_only", "full_src"])
    args = ap.parse_args()

    cfg = BuildConfig()

    project_root = Path(__file__).resolve().parents[1]

    data_dir = project_root / cfg.data_dir
    gold_path = Path(args.gold) if args.gold else _detect_gold_file(data_dir)

    if gold_path is None or not gold_path.exists():
        raise SystemExit("Could not find gold TSV.")

    gold_pairs = load_gold_pairs(gold_path)
    if not gold_pairs:
        raise SystemExit(f"Gold file empty: {gold_path}")

    out_root = project_root / "ablation_study" / "results"
    out_root.mkdir(parents=True, exist_ok=True)

    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"run_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "gold": str(gold_path),
                "ks": args.ks,
                "models": args.models,
                "modes": args.modes,
                "build_config": asdict(cfg),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    device = resolve_device(cfg.device)

    summary_rows: List[Dict] = []

    for model_key in args.models:
        model_name = model_name_for(cfg, model_key)
        print(f"\n[MODEL] {model_key}: {model_name}")

        tok, mdl = load_encoder(model_name, device)

        for direction in ["hp_to_mp", "mp_to_hp"]:
            for mode in args.modes:
                out_dir = run_dir / direction / model_key / mode
                res = evaluate_condition(
                    cfg,
                    gold_pairs,
                    direction,
                    model_key,
                    mode,
                    args.ks,
                    out_dir,
                    project_root,
                    device,
                    tok,
                    mdl,
                )
                summary_rows.append(res)

        del tok, mdl
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary_path = run_dir / "summary.csv"
    cols = list(summary_rows[0].keys())

    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\n[DONE] {run_dir}")


if __name__ == "__main__":
    main()