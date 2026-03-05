from __future__ import annotations

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
from retrieve import resolve_device, load_encoder, load_collection, fetch_top_k


def canonicalize_id(x: str) -> str:
    return str(x).strip().replace(":", "_")

def normalize_gold_id(x: str) -> str:
    s = (x or "").strip()
    if not s:
        return ""

    if s.lower().startswith("mondo:"):
        s = "MONDO:" + s.split(":", 1)[1]

    if s.lower().startswith("mesh:"):
        s = "mesh:" + s.split(":", 1)[1]

    return canonicalize_id(s)

def _detect_gold_file(data_dir: Path) -> Optional[Path]:
    """
    Heuristic fallback: pick the first TSV/CSV-like file in data_dir that looks like a gold mapping file.
    """
    candidates: List[Path] = []
    for pat in ("*.tsv", "*.csv", "*.txt"):
        candidates.extend(sorted(data_dir.glob(pat)))
    return candidates[0] if candidates else None


def _delimiter_for(path: Path) -> str:
    suf = path.suffix.lower()
    if suf == ".csv":
        return ","
    # default: tsv / txt
    return "\t"


def load_gold_pairs(
    gold_path: Path,
    *,
    src_col: Optional[str] = None,
    tgt_col: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """
    Loads gold pairs as (src_id, tgt_id).

    Supported formats:
      - Two columns (no header): first two columns are used.
      - With header:
          - If src_col/tgt_col provided: use those column names (case-insensitive).
          - Else: auto-detect columns containing src/source/left and tgt/target/right.
          - Else: fall back to first two columns.

    IDs are canonicalized (":" -> "_").
    """
    pairs: List[Tuple[str, str]] = []
    delim = _delimiter_for(gold_path)

    with open(gold_path, "r", encoding="utf-8", newline="") as f:
        r = csv.reader(f, delimiter=delim)
        rows = [row for row in r if any(str(c).strip() for c in row)]

    if not rows:
        return pairs

    header_l = [c.strip().lower() for c in rows[0]]

    def add(a: str, b: str) -> None:
        a2 = normalize_gold_id(a)
        b2 = normalize_gold_id(b)
        if a2 and b2:
            pairs.append((a2, b2))

    # Determine whether first row is a header and which indices to use
    src_i: Optional[int] = None
    tgt_i: Optional[int] = None

    if src_col and tgt_col:
        s = src_col.strip().lower()
        t = tgt_col.strip().lower()
        for i, h in enumerate(header_l):
            if src_i is None and h == s:
                src_i = i
            if tgt_i is None and h == t:
                tgt_i = i
        has_header = (src_i is not None and tgt_i is not None)
    else:
        # heuristic header detection
        src_keys = ("src", "source", "left")
        tgt_keys = ("tgt", "target", "right", "dest", "destination")

        for i, h in enumerate(header_l):
            if src_i is None and any(k in h for k in src_keys):
                src_i = i
            if tgt_i is None and any(k in h for k in tgt_keys):
                tgt_i = i

        has_header = (src_i is not None and tgt_i is not None)

    if has_header:
        assert src_i is not None and tgt_i is not None
        for row in rows[1:]:
            if len(row) <= max(src_i, tgt_i):
                continue
            add(row[src_i], row[tgt_i])
        return pairs

    # No header (or could not detect): use first two columns for all rows
    for row in rows:
        if len(row) >= 2:
            add(row[0], row[1])

    return pairs


def model_name_for(cfg: BuildConfig, model_key: str) -> str:
    return cfg.base_model_name if model_key == "base" else cfg.ft_model_path


def evaluate_condition(
    cfg: BuildConfig,
    gold_pairs: List[Tuple[str, str]],
    *,
    src_collection: str,
    tgt_collection: str,
    model_key: str,
    query_mode: str,
    ks: List[int],
    out_dir: Path,
    project_root: Path,
    device: torch.device,
    tok,
    mdl,
) -> Dict:
    """
    Evaluates recall@k for (src_collection -> tgt_collection) given gold_pairs in that direction.
    """
    src_db = load_collection(cfg, src_collection, project_root)
    tgt_db = load_collection(cfg, tgt_collection, project_root)

    max_k = max(ks)

    total = 0
    hits_at = {k: 0 for k in ks}
    skipped_src_missing = 0
    skipped_tgt_missing = 0
    missing_src: List[Tuple[str, str]] = []
    missing_tgt: List[Tuple[str, str]] = []

    label = f"{src_collection}_to_{tgt_collection} | {model_key} | {query_mode}"
    pbar = tqdm(gold_pairs, desc=label, unit="pair")

    for src_id, tgt_id in pbar:
        # tgt existence check
        if tgt_id not in tgt_db.id2pos:
            skipped_tgt_missing += 1
            # store (src, missing_tgt)
            if len(missing_tgt) < 200000:
                missing_tgt.append((src_id, tgt_id))
            continue

        payload = src_db.get_payload_by_id(src_id)
        if payload is None:
            skipped_src_missing += 1
            # store (missing_src, tgt)
            if len(missing_src) < 200000:
                missing_src.append((src_id, tgt_id))
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
        ranked_ids = [tid for tid, _ in results]

        total += 1

        # membership checks
        ranked_set_prefix: List[str] = ranked_ids  # readability
        for k in ks:
            if tgt_id in set(ranked_set_prefix[:k]):
                hits_at[k] += 1

        if total and 50 in hits_at:
            pbar.set_postfix({"eval": total, "r@50": hits_at[50] / total})

    result: Dict[str, object] = {
        "direction": f"{src_collection}_to_{tgt_collection}",
        "model_key": model_key,
        "query_mode": query_mode,
        "src_collection": src_collection,
        "tgt_collection": tgt_collection,
        "evaluated": total,
        "skipped_src_missing": skipped_src_missing,
        "skipped_tgt_missing": skipped_tgt_missing,
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    # write missing lists
    if missing_src:
        with open(out_dir / "missing_src.tsv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["missing_src_id", "paired_tgt_id"])
            w.writerows(missing_src)

    if missing_tgt:
        with open(out_dir / "missing_tgt.tsv", "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["paired_src_id", "missing_tgt_id"])
            w.writerows(missing_tgt)    


    for k in ks:
        result[f"recall@{k}"] = (hits_at[k] / total) if total else 0.0

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=str, default=None, help="Path to gold mapping file (tsv/csv).")
    ap.add_argument("--src_collection", type=str, required=True, help="Source collection name (must exist in config).")
    ap.add_argument("--tgt_collection", type=str, required=True, help="Target collection name (must exist in config).")
    ap.add_argument("--reverse", action="store_true", help="Also evaluate reverse direction (tgt -> src).")

    ap.add_argument("--src_col", type=str, default=None, help="Header column name for source IDs (optional).")
    ap.add_argument("--tgt_col", type=str, default=None, help="Header column name for target IDs (optional).")

    ap.add_argument("--ks", nargs="*", type=int, default=[1, 50, 100, 200])
    ap.add_argument("--models", nargs="*", type=str, default=["base", "ft"], choices=["base", "ft"])
    ap.add_argument("--modes", nargs="*", type=str, default=["label_only", "full_src"], choices=["label_only", "full_src"])

    args = ap.parse_args()

    cfg = BuildConfig()
    project_root = Path(__file__).resolve().parents[1]

    if args.src_collection not in COLLECTIONS:
        raise SystemExit(f"Unknown src_collection: {args.src_collection}. Available: {sorted(COLLECTIONS.keys())}")
    if args.tgt_collection not in COLLECTIONS:
        raise SystemExit(f"Unknown tgt_collection: {args.tgt_collection}. Available: {sorted(COLLECTIONS.keys())}")

    data_dir = project_root / cfg.data_dir
    gold_path = Path(args.gold) if args.gold else _detect_gold_file(data_dir)
    if gold_path is None or not gold_path.exists():
        raise SystemExit("Could not find gold file. Provide --gold explicitly.")

    gold_pairs = load_gold_pairs(gold_path, src_col=args.src_col, tgt_col=args.tgt_col)
    if not gold_pairs:
        raise SystemExit(f"Gold file empty or unreadable: {gold_path}")

    # Quick coverage check: how many gold ids exist in the built dbs?
    src_db = load_collection(cfg, args.src_collection, project_root)
    tgt_db = load_collection(cfg, args.tgt_collection, project_root)

    src_ok = sum(1 for a, _ in gold_pairs if a in src_db.id2pos)
    tgt_ok = sum(1 for _, b in gold_pairs if b in tgt_db.id2pos)

    print(f"[GOLD COVERAGE] src_in_db: {src_ok}/{len(gold_pairs)}  tgt_in_db: {tgt_ok}/{len(gold_pairs)}")

    out_root = project_root / "ablation_study" / "results"
    out_root.mkdir(parents=True, exist_ok=True)

    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"run_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "gold": str(gold_path),
                "src_collection": args.src_collection,
                "tgt_collection": args.tgt_collection,
                "reverse": bool(args.reverse),
                "src_col": args.src_col,
                "tgt_col": args.tgt_col,
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

    # Which directions to run
    dir_pairs: List[Tuple[str, str, List[Tuple[str, str]]]] = [
        (args.src_collection, args.tgt_collection, gold_pairs)
    ]

    if args.reverse:
        # reverse direction expects the gold pairs as (tgt, src)
        gold_pairs_rev = [(b, a) for (a, b) in gold_pairs]
        dir_pairs.append((args.tgt_collection, args.src_collection, gold_pairs_rev))

    summary_rows: List[Dict] = []

    for model_key in args.models:
        model_name = model_name_for(cfg, model_key)
        print(f"\n[MODEL] {model_key}: {model_name}")

        tok, mdl = load_encoder(model_name, device)

        for (src_c, tgt_c, pairs) in dir_pairs:
            for mode in args.modes:
                out_dir = run_dir / f"{src_c}_to_{tgt_c}" / model_key / mode
                res = evaluate_condition(
                    cfg,
                    pairs,
                    src_collection=src_c,
                    tgt_collection=tgt_c,
                    model_key=model_key,
                    query_mode=mode,
                    ks=args.ks,
                    out_dir=out_dir,
                    project_root=project_root,
                    device=device,
                    tok=tok,
                    mdl=mdl,
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