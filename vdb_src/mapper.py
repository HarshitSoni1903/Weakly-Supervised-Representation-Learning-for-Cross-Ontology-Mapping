"""
Full ontology mapper. Reads study config from MAPPINGS in config.py.
Runs both directions automatically, saves separate files, evaluates against gold.

Usage:
    python mapper.py --study hp2mp
    python mapper.py --study hp2mp --threshold 0.9
    python mapper.py --study mondo2mesh --top_k 5
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from tqdm import tqdm

from config import BuildConfig, COLLECTIONS, MAPPINGS, PROJECT_ROOT, resolve_path
from utils import (
    get_logger,
    load_collection,
    load_gold_pairs,
    write_results_csv,
    normalize_prefix,
)


def _run_one_direction(
    src_name: str,
    tgt_name: str,
    cfg: BuildConfig,
    threshold: float,
    top_k: int,
    batch_size: int,
    gold_by_src: Dict[str, set],
    out_path: Path,
    logger,
) -> Dict:
    src_db = load_collection(cfg, src_name)
    tgt_db = load_collection(cfg, tgt_name)

    # prefix filter
    spec = COLLECTIONS.get(tgt_name, {})
    raw_prefixes = spec.get("id_prefixes") or []
    if isinstance(raw_prefixes, str):
        raw_prefixes = [raw_prefixes]
    norm_prefixes = [normalize_prefix(p) for p in raw_prefixes]

    def ok_prefix(pid: str) -> bool:
        if not norm_prefixes:
            return True
        return any(pid.startswith(p) for p in norm_prefixes)

    n_src = src_db.count()
    search_k = min(max(top_k * cfg.overfetch_mult, 50), tgt_db.count())  # at least 50 candidates
    logger.info(f"  {src_name}->{tgt_name}: {n_src} src, {tgt_db.count()} tgt, search_k={search_k}, threshold={threshold}")

    all_results: List[Dict] = []
    predictions_by_src: Dict[str, str] = {}
    gold_src_ids = set(gold_by_src.keys()) if gold_by_src else None

    for batch_start in tqdm(range(0, n_src, batch_size), desc=f"Mapping {src_name}->{tgt_name}", unit="batch"):
        batch_end = min(batch_start + batch_size, n_src)
        src_vecs = np.vstack([src_db.index.reconstruct(i) for i in range(batch_start, batch_end)])
        scores_batch, idxs_batch = tgt_db.index.search(src_vecs, search_k)

        for local_i in range(batch_end - batch_start):
            pos = batch_start + local_i
            src_id = src_db.id_at_pos(pos)
            if not src_id:
                continue

            src_meta = src_db.get_payload_by_id(src_id) or {}
            scores = scores_batch[local_i]
            idxs = idxs_batch[local_i]

            # exact match boost
            src_label = src_meta.get("label", "")
            src_syns = src_meta.get("synonyms", [])
            exact_ids = set(tgt_db.exact_match_ids(src_label, src_syns))

            candidates: List[Tuple[str, float]] = []
            seen_ids: set = set()
            for s, ix in zip(scores.tolist(), idxs.tolist()):
                if ix < 0:
                    continue
                pid = tgt_db.id_at_pos(ix)
                if not pid or not ok_prefix(pid) or pid in seen_ids:
                    continue
                if s < threshold:
                    continue
                candidates.append((pid, float(s)))
                seen_ids.add(pid)

            for eid in exact_ids:
                if ok_prefix(eid) and eid not in seen_ids:
                    candidates.append((eid, 0.0))

            # unambiguous boost only
            boost_ids = exact_ids if len(exact_ids) == 1 else set()
            exact_ranked = [(pid, emb, 1.0) for pid, emb in candidates if pid in boost_ids]
            other_ranked = [(pid, emb, emb) for pid, emb in candidates if pid not in boost_ids]
            exact_ranked.sort(key=lambda x: -x[1])
            other_ranked.sort(key=lambda x: -x[1])
            final_ranked = [(pid, display) for pid, _, display in exact_ranked + other_ranked]

            matches: List[Dict] = []
            for pid, score in final_ranked[:top_k]:
                meta = tgt_db.get_payload_by_id(pid) or {}
                matches.append({
                    "id": pid,
                    "label": meta.get("label", ""),
                    "definition": meta.get("definition", ""),
                    "synonyms": meta.get("synonyms", []),
                    "score": score,
                    "rank": len(matches) + 1,
                })

            all_results.append({
                "src_id": src_id,
                "src_label": src_meta.get("label", ""),
                "matches": matches,
            })

            if matches:
                predictions_by_src[src_id] = matches[0]["id"]

    # write results
    write_results_csv(all_results, out_path, gold_src_ids=gold_src_ids)
    logger.info(f"  Mappings written to {out_path}: {len(all_results)} concepts")

    # gold evaluation — for each gold src where both src and tgt exist in db:
    #   TP: predicted correct target
    #   FP: predicted wrong target
    #   unmapped: nothing above threshold — still a failure if gold target existed
    tp, fp, unmapped, gold_src_missing, gold_tgt_missing = 0, 0, 0, 0, 0
    if gold_by_src:
        for src_id, expected_tgts in gold_by_src.items():
            if src_id not in src_db.id2pos:
                gold_src_missing += 1
                continue
            valid_tgts = {t for t in expected_tgts if t in tgt_db.id2pos}
            if not valid_tgts:
                gold_tgt_missing += 1
                continue
            predicted_tgt = predictions_by_src.get(src_id)
            if predicted_tgt is None:
                unmapped += 1
            elif predicted_tgt in valid_tgts:
                tp += 1
            else:
                fp += 1

        testable = tp + fp + unmapped  # all gold pairs where both src and tgt exist
        mapped = tp + fp               # subset that produced a prediction
        accuracy = tp / testable if testable else 0.0      # of all testable, how many correct
        precision = tp / mapped if mapped else 0.0          # of predictions made, how many correct
        logger.info(f"  Eval: testable={testable} TP={tp} FP={fp} unmapped={unmapped} accuracy={accuracy:.4f} precision={precision:.4f} (gold_src_missing={gold_src_missing} gold_tgt_missing={gold_tgt_missing})")
    else:
        accuracy, precision = 0.0, 0.0
        testable, mapped = 0, 0
        tp, fp, unmapped, gold_src_missing, gold_tgt_missing = 0, 0, 0, 0, 0

    return {
        "direction": f"{src_name}->{tgt_name}",
        "src_concepts": n_src,
        "tgt_concepts": tgt_db.count(),
        "threshold": threshold,
        "top_k": top_k,
        "gold_total": len(gold_by_src),
        "gold_src_missing": gold_src_missing,
        "gold_tgt_missing": gold_tgt_missing,
        "testable": testable,
        "tp": tp,
        "fp": fp,
        "unmapped": unmapped,
        "accuracy": accuracy,
        "precision": precision,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run full ontology mapping from config presets.")
    ap.add_argument("--study", required=True, help=f"Study key from MAPPINGS. Available: {sorted(MAPPINGS.keys())}")
    ap.add_argument("--threshold", type=float, default=None, help="Override config threshold")
    ap.add_argument("--top_k", type=int, default=None, help="Override config top_k")
    ap.add_argument("--batch_size", type=int, default=512)
    args = ap.parse_args()

    if args.study not in MAPPINGS:
        raise SystemExit(f"Unknown study: {args.study}. Available: {sorted(MAPPINGS.keys())}")

    study = MAPPINGS[args.study]
    cfg = BuildConfig()
    logger = get_logger("mapper", cfg.log_dir)

    threshold = args.threshold if args.threshold is not None else study.get("threshold", cfg.threshold)
    top_k = args.top_k if args.top_k is not None else study.get("top_k", 1)
    src_col_name = study["src_collection"]
    tgt_col_name = study["tgt_collection"]
    do_reverse = study.get("reverse", True)

    # load gold pairs — one src can have multiple valid targets
    gold_fwd: Dict[str, set] = {}  # src_id -> set of valid tgt_ids
    gold_rev: Dict[str, set] = {}
    gold_file = study.get("gold_file")
    if gold_file:
        gold_path = resolve_path(cfg.data_dir) / gold_file
        if gold_path.exists():
            raw_pairs = load_gold_pairs(gold_path, src_col=study.get("src_col"), tgt_col=study.get("tgt_col"))
            logger.info(f"Gold pairs loaded: {len(raw_pairs)} ({len(set(raw_pairs))} unique)")

            src_prefix = normalize_prefix(COLLECTIONS[src_col_name]["id_prefixes"][0])
            tgt_prefix = normalize_prefix(COLLECTIONS[tgt_col_name]["id_prefixes"][0])

            for a, b in raw_pairs:
                if a.startswith(src_prefix) and b.startswith(tgt_prefix):
                    gold_fwd.setdefault(a, set()).add(b)
                    gold_rev.setdefault(b, set()).add(a)
                elif b.startswith(src_prefix) and a.startswith(tgt_prefix):
                    gold_fwd.setdefault(b, set()).add(a)
                    gold_rev.setdefault(a, set()).add(b)

            logger.info(f"Gold fwd ({src_col_name}->{tgt_col_name}): {len(gold_fwd)} src concepts, rev: {len(gold_rev)} src concepts")
        else:
            logger.warning(f"Gold file not found: {gold_path}")

    # output directory
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "mapper_results" / args.study / f"run_{run_stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # save run config
    (out_dir / "run_config.json").write_text(json.dumps({
        "study": args.study, "threshold": threshold, "top_k": top_k,
        "reverse": do_reverse, "study_params": study,
    }, indent=2), encoding="utf-8")

    logger.info(f"Study: {args.study}, threshold={threshold}, top_k={top_k}, reverse={do_reverse}")

    results = []

    # forward direction
    fwd_path = out_dir / f"{src_col_name}_to_{tgt_col_name}.tsv"
    fwd_metrics = _run_one_direction(
        src_col_name, tgt_col_name, cfg, threshold, top_k, args.batch_size,
        gold_fwd, fwd_path, logger,
    )
    results.append(fwd_metrics)

    # reverse direction
    if do_reverse:
        rev_path = out_dir / f"{tgt_col_name}_to_{src_col_name}.tsv"
        rev_metrics = _run_one_direction(
            tgt_col_name, src_col_name, cfg, threshold, top_k, args.batch_size,
            gold_rev, rev_path, logger,
        )
        results.append(rev_metrics)

    # write summary
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info(f"Summary: {summary_path}")
    logger.info("Mapper complete.")


if __name__ == "__main__":
    main()
