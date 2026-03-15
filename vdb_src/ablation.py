"""
Ablation study: evaluate retrieval recall/precision/F1 on gold standard pairs.
Reads study params from ABLATIONS dict in config, or overrides via CLI.

Usage:
    python ablation.py --study hp2mp
    python ablation.py --study hp2mp --ks 1 50 200 --models ft --modes full_src
    python ablation.py --study mondo2mesh --reverse
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from config import BuildConfig, COLLECTIONS, ABLATIONS, resolve_path, PROJECT_ROOT
from utils import (
    get_logger,
    resolve_device,
    load_encoder,
    free_encoder,
    load_collection,
    fetch_top_k,
    load_gold_pairs,
    model_name_for,
    collection_name_for_model,
    canonicalize_id,
    build_embedding_text,
    embed_texts_batched,
    normalize_prefix,
)


def _evaluate_direction(
    cfg: BuildConfig,
    gold_pairs: List[Tuple[str, str]],
    src_collection: str,
    tgt_collection: str,
    model_key: str,
    query_mode: str,
    ks: List[int],
    out_dir: Path,
    device: torch.device,
    tok,
    mdl,
    logger,
) -> Dict:
    """Evaluate recall@k (and F1 for k=1) for one direction."""
    src_db = load_collection(cfg, src_collection)
    tgt_db = load_collection(cfg, tgt_collection)

    max_k = max(ks)

    # phase 1: filter valid pairs and build query texts
    valid_pairs: List[Tuple[str, str, Dict]] = []  # (src_id, tgt_id, payload)
    skipped_src = 0
    skipped_tgt = 0

    for src_id, tgt_id in gold_pairs:
        if tgt_id not in tgt_db.id2pos:
            skipped_tgt += 1
            continue
        payload = src_db.get_payload_by_id(src_id)
        if payload is None:
            skipped_src += 1
            continue
        valid_pairs.append((src_id, tgt_id, payload))

    if not valid_pairs:
        logger.warning(f"No valid pairs for {src_collection}->{tgt_collection}")
        return {"direction": f"{src_collection}->{tgt_collection}", "evaluated": 0,
                "skipped_src_missing": skipped_src, "skipped_tgt_missing": skipped_tgt}

    # build query texts
    qtexts = []
    for _, _, payload in valid_pairs:
        label = str(payload.get("label", "") or "").strip()
        definition = str(payload.get("definition", "") or "").strip()
        synonyms = payload.get("synonyms", []) or []
        if query_mode == "label_only":
            qtexts.append(f"Label: {label}")
        else:
            qtexts.append(build_embedding_text(label, definition, list(synonyms), cfg.synonym_cap))

    # phase 2: batch embed all queries at once
    label_str = f"{src_collection}->{tgt_collection} | {model_key} | {query_mode}"
    logger.info(f"  {label_str}: embedding {len(qtexts)} queries...")
    all_qvecs = embed_texts_batched(qtexts, tok, mdl, device, cfg.max_length,
                                     cfg.embed_batch_size, desc=f"[EMBED] {label_str}")

    # prefix filter setup
    spec = COLLECTIONS.get(tgt_db.cdir.name, {})
    raw_prefixes = spec.get("id_prefixes") or []
    if isinstance(raw_prefixes, str):
        raw_prefixes = [raw_prefixes]
    norm_prefixes = [normalize_prefix(p) for p in raw_prefixes]

    def ok_prefix(pid: str) -> bool:
        if not norm_prefixes:
            return True
        return any(pid.startswith(p) for p in norm_prefixes)

    # phase 3: FAISS search + scoring (fast, no model calls)
    total = 0
    hits_at = {k: 0 for k in ks}
    per_k_rows: Dict[int, List] = {k: [] for k in ks}
    threshold = float(cfg.threshold)

    search_limit = min(max_k * cfg.overfetch_mult, tgt_db.count())

    for i, (src_id, tgt_id, payload) in enumerate(tqdm(valid_pairs, desc=label_str, unit="pair")):
        qvec = all_qvecs[i : i + 1]  # (1, dim)
        scores, idxs = tgt_db.search(qvec, search_limit)

        # get exact match ids for boosting
        src_label = str(payload.get("label", "") or "")
        src_syns = payload.get("synonyms", []) or []
        exact_ids = set(tgt_db.exact_match_ids(src_label, src_syns))

        # collect all candidates with their embedding scores
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
            if len(candidates) >= max_k * 2:
                break

        # add exact matches not already in candidates
        for eid in exact_ids:
            if ok_prefix(eid) and eid not in seen_ids:
                candidates.append((eid, 0.0))
                seen_ids.add(eid)

        # only boost if unambiguous (single exact match), otherwise cosine decides
        boost_ids = exact_ids if len(exact_ids) == 1 else set()

        exact_ranked = [(pid, emb, 1.0) for pid, emb in candidates if pid in boost_ids]
        other_ranked = [(pid, emb, emb) for pid, emb in candidates if pid not in boost_ids]
        exact_ranked.sort(key=lambda x: -x[1])
        other_ranked.sort(key=lambda x: -x[1])
        ranked = [(pid, display) for pid, _, display in exact_ranked + other_ranked]

        total += 1
        for k in ks:
            top_k = ranked[:k]
            top_k_ids = {pid for pid, _ in top_k}
            is_hit = tgt_id in top_k_ids
            if is_hit:
                hits_at[k] += 1
            per_k_rows[k].append({
                "src_id": src_id,
                "tgt_id_gold": tgt_id,
                "hit": is_hit,
                "top_matches": [{"id": pid, "score": round(sc, 6)} for pid, sc in top_k],
            })

        if total % 200 == 0 and 50 in hits_at:
            pass  # tqdm handles progress

    # compute metrics
    metrics: Dict = {
        "direction": f"{src_collection}->{tgt_collection}",
        "model": model_key,
        "query_mode": query_mode,
        "evaluated": total,
        "skipped_src_missing": skipped_src,
        "skipped_tgt_missing": skipped_tgt,
    }

    for k in ks:
        recall = (hits_at[k] / total) if total else 0.0
        metrics[f"recall@{k}"] = recall

        if k == 1:
            tp = hits_at[1]
            fp = total - tp
            accuracy = tp / total if total else 0.0
            metrics["tp@1"] = tp
            metrics["fp@1"] = fp
            metrics["accuracy@1"] = accuracy

    # write outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    for k in ks:
        k_path = out_dir / f"predictions_at_{k}.jsonl"
        with open(k_path, "w", encoding="utf-8") as f:
            for row in per_k_rows[k]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info(f"  {label}: evaluated={total}, " + ", ".join(f"r@{k}={metrics[f'recall@{k}']:.4f}" for k in ks))
    if 1 in ks:
        logger.info(f"    accuracy@1={metrics.get('accuracy@1', 0):.4f}  TP={metrics.get('tp@1', 0)} FP={metrics.get('fp@1', 0)}")

    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Run ablation study from config presets.")
    ap.add_argument("--study", required=True, help=f"Study key from ABLATIONS. Available: {sorted(ABLATIONS.keys())}")
    ap.add_argument("--ks", nargs="*", type=int, default=None, help="Override recall k values")
    ap.add_argument("--models", nargs="*", type=str, default=None, choices=["base", "ft"])
    ap.add_argument("--modes", nargs="*", type=str, default=None, choices=["label_only", "full_src"])
    ap.add_argument("--reverse", action="store_true", default=None, help="Also run reverse direction")
    args = ap.parse_args()

    if args.study not in ABLATIONS:
        raise SystemExit(f"Unknown study: {args.study}. Available: {sorted(ABLATIONS.keys())}")

    study = ABLATIONS[args.study]
    cfg = BuildConfig()
    logger = get_logger("ablation", cfg.log_dir)

    # merge CLI overrides into study params
    ks = args.ks or study["ks"]
    models = args.models or study["models"]
    modes = args.modes or study["modes"]
    do_reverse = args.reverse if args.reverse is not None else study.get("reverse", False)

    src_col = study["src_collection"]
    tgt_col = study["tgt_collection"]

    # load gold pairs
    gold_path = resolve_path(cfg.data_dir) / study["gold_file"]
    if not gold_path.exists():
        raise SystemExit(f"Gold file not found: {gold_path}")

    gold_pairs = load_gold_pairs(gold_path, study.get("src_col"), study.get("tgt_col"))
    if not gold_pairs:
        raise SystemExit(f"No valid pairs in {gold_path}")

    logger.info(f"Study: {args.study}, gold pairs: {len(gold_pairs)}, ks={ks}, models={models}, modes={modes}, reverse={do_reverse}")

    # set up output directory
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = PROJECT_ROOT / "ablation_results" / f"{src_col}_{tgt_col}" / f"run_{run_stamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    # save run config for reproducibility
    (out_root / "run_config.json").write_text(json.dumps({
        "study": args.study,
        "study_params": study,
        "ks": ks, "models": models, "modes": modes, "reverse": do_reverse,
        "gold_pairs": len(gold_pairs),
        "build_config": asdict(cfg),
    }, indent=2), encoding="utf-8")

    # directions to evaluate
    directions: List[Tuple[str, str, List[Tuple[str, str]]]] = [
        (src_col, tgt_col, gold_pairs),
    ]
    if do_reverse:
        directions.append((tgt_col, src_col, [(b, a) for a, b in gold_pairs]))

    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    device = resolve_device(cfg.device)
    summary: List[Dict] = []

    for model_key in models:
        mname = model_name_for(cfg, model_key)
        logger.info(f"Loading model [{model_key}]: {mname}")
        tok, mdl = load_encoder(mname, device)

        for s_col, t_col, pairs in directions:
            # resolve actual collection names (e.g. hp + base -> hp_base)
            s_cname = collection_name_for_model(s_col, model_key)
            t_cname = collection_name_for_model(t_col, model_key)

            for mode in modes:
                out_dir = out_root / f"{s_cname}_to_{t_cname}" / model_key / mode
                metrics = _evaluate_direction(
                    cfg, pairs, s_cname, t_cname,
                    model_key, mode, ks, out_dir,
                    device, tok, mdl, logger,
                )
                summary.append(metrics)

        free_encoder(tok, mdl, device)

    # write summary CSV
    if summary:
        summary_path = out_root / "summary.csv"
        cols = list(summary[0].keys())
        with open(summary_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(summary)
        logger.info(f"Summary written to {summary_path}")

    logger.info(f"Ablation complete: {out_root}")


if __name__ == "__main__":
    main()
