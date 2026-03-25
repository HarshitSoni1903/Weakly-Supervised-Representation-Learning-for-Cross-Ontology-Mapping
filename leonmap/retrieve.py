"""
Retrieve matching concepts from a target collection.
Supports single queries and batch queries with optional CSV output.

Usage:
    # single label lookup
    python retrieve.py --label "Abnormal heart morphology" --tgt mesh --top_k 5

    # batch from file
    python retrieve.py --input queries.tsv --tgt mesh --top_k 50 --out results.tsv

    # by id (fills label/def/syns from source collection automatically)
    python retrieve.py --id "HP:0001627" --tgt mp --top_k 10
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from leonmap.config import BuildConfig, COLLECTIONS
from leonmap.utils import (
    get_logger,
    resolve_device,
    load_encoder,
    free_encoder,
    load_collection,
    resolve_payload,
    fetch_top_k,
    write_results_csv,
    canonicalize_id,
    normalize_prefix,
    model_name_for,
)


def _infer_src_collection(cid: str, model_key: str = "ft") -> Optional[str]:
    """Try to figure out which collection an id belongs to based on prefix and model."""
    cid = canonicalize_id(cid)
    for cname, spec in COLLECTIONS.items():
        if spec.get("model") != model_key:
            continue
        raw_prefixes = spec.get("id_prefixes") or []
        if isinstance(raw_prefixes, str):
            raw_prefixes = [raw_prefixes]
        for p in raw_prefixes:
            if cid.startswith(normalize_prefix(p)):
                return cname
    return None


def retrieve_batch(
    queries: List[Dict],
    tgt_collection: str,
    cfg: BuildConfig,
    top_k: int = 1,
    query_mode: str = "full_src",
    src_collection: Optional[str] = None,
    out_csv: Optional[str] = None,
    logger=None,
) -> List[Dict]:
    """
    Run retrieval for a list of query dicts.
    Returns list of {src_id, src_label, matches: [{id, label, definition, synonyms, score, rank}]}
    """
    device = resolve_device(cfg.device)
    model_key = COLLECTIONS[tgt_collection]["model"]
    tok, mdl = load_encoder(model_name_for(cfg, model_key), device)

    tgt_db = load_collection(cfg, tgt_collection)

    # optionally load source db for payload enrichment
    src_db = None
    src_db_cache: Dict[str, object] = {}
    if src_collection:
        src_db = load_collection(cfg, src_collection)

    all_results: List[Dict] = []

    for q in tqdm(queries, desc=f"Retrieving -> {tgt_collection}", unit="query"):
        # if src_collection not given explicitly, try to infer from id
        effective_src_db = src_db
        if effective_src_db is None and q.get("id"):
            inferred = _infer_src_collection(q["id"], model_key)
            if inferred:
                if inferred not in src_db_cache:
                    try:
                        src_db_cache[inferred] = load_collection(cfg, inferred)
                    except SystemExit:
                        src_db_cache[inferred] = None
                effective_src_db = src_db_cache[inferred]

        payload = resolve_payload(q, effective_src_db)

        if not payload["label"]:
            if logger:
                logger.warning(f"Skipping query with no label: {q}")
            continue

        matches = fetch_top_k(
            cfg=cfg,
            src_payload=payload,
            tgt_db=tgt_db,
            model=mdl,
            tokenizer=tok,
            top_k=top_k,
            query_mode=query_mode,
            device=device,
        )

        for rank, m in enumerate(matches, 1):
            m["rank"] = rank

        all_results.append({
            "src_id": payload.get("id", ""),
            "src_label": payload.get("label", ""),
            "matches": matches,
        })

    free_encoder(tok, mdl, device)

    if out_csv:
        write_results_csv(all_results, out_csv)
        if logger:
            logger.info(f"Results written to {out_csv}")

    return all_results


def _load_queries_from_file(path: str) -> List[Dict]:
    """Load queries from a TSV/CSV file. Expects columns: id, label, definition, synonyms."""
    queries = []
    delim = "," if path.endswith(".csv") else "\t"
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delim)
        for row in reader:
            queries.append({
                "id": row.get("id", "").strip(),
                "label": row.get("label", "").strip(),
                "definition": row.get("definition", "").strip(),
                "synonyms": [s.strip() for s in row.get("synonyms", "").split(";") if s.strip()],
            })
    return queries


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrieve ontology mappings from a target collection.")
    ap.add_argument("--tgt", required=True, help="Target collection name")
    ap.add_argument("--top_k", type=int, default=1)
    ap.add_argument("--mode", default="full_src", choices=["label_only", "full_src"])
    ap.add_argument("--src", default=None, help="Source collection (optional, for enrichment)")
    ap.add_argument("--out", default=None, help="Output CSV/TSV path (optional)")

    # single query
    ap.add_argument("--id", default=None, help="Single concept id")
    ap.add_argument("--label", default=None, help="Single concept label")
    ap.add_argument("--config", default=None, help="Path to YAML config override")

    # batch query
    ap.add_argument("--input", default=None, help="Input file with queries (TSV/CSV)")

    args = ap.parse_args()
    if args.config:
        from leonmap.config_loader import load_user_config
        load_user_config(args.config)
        
    cfg = BuildConfig()
    logger = get_logger("retrieve", cfg.log_dir)

    if args.tgt not in COLLECTIONS:
        raise SystemExit(f"Unknown target collection: {args.tgt}. Available: {sorted(COLLECTIONS.keys())}")

    if args.input:
        queries = _load_queries_from_file(args.input)
    elif args.id or args.label:
        queries = [{"id": args.id or "", "label": args.label or ""}]
    else:
        raise SystemExit("Provide --input file or --id/--label for a single query.")

    logger.info(f"Queries: {len(queries)}, target: {args.tgt}, top_k: {args.top_k}, mode: {args.mode}")

    results = retrieve_batch(
        queries, args.tgt, cfg,
        top_k=args.top_k, query_mode=args.mode,
        src_collection=args.src, out_csv=args.out, logger=logger,
    )

    # print to console if no output file
    if not args.out:
        for r in results:
            print(f"\n[SRC] {r['src_id']}  {r['src_label']}")
            for m in r["matches"]:
                remark = f"  ({m['remarks']})" if m.get('remarks') else ""
                print(f"  #{m['rank']}  {m['id']}  {m['label']}  score={m['score']:.6f}{remark}")

    logger.info(f"Done. {len(results)} queries processed.")


if __name__ == "__main__":
    main()
