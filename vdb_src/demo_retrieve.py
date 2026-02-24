# vdb_src/demo_retrieve.py

import os
import platform

# ---- IMPORTANT: set OpenMP env BEFORE importing torch/faiss/transformers ----
if platform.system() == "Darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse
from pathlib import Path
import csv

from config import BuildConfig
from retrieve import resolve_device, load_encoder, load_collection, fetch_top_k


def canonicalize_id(x: str) -> str:
    return str(x).strip().replace(":", "_")


def infer_collection_from_id(src_id: str) -> str:
    """
    If id = HP:0001627 → collection = "hp"
    If id = MP:0001234 → collection = "mp"
    """
    return src_id.split(":")[0].lower()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--id", type=str, default="")
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--definition", type=str, default="")
    parser.add_argument("--synonyms", type=str, default="")  # comma-separated
    parser.add_argument("--tgt", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--query_mode", type=str, default="full_src")

    args = parser.parse_args()

    cfg = BuildConfig()
    project_root = Path(__file__).resolve().parents[1]

    device = resolve_device(cfg.device)
    tok, mdl = load_encoder(cfg.ft_model_path, device)

    tgt_db = load_collection(cfg, args.tgt, project_root)

    # ---- build payload ----
    src_payload = {
        "id": args.id,
        "label": args.label,
        "definition": args.definition,
        "synonyms": [s.strip() for s in args.synonyms.split(",") if s.strip()],
    }

    # ---- if ID provided → fill missing fields from source collection ----
    if args.id:
        src_collection = infer_collection_from_id(args.id)
        src_db = load_collection(cfg, src_collection, project_root)

        cid = canonicalize_id(args.id)
        db_payload = src_db.get_payload_by_id(cid)

        if db_payload:
            if not src_payload["label"]:
                src_payload["label"] = db_payload.get("label", "")
            if not src_payload["definition"]:
                src_payload["definition"] = db_payload.get("definition", "")
            if not src_payload["synonyms"]:
                src_payload["synonyms"] = db_payload.get("synonyms", [])

        if not src_payload["label"]:
            src_payload["label"] = cid

    if not src_payload["label"]:
        raise SystemExit("Provide at least --label or --id")

    # ---- retrieve ----
    results = fetch_top_k(
        cfg=cfg,
        src_payload=src_payload,
        tgt_db=tgt_db,
        model=mdl,
        tokenizer=tok,
        top_k=args.top_k,
        query_mode=args.query_mode,
    )

    # ---- print ----
    for i, (tid, score) in enumerate(results, 1):
        print(f"{i:02d}\t{tid}\t{score:.6f}")

    # ---- write CSV ----
    out_path = Path("demo_results.csv")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "src_id", "src_label", "target_id", "score"])
        for i, (tid, score) in enumerate(results, 1):
            w.writerow([i, src_payload["id"], src_payload["label"], tid, score])

    print(f"\nWrote: {out_path.resolve()}")


if __name__ == "__main__":
    main()