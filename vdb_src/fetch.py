from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss  # type: ignore
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from config import BuildConfig, COLLECTIONS
from build_vdb import build_embedding_text, canonicalize_id, resolve_device, mean_pool


def _cdir(cfg: BuildConfig, collection: str) -> Path:
    return Path(cfg.db_dir) / collection


def load_collection(cfg: BuildConfig, collection: str):
    cdir = _cdir(cfg, collection)
    index = faiss.read_index(str(cdir / "index.faiss"))
    meta_path = cdir / "meta.jsonl"
    id2pos = json.loads((cdir / "id2pos.json").read_text(encoding="utf-8"))
    label2pos = json.loads((cdir / "label2pos.json").read_text(encoding="utf-8"))

    metas = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))
    return index, metas, id2pos, label2pos


@torch.no_grad()
def embed_text(text: str, tokenizer, model, device: torch.device, max_length: int) -> np.ndarray:
    enc = tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    out = model(**enc)
    pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
    pooled = F.normalize(pooled, p=2, dim=1)
    return pooled.detach().cpu().numpy().astype("float32")


def fetch_payload_by_id(metas: List[Dict], id2pos: Dict[str, int], concept_id: str) -> Optional[Dict]:
    pos = id2pos.get(concept_id)
    if pos is None:
        return None
    return metas[int(pos)]


def fetch_payload_by_label(metas: List[Dict], label2pos: Dict[str, int], label: str) -> Optional[Dict]:
    pos = label2pos.get(label)
    if pos is None:
        return None
    return metas[int(pos)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="Label or CURIE-like id (e.g., HP_0001250)")
    ap.add_argument("--tgt_collection", required=True)
    ap.add_argument("--src_collection", default=None, help="If set, query is treated as an id or label in src, used to build text like ablation2v2")
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--min_score", type=float, default=None)
    ap.add_argument("--id_prefix", default=None, help="Optional post-filter for returned candidate ids (e.g., MP_)")
    args = ap.parse_args()

    cfg = BuildConfig()

    if args.tgt_collection not in COLLECTIONS:
        raise SystemExit(f"Unknown tgt_collection: {args.tgt_collection}")

    tgt_spec = COLLECTIONS[args.tgt_collection]
    model_key = tgt_spec["model"]
    model_name = cfg.base_model_name if model_key == "base" else cfg.ft_model_path

    device = resolve_device(cfg.device)
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device).eval()

    tgt_index, tgt_metas, tgt_id2pos, tgt_label2pos = load_collection(cfg, args.tgt_collection)

    # Build query text (closest to ablation2v2):
    # - If src_collection is provided, first locate payload in src and use label/definition/synonyms to build text.
    # - Otherwise, treat query as free text label.
    q = args.query.strip()
    q_id = canonicalize_id(q)

    if args.src_collection:
        if args.src_collection not in COLLECTIONS:
            raise SystemExit(f"Unknown src_collection: {args.src_collection}")

        src_index, src_metas, src_id2pos, src_label2pos = load_collection(cfg, args.src_collection)

        src_payload = fetch_payload_by_id(src_metas, src_id2pos, q_id)
        if src_payload is None:
            src_payload = fetch_payload_by_label(src_metas, src_label2pos, q)

        if src_payload is None:
            raise SystemExit(f"Query not found in src_collection: {q}")

        text = build_embedding_text(
            src_payload.get("label", ""),
            src_payload.get("definition", ""),
            src_payload.get("synonyms", []),
            cfg.synonym_cap,
        )
    else:
        text = f"Label: {q}"

    qvec = embed_text(text, tok, mdl, device, cfg.max_length)

    top_k_target = int(args.top_k)
    overfetch = max(cfg.overfetch_mult, 1)
    limit = top_k_target * overfetch
    max_limit = top_k_target * max(cfg.max_limit_mult, 1)

    threshold = cfg.threshold if args.min_score is None else args.min_score
    id_prefix = args.id_prefix or tgt_spec.get("id_prefix", "")

    results = []
    while True:
        D, I = tgt_index.search(qvec, int(limit))
        ids = I[0].tolist()
        scores = D[0].tolist()

        results = []
        for pos, score in zip(ids, scores):
            if pos < 0:
                continue
            if float(score) < float(threshold):
                continue
            payload = tgt_metas[int(pos)]
            rid = str(payload.get("id", "")).strip()
            if id_prefix and not rid.startswith(id_prefix):
                continue
            results.append({"id": rid, "label": payload.get("label", ""), "score": float(score)})
            if len(results) >= top_k_target:
                break

        if len(results) >= top_k_target:
            break
        if limit >= max_limit:
            break
        if len(ids) < limit:
            break
        limit *= 2

    print(json.dumps({"query": q, "tgt_collection": args.tgt_collection, "top_k": top_k_target, "results": results}, indent=2))


if __name__ == "__main__":
    main()
