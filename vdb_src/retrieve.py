from typing import List
import numpy as np
import torch

from config import BuildConfig, COLLECTIONS
from ablation_faiss import (
    build_embedding_text,
    embed_one,
    resolve_device,
    FaissCollection,
)


def fetch_top_k(
    cfg: BuildConfig,
    src_payload: dict,
    tgt_db: FaissCollection,
    model,
    tokenizer,
    top_k: int,
    query_mode: str = "full_src",  # "label_only" | "full_src"
) -> List[str]:
    """
    Fetch top-k target IDs for a single source concept.

    Args:
        cfg: BuildConfig
        src_payload: dict with keys {label, definition, synonyms}
        tgt_db: loaded FaissCollection (target ontology)
        model, tokenizer: loaded encoder
        top_k: number of results to return
        query_mode: "label_only" or "full_src"

    Returns:
        List[str]: top-k target IDs
    """

    device = resolve_device(cfg.device)

    label = str(src_payload.get("label", "") or "").strip()
    definition = str(src_payload.get("definition", "") or "").strip()
    synonyms = src_payload.get("synonyms", []) or []

    # ---- Build query text ----
    if query_mode == "label_only":
        qtext = f"Label: {label}"
    elif query_mode == "full_src":
        qtext = build_embedding_text(
            label,
            definition,
            list(synonyms),
            cfg.synonym_cap,
        )
    else:
        raise ValueError(query_mode)

    # ---- Embed ----
    qvec = embed_one(qtext, tokenizer, model, device, cfg.max_length)

    # ---- Search----
    threshold = float(cfg.threshold)
    overfetch_mult = int(cfg.overfetch_mult)
    max_limit_mult = int(cfg.max_limit_mult)

    need = top_k
    limit = max(need * overfetch_mult, need)
    max_limit = need * max_limit_mult

    tgt_prefix = None
    # infer from collection dir name
    for k, v in COLLECTIONS.items():
        if tgt_db.cdir.name == k:
            tgt_prefix = v["id_prefix"]
            break

    if tgt_prefix is None:
        raise RuntimeError("Could not infer id_prefix for target collection")

    filtered: List[str] = []

    while True:
        scores, idxs = tgt_db.search(qvec, min(limit, tgt_db.count()))

        filtered = []
        for s, ix in zip(scores.tolist(), idxs.tolist()):
            if ix < 0:
                continue

            pid = tgt_db.id_at_pos(ix)
            if not pid:
                continue

            if not pid.startswith(tgt_prefix):
                continue

            if s < threshold:
                continue

            filtered.append(pid)

            if len(filtered) >= need:
                break

        if len(filtered) >= need or limit >= max_limit:
            break

        limit = min(limit * 2, max_limit)

    return filtered[:top_k]