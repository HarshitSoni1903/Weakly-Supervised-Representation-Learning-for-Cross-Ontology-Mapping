from __future__ import annotations
import os
import platform


# ---- IMPORTANT: set OpenMP env BEFORE importing torch/faiss/transformers ----
if platform.system() == "Darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import csv

from config import BuildConfig, COLLECTIONS
from retrieve import resolve_device, load_encoder, load_collection, fetch_top_k


def canonicalize_id(x: str) -> str:
    return str(x).strip().replace(":", "_")


def infer_collection_from_id(src_id: str) -> str:
    cid = canonicalize_id(src_id)

    # Prefer config prefix matching
    for cname, spec in COLLECTIONS.items():
        prefixes = spec.get("id_prefixes") or spec.get("id_prefix") or []
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        if any(cid.startswith(p) for p in prefixes):
            return cname

    # Fallback: old HP:000... style namespace token
    if ":" in src_id:
        return str(src_id).split(":")[0].lower()

    raise ValueError(f"Could not infer collection for id={src_id}")


def fill_payload_from_id(
    cfg: BuildConfig,
    project_root: Path,
    src_payload: Dict,
    src_db_cache: Dict[str, object],
) -> Dict:
    """
    If src_payload has an id, fill missing label/definition/synonyms from the source collection.
    Does NOT override any non-empty user-provided fields.
    """
    src_id = str(src_payload.get("id", "") or "").strip()
    if not src_id:
        return src_payload

    src_col = infer_collection_from_id(src_id)
    if src_col not in src_db_cache:
        src_db_cache[src_col] = load_collection(cfg, src_col, project_root)

    src_db = src_db_cache[src_col]
    cid = canonicalize_id(src_id)
    db_payload = src_db.get_payload_by_id(cid)

    if db_payload:
        if not str(src_payload.get("label", "") or "").strip():
            src_payload["label"] = db_payload.get("label", "") or ""
        if not str(src_payload.get("definition", "") or "").strip():
            src_payload["definition"] = db_payload.get("definition", "") or ""
        if not (src_payload.get("synonyms") or []):
            src_payload["synonyms"] = db_payload.get("synonyms", []) or []

    # if still missing label, fallback to id-as-label
    if not str(src_payload.get("label", "") or "").strip():
        src_payload["label"] = cid

    return src_payload


def batch_retrieve_to_csv(
    inputs: Iterable[Dict],
    tgt_collection: str,
    out_csv: str | Path,
    *,
    top_k: int = 50,
    query_mode: str = "full_src",
) -> Path:
    """
    inputs: iterable of payload dicts, each should have keys:
      - id (optional)
      - label (optional if id is resolvable)
      - definition (optional)
      - synonyms (optional list)
    Writes one row per (src, candidate) with score.
    """
    cfg = BuildConfig()
    project_root = Path(__file__).resolve().parents[1]

    device = resolve_device(cfg.device)
    tok, mdl = load_encoder(cfg.ft_model_path, device)

    tgt_db = load_collection(cfg, tgt_collection, project_root)

    # cache source DBs by prefix (hp/mp/whatever)
    src_db_cache: Dict[str, object] = {}

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["src_id", "src_label", "target_id", "rank", "score", "tgt_collection", "query_mode"]
        )

        for src in inputs:
            # normalize fields
            payload = {
                "id": str(src.get("id", "") or "").strip(),
                "label": str(src.get("label", "") or "").strip(),
                "definition": str(src.get("definition", "") or "").strip(),
                "synonyms": src.get("synonyms", []) or [],
            }

            # if id exists -> fill missing
            payload = fill_payload_from_id(cfg, project_root, payload, src_db_cache)

            # require at least label after fill
            if not payload["label"]:
                continue

            results: List[Tuple[str, float]] = fetch_top_k(
                cfg=cfg,
                src_payload=payload,
                tgt_db=tgt_db,
                model=mdl,
                tokenizer=tok,
                top_k=top_k,
                query_mode=query_mode,
            )

            for rank, (tid, score) in enumerate(results, start=1):
                w.writerow(
                    [
                        payload.get("id", ""),
                        payload.get("label", ""),
                        tid,
                        rank,
                        f"{score:.8f}",
                        tgt_collection,
                        query_mode,
                    ]
                )

    return out_csv