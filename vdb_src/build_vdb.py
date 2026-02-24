from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from owlready2 import get_ontology

from config import BuildConfig, COLLECTIONS
from sanity_checks import check_count


def canonicalize_id(x: str) -> str:
    return str(x).strip().replace(":", "_")


def _clean(s: str) -> str:
    return " ".join(str(s).strip().split())


def build_embedding_text(label: str, definition: str, synonyms: List[str], synonym_cap: int) -> str:
    label = _clean(label)
    definition = _clean(definition) if definition else ""

    syns: List[str] = []
    seen = set()
    for s in synonyms or []:
        s2 = _clean(s)
        if not s2 or s2 == label or s2 in seen:
            continue
        syns.append(s2)
        seen.add(s2)
        if len(syns) >= synonym_cap:
            break

    parts = [f"Label: {label}"]
    if definition:
        parts.append(f"Definition: {definition}")
    if syns:
        parts.append("Synonyms: " + "; ".join(syns))
    return "\n".join(parts)


def resolve_device(pref: str) -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device("cpu")


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.no_grad()
def embed_batch(texts: List[str], tokenizer, model, device: torch.device, max_length: int) -> np.ndarray:
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    out = model(**enc)
    pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
    pooled = F.normalize(pooled, p=2, dim=1)
    return pooled.detach().cpu().numpy().astype("float32")


def _owl_class_id(cls) -> str:
    name = getattr(cls, "name", None)
    if name:
        return canonicalize_id(name)
    iri = str(getattr(cls, "iri", "") or "")
    frag = iri.split("#")[-1].rsplit("/", 1)[-1]
    return canonicalize_id(frag)


def _first_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    try:
        if len(v) > 0:
            return str(v[0])
    except Exception:
        pass
    return str(v)


def _list_str(v) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    try:
        return [str(x) for x in list(v) if str(x).strip()]
    except Exception:
        return [str(v)] if str(v).strip() else []


def load_owl_concepts(owl_path: str) -> List[Dict]:
    onto = get_ontology(Path(owl_path).as_posix()).load()
    concepts: List[Dict] = []

    syn_attrs = [
        "hasExactSynonym",
        "hasRelatedSynonym",
        "hasBroadSynonym",
        "hasNarrowSynonym",
        "hasSynonym",
        "synonym",
        "alternative_term",
    ]

    for cls in onto.classes():
        cid = _owl_class_id(cls)
        label = _first_str(getattr(cls, "label", "")) or cid

        definition = ""
        if hasattr(cls, "IAO_0000115"):
            definition = _first_str(getattr(cls, "IAO_0000115", ""))
        if not definition and hasattr(cls, "definition"):
            definition = _first_str(getattr(cls, "definition", ""))

        synonyms: List[str] = []
        for attr in syn_attrs:
            if hasattr(cls, attr):
                synonyms.extend(_list_str(getattr(cls, attr, None)))

        # de-dup synonyms but preserve order
        seen = set()
        syn2: List[str] = []
        for s in synonyms:
            s = str(s).strip()
            if not s or s in seen:
                continue
            syn2.append(s)
            seen.add(s)

        concepts.append(
            {
                "id": cid,
                "label": str(label),
                "definition": str(definition),
                "synonyms": syn2,
                "iri": str(getattr(cls, "iri", "") or ""),
            }
        )

    return concepts


def _write_collection(
    cfg: BuildConfig,
    collection: str,
    concepts: List[Dict],
    tok,
    mdl,
    device: torch.device,
) -> None:
    cdir = Path(cfg.db_dir) / collection
    cdir.mkdir(parents=True, exist_ok=True)

    index_path = cdir / "index.faiss"
    meta_path = cdir / "meta.jsonl"
    id2pos_path = cdir / "id2pos.json"
    label2pos_path = cdir / "label2pos.json"

    concepts.sort(key=lambda d: d["id"])

    vectors: List[np.ndarray] = []
    batch: List[str] = []

    pbar = tqdm(concepts, desc=f"[EMBED] {collection}", unit="cls")
    for c in pbar:
        batch.append(build_embedding_text(c.get("label", ""), c.get("definition", ""), c.get("synonyms", []), cfg.synonym_cap))
        if len(batch) >= cfg.embed_batch_size:
            vectors.append(embed_batch(batch, tok, mdl, device, cfg.max_length))
            batch = []

    if batch:
        vectors.append(embed_batch(batch, tok, mdl, device, cfg.max_length))

    X = np.vstack(vectors).astype("float32")
    dim = int(X.shape[1])

    index = faiss.IndexFlatIP(dim)
    index.add(X)
    faiss.write_index(index, str(index_path))

    with open(meta_path, "w", encoding="utf-8") as f:
        for c in concepts:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    id2pos = {c["id"]: i for i, c in enumerate(concepts)}
    label2pos: Dict[str, int] = {}
    for i, c in enumerate(concepts):
        lbl = str(c.get("label", "")).strip()
        if lbl and lbl not in label2pos:
            label2pos[lbl] = i

    id2pos_path.write_text(json.dumps(id2pos), encoding="utf-8")
    label2pos_path.write_text(json.dumps(label2pos), encoding="utf-8")

    print(f"[BUILD] {collection}: n={len(concepts)} dim={dim}")
    check_count(cfg, collection)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collections", nargs="*", default=None, help="Collections to build (default: all)")
    args = ap.parse_args()

    cfg = BuildConfig()
    device = resolve_device(cfg.device)

    # load OWLs once (same OWLs used for base + ft)
    hp_concepts = load_owl_concepts(cfg.hp_owl_path)
    mp_concepts = load_owl_concepts(cfg.mp_owl_path)

    # choose which collections to build
    cols = args.collections or list(COLLECTIONS.keys())
    for c in cols:
        if c not in COLLECTIONS:
            raise SystemExit(f"Unknown collection: {c}")

    # group by model_key so we load each model once
    want_base = any(COLLECTIONS[c]["model"] == "base" for c in cols)
    want_ft = any(COLLECTIONS[c]["model"] == "ft" for c in cols)

    if want_base:
        model_name = cfg.base_model_name
        print(f"\n[MODEL] base: {model_name}")
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModel.from_pretrained(model_name).to(device).eval()

        for c in cols:
            if COLLECTIONS[c]["model"] != "base":
                continue
            prefix = COLLECTIONS[c]["prefix"]
            concepts = hp_concepts if prefix == "hp" else mp_concepts
            _write_collection(cfg, c, list(concepts), tok, mdl, device)

        del tok, mdl
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if want_ft:
        model_name = cfg.ft_model_path
        print(f"\n[MODEL] ft: {model_name}")
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModel.from_pretrained(model_name).to(device).eval()

        for c in cols:
            if COLLECTIONS[c]["model"] != "ft":
                continue
            prefix = COLLECTIONS[c]["prefix"]
            concepts = hp_concepts if prefix == "hp" else mp_concepts
            _write_collection(cfg, c, list(concepts), tok, mdl, device)

        del tok, mdl
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()