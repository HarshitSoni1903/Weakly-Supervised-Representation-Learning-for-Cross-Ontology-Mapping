# faiss_vdb/retrieve.py

from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
import faiss
from transformers import AutoTokenizer, AutoModel

from config import BuildConfig, COLLECTIONS


# -------------------------
# Text helpers
# -------------------------

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
def embed_one(text: str, tok, model, device: torch.device, max_length: int) -> np.ndarray:
    enc = tok(
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


def load_encoder(model_name: str, device: torch.device):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device).eval()
    return tok, mdl


# -------------------------
# FAISS wrapper
# -------------------------

class FaissCollection:
    def __init__(self, cdir: Path):
        self.cdir = cdir
        self.index = faiss.read_index(str(cdir / "index.faiss"))
        self.meta_path = cdir / "meta.jsonl"
        self.id2pos = json.loads((cdir / "id2pos.json").read_text(encoding="utf-8"))

        self.pos2id: List[str] = []
        with open(self.meta_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    self.pos2id.append(str(json.loads(line).get("id", "")))
                except Exception:
                    self.pos2id.append("")

    def count(self) -> int:
        return int(self.index.ntotal)

    def get_payload_by_id(self, cid: str) -> Optional[Dict]:
        pos = self.id2pos.get(cid)
        if pos is None:
            return None
        with open(self.meta_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == pos:
                    return json.loads(line)
        return None

    def search(self, qvec: np.ndarray, limit: int) -> Tuple[np.ndarray, np.ndarray]:
        scores, idxs = self.index.search(qvec, limit)
        return scores[0], idxs[0]

    def id_at_pos(self, pos: int) -> str:
        if pos < 0 or pos >= len(self.pos2id):
            return ""
        return self.pos2id[pos]


def load_collection(cfg: BuildConfig, name: str, project_root: Path) -> FaissCollection:
    cdir = project_root / cfg.db_dir / name
    if not cdir.exists():
        raise SystemExit(f"Missing collection dir: {cdir}")
    return FaissCollection(cdir)


# -------------------------
# MAIN RETRIEVAL FUNCTION
# -------------------------

def fetch_top_k(
    cfg: BuildConfig,
    src_payload: dict,
    tgt_db: FaissCollection,
    model,
    tokenizer,
    top_k: int,
    query_mode: str,
) -> List[str]:

    device = resolve_device(cfg.device)

    label = str(src_payload.get("label", "") or "").strip()
    definition = str(src_payload.get("definition", "") or "").strip()
    synonyms = src_payload.get("synonyms", []) or []

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

    qvec = embed_one(qtext, tokenizer, model, device, cfg.max_length)

    threshold = float(cfg.threshold)
    overfetch_mult = int(cfg.overfetch_mult)
    max_limit_mult = int(cfg.max_limit_mult)

    need = top_k
    limit = max(need * overfetch_mult, need)
    max_limit = need * max_limit_mult

    tgt_prefix = None
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