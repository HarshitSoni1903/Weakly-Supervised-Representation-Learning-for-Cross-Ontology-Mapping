from __future__ import annotations

# ---- IMPORTANT: set OpenMP env BEFORE importing faiss/torch/numpy ----
import os
import platform

if platform.system() == "Darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss  # type: ignore
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from config import BuildConfig, COLLECTIONS


# -------------------------
# Canon + text helpers
# -------------------------

def canonicalize_id(x: str) -> str:
    # IMPORTANT: must match build_vdb canonicalization
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
# FAISS collection wrapper
# -------------------------

class FaissCollection:
    def __init__(self, cdir: Path):
        self.cdir = cdir
        self.index = faiss.read_index(str(cdir / "index.faiss"))
        self.meta_path = cdir / "meta.jsonl"
        self.id2pos = json.loads((cdir / "id2pos.json").read_text(encoding="utf-8"))

        # load ids aligned with FAISS positions once (fast lookup)
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
    for fn in ["index.faiss", "meta.jsonl", "id2pos.json"]:
        if not (cdir / fn).exists():
            raise SystemExit(f"Missing {fn} in {cdir}")
    return FaissCollection(cdir)


# -------------------------
# Gold parsing (auto)
# -------------------------

def _detect_gold_file(data_dir: Path) -> Optional[Path]:
    # common names
    for cand in [
        data_dir / "hp_mp_gold.tsv",
        data_dir / "gold.tsv",
        data_dir / "gold_mappings.tsv",
        data_dir / "mappings.tsv",
    ]:
        if cand.exists():
            return cand
    # otherwise first TSV with >=2 columns
    for p in sorted(data_dir.glob("*.tsv")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                row = next(csv.reader(f, delimiter="\t"))
            if len(row) >= 2:
                return p
        except Exception:
            continue
    return None


def load_gold_pairs(path: Path) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        rows = list(reader)
    if not rows:
        return pairs

    header_l = [c.strip().lower() for c in rows[0]]
    has_header = any(("hp" in c or "mp" in c) for c in header_l)

    def add(a: str, b: str):
        a2 = canonicalize_id(a)
        b2 = canonicalize_id(b)
        if a2 and b2:
            pairs.append((a2, b2))

    if has_header:
        hp_i = None
        mp_i = None
        for i, h in enumerate(header_l):
            if hp_i is None and "hp" in h:
                hp_i = i
            if mp_i is None and "mp" in h:
                mp_i = i
        if hp_i is None or mp_i is None:
            hp_i, mp_i = 0, 1

        for r in rows[1:]:
            if len(r) <= max(hp_i, mp_i):
                continue
            add(r[hp_i], r[mp_i])
    else:
        if len(rows[0]) >= 2:
            add(rows[0][0], rows[0][1])
        for r in rows[1:]:
            if len(r) >= 2:
                add(r[0], r[1])

    return pairs


# -------------------------
# Ablation core
# -------------------------

def model_name_for(cfg: BuildConfig, model_key: str) -> str:
    return cfg.base_model_name if model_key == "base" else cfg.ft_model_path


def collections_for(model_key: str, direction: str) -> Tuple[str, str]:
    # Your config keys only
    if direction == "hp_to_mp":
        return (f"hp_{model_key}", f"mp_{model_key}")
    if direction == "mp_to_hp":
        return (f"mp_{model_key}", f"hp_{model_key}")
    raise ValueError(direction)


def evaluate_condition(
    cfg: BuildConfig,
    gold_pairs: List[Tuple[str, str]],
    direction: str,
    model_key: str,
    query_mode: str,  # label_only | full_src
    ks: List[int],
    out_dir: Path,
    project_root: Path,
    device: torch.device,
    tok,
    mdl,
) -> Dict:
    src_collection, tgt_collection = collections_for(model_key, direction)
    src_db = load_collection(cfg, src_collection, project_root)
    tgt_db = load_collection(cfg, tgt_collection, project_root)

    tgt_prefix = COLLECTIONS[tgt_collection]["id_prefix"]
    max_k = max(ks)

    total = 0
    hits_at = {k: 0 for k in ks}
    skipped_src_missing = 0
    skipped_tgt_missing = 0

    threshold = float(cfg.threshold)
    overfetch_mult = int(cfg.overfetch_mult)
    max_limit_mult = int(cfg.max_limit_mult)

    pbar = tqdm(gold_pairs, desc=f"{direction} | {model_key} | {query_mode}", unit="pair")
    for hp_id, mp_id in pbar:
        src_id, tgt_id = (hp_id, mp_id) if direction == "hp_to_mp" else (mp_id, hp_id)

        # ids already canonicalized from gold loader
        if tgt_id not in tgt_db.id2pos:
            skipped_tgt_missing += 1
            continue

        payload = src_db.get_payload_by_id(src_id)
        if payload is None:
            skipped_src_missing += 1
            continue

        label = str(payload.get("label", "") or "").strip()
        definition = str(payload.get("definition", "") or "").strip()
        synonyms = payload.get("synonyms", []) or []

        if query_mode == "label_only":
            qtext = f"Label: {label if label else src_id}"
        elif query_mode == "full_src":
            qtext = build_embedding_text(label if label else src_id, definition, list(synonyms), cfg.synonym_cap)
        else:
            raise ValueError(query_mode)

        qvec = embed_one(qtext, tok, mdl, device, cfg.max_length)

        need = max_k
        limit = max(need * overfetch_mult, need)
        max_limit = need * max_limit_mult

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

        total += 1
        for k in ks:
            if tgt_id in set(filtered[:k]):
                hits_at[k] += 1

        if total and 50 in hits_at:
            pbar.set_postfix({"eval": total, "r@50": hits_at[50] / total})

    result = {
        "direction": direction,
        "model_key": model_key,
        "query_mode": query_mode,
        "src_collection": src_collection,
        "tgt_collection": tgt_collection,
        "evaluated": total,
        "skipped_src_missing": skipped_src_missing,
        "skipped_tgt_missing": skipped_tgt_missing,
    }
    for k in ks:
        result[f"recall@{k}"] = (hits_at[k] / total) if total else 0.0

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", type=str, default=None, help="TSV with HP/MP ids (default: auto-detect in data/)")
    ap.add_argument("--ks", nargs="*", type=int, default=[1, 50, 100, 200])
    ap.add_argument("--models", nargs="*", type=str, default=["base", "ft"], choices=["base", "ft"])
    ap.add_argument("--modes", nargs="*", type=str, default=["label_only", "full_src"], choices=["label_only", "full_src"])
    args = ap.parse_args()

    cfg = BuildConfig()

    # project root = parent of faiss_vdb directory (since you run python faiss_vdb/ablation_faiss.py)
    project_root = Path(__file__).resolve().parents[1]

    data_dir = project_root / cfg.data_dir
    gold_path = Path(args.gold) if args.gold else _detect_gold_file(data_dir)
    if gold_path is None or not gold_path.exists():
        raise SystemExit("Could not find gold TSV. Pass --gold <path> or place one under data/")

    gold_pairs = load_gold_pairs(gold_path)
    if not gold_pairs:
        raise SystemExit(f"Gold file empty/unreadable: {gold_path}")

    out_root = project_root / "ablation_study" / "results"
    out_root.mkdir(parents=True, exist_ok=True)

    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"run_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "gold": str(gold_path),
                "ks": args.ks,
                "models": args.models,
                "modes": args.modes,
                "build_config": asdict(cfg),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # keep threads low on mac to avoid OpenMP weirdness
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    device = resolve_device(cfg.device)

    summary_rows: List[Dict] = []
    for model_key in args.models:
        model_name = model_name_for(cfg, model_key)
        print(f"\n[MODEL] {model_key}: {model_name}")
        tok, mdl = load_encoder(model_name, device)

        for direction in ["hp_to_mp", "mp_to_hp"]:
            for mode in args.modes:
                out_dir = run_dir / direction / model_key / mode
                res = evaluate_condition(cfg, gold_pairs, direction, model_key, mode, args.ks, out_dir, project_root, device, tok, mdl)
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