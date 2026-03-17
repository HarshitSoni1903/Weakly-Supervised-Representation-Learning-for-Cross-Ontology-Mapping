from __future__ import annotations

import csv
import json
import logging
import os
import platform
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from owlready2 import get_ontology

from config import BuildConfig, COLLECTIONS, resolve_path

# OpenMP env (must happen before faiss/torch spin up threads) usually an issue on MacOS
if platform.system() == "Darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


# Logging

def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    log_path = resolve_path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(log_path / f"{name}.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger


# ID helpers

def canonicalize_id(raw: str, prefix: str = "") -> str:
    """
    Normalize to lowercase-prefix:numeric_id form.
    Examples:
        HP:0001234  -> hp:0001234
        MONDO:12345 -> mondo:12345
        mesh:D012345 -> mesh:D012345
    """
    s = str(raw).strip()
    if not s:
        return ""

    # split on first colon (or underscore for purl-style ids like HP_0001234)
    if ":" in s:
        ns, local = s.split(":", 1)
    elif "_" in s:
        ns, local = s.split("_", 1)
    else:
        # no namespace — use prefix param or return as-is lowercase
        return f"{prefix.lower()}:{s}" if prefix else s.lower()

    ns = (prefix if prefix else ns).lower()
    return f"{ns}:{local}"


def normalize_prefix(p: str) -> str:
    """Convert a config id_prefix like 'HP_' or 'MONDO_' to the canonical form 'hp:' or 'mondo:'."""
    p = p.strip()
    if "_" in p:
        ns = p.split("_", 1)[0]
        return f"{ns.lower()}:"
    if ":" in p:
        ns = p.split(":", 1)[0]
        return f"{ns.lower()}:"
    return p.lower()


def _clean(s: str) -> str:
    return " ".join(str(s).strip().split())


# Embedding text construction

def _is_junk(s: str) -> bool:
    """Catch stringified empty containers that slipped through OWL parsing."""
    return s in ("[]", "{}", "None", "none", "()")


def build_embedding_text(
    label: str,
    definition: str = "",
    synonyms: Optional[List[str]] = None,
    synonym_cap: int = 10,
) -> str:
    label = _clean(label)
    definition = _clean(definition) if definition else ""
    if _is_junk(definition):
        definition = ""

    syns: List[str] = []
    seen = set()
    for s in synonyms or []:
        s2 = _clean(s)
        if not s2 or s2 == label or s2 in seen or _is_junk(s2):
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


# Device resolution

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


# Model loading

def load_encoder(model_name: str, device: torch.device):
    resolved = resolve_path(model_name)
    if resolved.is_dir():
        model_name = str(resolved)
    else:
        if not _looks_like_hub_id(model_name):
            raise FileNotFoundError(
                f"Local model not found: '{model_name}'\n"
                f"  resolved to: {resolved}\n"
                f"  PROJECT_ROOT: {resolve_path('.')}\n"
                f"  Hint: check that the directory exists and config.py PROJECT_ROOT is correct."
            )
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device).eval()
    return tok, mdl


def _looks_like_hub_id(name: str) -> bool:
    """Hub ids look like 'org/model-name'. Local paths don't have exactly one slash between two alphanum parts."""
    parts = name.split("/")
    if len(parts) == 2 and all(p and not p.startswith(".") for p in parts):
        return True
    return False


def free_encoder(tok, mdl, device: torch.device):
    del tok, mdl
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ── Pooling & embedding ────────────────────────────────────────────────

def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.no_grad()
def embed_texts(
    texts: List[str],
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
) -> np.ndarray:
    """Embed one or many texts. Returns (N, dim) float32 array, L2-normalized."""
    # pre-truncate: rough char limit to avoid tokenizer processing huge strings
    # ~4 chars per token is a safe estimate, 2x headroom
    char_limit = max_length * 8
    texts = [t[:char_limit] for t in texts]

    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    out = model(**enc)
    pooled = _mean_pool(out.last_hidden_state, enc["attention_mask"])
    pooled = F.normalize(pooled, p=2, dim=1)
    return pooled.detach().cpu().numpy().astype("float32")


def embed_texts_batched(
    texts: List[str],
    tokenizer,
    model,
    device: torch.device,
    max_length: int,
    batch_size: int = 64,
    desc: str = "Embedding",
) -> np.ndarray:
    """Embed a large list in batches. Returns (N, dim) float32."""
    vectors: List[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc, unit="batch"):
        batch = texts[i : i + batch_size]
        vectors.append(embed_texts(batch, tokenizer, model, device, max_length))
    return np.vstack(vectors).astype("float32")


# OWL parsing helpers

def _first_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    try:
        if len(v) > 0:
            return str(v[0])
        return ""  # empty list/tuple
    except Exception:
        pass
    return str(v) if v else ""


def _list_str(v) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    try:
        return [str(x) for x in list(v) if str(x).strip()]
    except Exception:
        return [str(v)] if str(v).strip() else []


def _owl_class_id(cls) -> str:
    """
    Extract a canonical id from an OWL class.
    Priority: oboInOwl_id field > IRI tail > name attribute.
    Result is canonicalized to lowercase-prefix:id (e.g. hp:0001234).
    The namespace comes from the id/IRI itself, never overridden.
    """
    # priority 1: explicit id field
    if hasattr(cls, "oboInOwl_id"):
        v = _first_str(getattr(cls, "oboInOwl_id", None))
        if v:
            return canonicalize_id(v)

    # priority 2: IRI
    iri = str(getattr(cls, "iri", "") or "")
    tail = iri.split("#")[-1].rsplit("/", 1)[-1].strip()
    if tail:
        if "id.nlm.nih.gov/mesh/" in iri or "obo/mesh#" in iri or "purl.obolibrary.org/obo/mesh" in iri:
            return canonicalize_id(f"mesh:{tail}")
        return canonicalize_id(tail)

    # priority 3: name
    name = getattr(cls, "name", None)
    if name:
        return canonicalize_id(str(name))

    return canonicalize_id(tail)


_SYN_ATTRS = [
    "hasExactSynonym", "hasRelatedSynonym", "hasBroadSynonym",
    "hasNarrowSynonym", "hasSynonym", "synonym", "alternative_term",
]


def load_owl_concepts(owl_path: str, id_prefixes: Optional[List[str]] = None) -> List[Dict]:
    """
    Parse OWL file and return concept dicts.
    If id_prefixes is given (e.g. ["HP_"]), only keep classes whose raw id
    starts with one of those prefixes. This filters out PATO, GO, etc from HP files.
    """
    onto = get_ontology(Path(owl_path).as_posix()).load()
    concepts: List[Dict] = []

    norm_prefixes = None
    if id_prefixes:
        norm_prefixes = [normalize_prefix(p) for p in id_prefixes]

    for cls in onto.classes():
        cid = _owl_class_id(cls)

        # filter: skip classes that don't belong to this ontology
        if norm_prefixes and not any(cid.startswith(np) for np in norm_prefixes):
            continue

        label = _first_str(getattr(cls, "label", "")) or cid

        definition = ""
        if hasattr(cls, "IAO_0000115"):
            definition = _first_str(getattr(cls, "IAO_0000115", ""))
        if not definition and hasattr(cls, "definition"):
            definition = _first_str(getattr(cls, "definition", ""))
        if not definition and hasattr(cls, "description"):
            definition = _first_str(getattr(cls, "description", ""))

        synonyms: List[str] = []
        for attr in _SYN_ATTRS:
            if hasattr(cls, attr):
                synonyms.extend(_list_str(getattr(cls, attr, None)))

        # dedup preserving order
        seen = set()
        syn2: List[str] = []
        for s in synonyms:
            s = str(s).strip()
            if not s or s in seen:
                continue
            syn2.append(s)
            seen.add(s)

        concepts.append({
            "id": cid,
            "label": str(label),
            "definition": str(definition),
            "synonyms": syn2,
            "iri": str(getattr(cls, "iri", "") or ""),
        })

    return concepts


def load_csv_concepts(
    csv_path: str,
    id_col: str = "id",
    label_col: str = "label",
    def_col: str = "definition",
    syn_col: str = "synonyms",
    delimiter: str = "\t",
) -> List[Dict]:
    """Load concepts from a flat CSV/TSV file."""
    concepts: List[Dict] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            cid = canonicalize_id(row.get(id_col, ""))
            label = row.get(label_col, "").strip()
            definition = row.get(def_col, "").strip()
            syns_raw = row.get(syn_col, "").strip()
            syns = [s.strip() for s in syns_raw.split(";") if s.strip()] if syns_raw else []
            if not cid and not label:
                continue
            concepts.append({
                "id": cid,
                "label": label or cid,
                "definition": definition,
                "synonyms": syns,
            })
    return concepts


# FAISS collection (read side)

class FaissCollection:
    def __init__(self, cdir: Path, use_gilda: bool = False):
        self.cdir = cdir
        self._use_gilda = use_gilda
        self.index = faiss.read_index(str(cdir / "index.faiss"))
        # try GPU for faster search, fall back silently
        try:
            if hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        except Exception:
            pass
        self.meta_path = cdir / "meta.jsonl"
        self.id2pos: Dict[str, int] = json.loads((cdir / "id2pos.json").read_text(encoding="utf-8"))

        # load all metadata into memory
        self._meta: List[Dict] = []
        self.label2pos: Dict[str, int] = {}
        self.syn2pos: Dict[str, int] = {}
        self.pos2id: List[str] = []
        with open(self.meta_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    rec = json.loads(line)
                except Exception:
                    rec = {}
                self._meta.append(rec)
                self.pos2id.append(str(rec.get("id", "")))
                lbl = str(rec.get("label", "")).strip().lower()
                if lbl and lbl not in self.label2pos:
                    self.label2pos[lbl] = i
                for syn in rec.get("synonyms", []):
                    s = str(syn).strip().lower()
                    if s and s not in self.syn2pos:
                        self.syn2pos[s] = i

    def count(self) -> int:
        return int(self.index.ntotal)

    def get_payload_by_id(self, cid: str) -> Optional[Dict]:
        pos = self.id2pos.get(cid)
        if pos is None or pos >= len(self._meta):
            return None
        return self._meta[pos]

    def get_payload_by_label(self, label: str) -> Optional[Dict]:
        """Case-insensitive exact match on label."""
        pos = self.label2pos.get(label.strip().lower())
        if pos is None or pos >= len(self._meta):
            return None
        return self._meta[pos]

    @staticmethod
    def _normalize_tokens(text: str) -> str:
        """Lowercase, strip punctuation, sort tokens."""
        import re
        text = re.sub(r"'s\b", "", text)
        tokens = re.findall(r'[a-z0-9]+', text.lower())
        return " ".join(sorted(tokens))

    @staticmethod
    def _normalize_gilda(text: str) -> str:
        """Normalize using gilda's biomedical text normalizer."""
        try:
            from gilda.process import normalize
            return normalize(text) or ""
        except ImportError:
            # fall back to token-based if gilda not installed
            return FaissCollection._normalize_tokens(text)

    def _get_normalizer(self):
        """Return the appropriate normalize function based on config."""
        if self._use_gilda:
            return self._normalize_gilda
        return self._normalize_tokens

    def exact_match_ids(self, label: str, synonyms: Optional[List[str]] = None) -> List[str]:
        """Find target ids where source label/synonyms match target label/synonyms."""
        normalizer = self._get_normalizer()
        cache_key = "_token_index_gilda" if self._use_gilda else "_token_index"

        if not hasattr(self, cache_key):
            index: Dict[str, set] = {}
            for i, meta in enumerate(self._meta):
                cid = self.pos2id[i] if i < len(self.pos2id) else ""
                if not cid:
                    continue
                texts = [meta.get("label", "")]
                texts.extend(meta.get("synonyms", []))
                for t in texts:
                    norm = normalizer(t)
                    if norm:
                        index.setdefault(norm, set()).add(cid)
            setattr(self, cache_key, index)

        token_index = getattr(self, cache_key)
        matches = set()
        terms = [label] if label else []
        terms.extend(synonyms or [])
        for term in terms:
            norm = normalizer(term)
            if norm and norm in token_index:
                matches.update(token_index[norm])
        return list(matches)

    def search(self, qvec: np.ndarray, limit: int) -> Tuple[np.ndarray, np.ndarray]:
        scores, idxs = self.index.search(qvec, limit)
        return scores[0], idxs[0]

    def reconstruct(self, pos: int) -> np.ndarray:
        return self.index.reconstruct(pos)

    def id_at_pos(self, pos: int) -> str:
        if pos < 0 or pos >= len(self.pos2id):
            return ""
        return self.pos2id[pos]


def load_collection(cfg: BuildConfig, name: str) -> FaissCollection:
    cdir = resolve_path(cfg.db_dir) / name
    if not cdir.exists():
        raise SystemExit(f"Missing collection dir: {cdir}")
    return FaissCollection(cdir, use_gilda=cfg.use_gilda)


# Writing a collection to disk

def write_collection(
    cfg: BuildConfig,
    collection: str,
    concepts: List[Dict],
    tokenizer,
    model,
    device: torch.device,
    logger: Optional[logging.Logger] = None,
) -> None:
    cdir = resolve_path(cfg.db_dir) / collection
    cdir.mkdir(parents=True, exist_ok=True)

    concepts.sort(key=lambda d: d["id"])

    texts = [
        build_embedding_text(
            c.get("label", ""),
            c.get("definition", ""),
            c.get("synonyms", []),
            cfg.synonym_cap,
        )
        for c in concepts
    ]

    X = embed_texts_batched(texts, tokenizer, model, device, cfg.max_length, cfg.embed_batch_size, desc=f"[EMBED] {collection}")
    dim = int(X.shape[1])

    index = faiss.IndexFlatIP(dim)
    index.add(X)
    faiss.write_index(index, str(cdir / "index.faiss"))

    with open(cdir / "meta.jsonl", "w", encoding="utf-8") as f:
        for c in concepts:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    id2pos = {c["id"]: i for i, c in enumerate(concepts)}
    label2pos: Dict[str, int] = {}
    for i, c in enumerate(concepts):
        lbl = str(c.get("label", "")).strip()
        if lbl and lbl not in label2pos:
            label2pos[lbl] = i

    (cdir / "id2pos.json").write_text(json.dumps(id2pos), encoding="utf-8")
    (cdir / "label2pos.json").write_text(json.dumps(label2pos), encoding="utf-8")

    msg = f"[BUILD] {collection}: n={len(concepts)} dim={dim}"
    if logger:
        logger.info(msg)
    else:
        print(msg)


# Sanity checks

def check_count(cfg: BuildConfig, collection: str, logger: Optional[logging.Logger] = None) -> int:
    cdir = resolve_path(cfg.db_dir) / collection
    index_path = cdir / "index.faiss"
    meta_path = cdir / "meta.jsonl"

    log = logger or logging.getLogger("sanity")

    if not index_path.exists():
        log.error(f"[{collection}] FAIL: missing {index_path}")
        return 0
    if not meta_path.exists():
        log.error(f"[{collection}] FAIL: missing {meta_path}")
        return 0

    idx = faiss.read_index(str(index_path))
    n_index = int(idx.ntotal)
    n_meta = sum(1 for _ in open(meta_path, "r", encoding="utf-8"))

    if n_index <= 0:
        log.error(f"[SANITY] {collection}: FAIL — index is empty")
        return 0
    if n_index != n_meta:
        log.error(f"[SANITY] {collection}: FAIL — index/meta count mismatch (index={n_index}, meta={n_meta})")
        return n_index

    log.info(f"[SANITY] {collection}: count OK — index={n_index} meta={n_meta}")
    return n_index


def check_embedding_health(cfg: BuildConfig, collection: str, trials: int = 20, logger: Optional[logging.Logger] = None) -> None:
    """Check that embeddings haven't collapsed to a single point (all cosine-sim ~1.0)."""
    cdir = resolve_path(cfg.db_dir) / collection
    idx = faiss.read_index(str(cdir / "index.faiss"))
    n = int(idx.ntotal)
    log = logger or logging.getLogger("sanity")

    if n < 2:
        log.warning(f"[SANITY] {collection}: embedding health skipped (only {n} vectors)")
        return

    sims = []
    pairs_detail = []
    for _ in range(trials):
        i, j = random.sample(range(n), 2)
        vi = idx.reconstruct(i)
        vj = idx.reconstruct(j)
        sim = float(np.dot(vi, vj))
        sims.append(sim)
        pairs_detail.append(f"({i},{j})={sim:.4f}")

    avg_sim = sum(sims) / len(sims)
    min_sim = min(sims)
    max_sim = max(sims)

    if avg_sim > 0.95:
        log.error(f"[SANITY] {collection}: FAIL — embeddings near-collapsed (avg_sim={avg_sim:.4f}, range=[{min_sim:.4f}, {max_sim:.4f}], pairs: {', '.join(pairs_detail)})")
    elif avg_sim > 0.80:
        log.warning(f"[SANITY] {collection}: WARNING — high avg similarity, embeddings may be poorly differentiated (avg_sim={avg_sim:.4f}, range=[{min_sim:.4f}, {max_sim:.4f}])")
    else:
        log.info(f"[SANITY] {collection}: embedding health OK — avg_sim={avg_sim:.4f}, range=[{min_sim:.4f}, {max_sim:.4f}]")


def self_retrieval_check(cfg: BuildConfig, collection: str, trials: int = 5, top_k: int = 10, logger: Optional[logging.Logger] = None) -> None:
    cdir = resolve_path(cfg.db_dir) / collection
    idx = faiss.read_index(str(cdir / "index.faiss"))
    n = int(idx.ntotal)
    log = logger or logging.getLogger("sanity")

    if n <= 0:
        log.error(f"[SANITY] {collection}: FAIL — index is empty, cannot run self-retrieval")
        return

    failures = []
    for _ in range(trials):
        i = random.randint(0, n - 1)
        v = idx.reconstruct(i)
        D, I = idx.search(v.reshape(1, -1), top_k)
        if i not in list(I[0]):
            top_ids = list(I[0][:5])
            top_scores = [f"{s:.4f}" for s in D[0][:5].tolist()]
            failures.append(f"pos={i} (top-5 ids={top_ids}, scores={top_scores})")

    if failures:
        log.warning(f"[SANITY] {collection}: self-retrieval — {len(failures)}/{trials} missed top-{top_k}. Details: {'; '.join(failures)}")
    else:
        log.info(f"[SANITY] {collection}: self-retrieval OK ({trials}/{trials} found in top-{top_k})")


def run_all_sanity_checks(cfg: BuildConfig, collection: str, logger: Optional[logging.Logger] = None) -> None:
    check_count(cfg, collection, logger)
    check_embedding_health(cfg, collection, logger=logger)
    self_retrieval_check(cfg, collection, logger=logger)


# Retrieval

def resolve_payload(
    query: Dict,
    src_db: Optional[FaissCollection] = None,
) -> Dict:
    """
    Given a query dict with any combination of {id, label, definition, synonyms},
    try to fill missing fields from src_db.
    - If id is given: look up by id.
    - If only label: try case-insensitive exact match on label.
    - Otherwise: use whatever is provided.
    """
    payload = {
        "id": str(query.get("id", "") or "").strip(),
        "label": str(query.get("label", "") or "").strip(),
        "definition": str(query.get("definition", "") or "").strip(),
        "synonyms": query.get("synonyms", []) or [],
    }

    if src_db is None:
        return payload

    db_rec = None
    if payload["id"]:
        cid = canonicalize_id(payload["id"])
        payload["id"] = cid
        db_rec = src_db.get_payload_by_id(cid)
    elif payload["label"]:
        db_rec = src_db.get_payload_by_label(payload["label"])

    if db_rec:
        if not payload["label"]:
            payload["label"] = db_rec.get("label", "")
        if not payload["definition"]:
            payload["definition"] = db_rec.get("definition", "")
        if not payload["synonyms"]:
            payload["synonyms"] = db_rec.get("synonyms", [])
        if not payload["id"]:
            payload["id"] = db_rec.get("id", "")

    # last resort: use id as label
    if not payload["label"] and payload["id"]:
        payload["label"] = payload["id"]

    return payload


def rank_pool(
    pool: List[Tuple[str, float]],
    tgt_db: FaissCollection,
    src_label: str,
    threshold: float = 0.0,
) -> List[Tuple[str, float, str]]:
    """
    Shared ranking logic for a candidate pool.
    
    pool: list of (concept_id, cosine_score), already filtered to cosine >= min_cosine
    tgt_db: target collection (for exact_match_ids lookup)
    src_label: source concept label for lexical matching
    threshold: output threshold — non-boosted candidates below this are excluded
    
    Returns: sorted list of (concept_id, display_score, remarks)
      - At most ONE candidate gets boosted to 1.0 (unambiguous label match)
      - All others keep their cosine score, filtered by threshold
    """
    if not pool:
        return []

    pool_ids = {pid for pid, _ in pool}

    # check for unambiguous label match in pool
    label_match_ids = set(tgt_db.exact_match_ids(src_label, []))
    boost_id = None
    if len(label_match_ids) == 1:
        eid = next(iter(label_match_ids))
        if eid in pool_ids:
            boost_id = eid

    ranked: List[Tuple[str, float, str]] = []
    for pid, cosine in pool:
        if pid == boost_id:
            ranked.append((pid, 1.0, f"cosine={cosine:.6f}"))
        elif cosine >= threshold:
            ranked.append((pid, cosine, ""))

    ranked.sort(key=lambda x: -x[1])
    return ranked


def evaluate_predictions(
    predictions_by_src: Dict[str, str],
    gold_by_src: Dict[str, set],
    src_db: "FaissCollection",
    tgt_db: "FaissCollection",
) -> Dict:
    """
    Shared evaluation logic for mapper.
    
    predictions_by_src: {src_id: predicted_tgt_id} — only for mapped concepts
    gold_by_src: {src_id: set of valid tgt_ids} — one-to-many gold
    
    Returns dict with: testable, tp, fp, unmapped, accuracy, precision, 
                       gold_src_missing, gold_tgt_missing
    """
    tp, fp, unmapped = 0, 0, 0
    gold_src_missing, gold_tgt_missing = 0, 0

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

    testable = tp + fp + unmapped
    mapped = tp + fp
    accuracy = tp / testable if testable else 0.0
    precision = tp / mapped if mapped else 0.0

    return {
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


def fetch_top_k(
    cfg: BuildConfig,
    src_payload: Dict,
    tgt_db: FaissCollection,
    model,
    tokenizer,
    top_k: int,
    query_mode: str = "full_src",
    device: Optional[torch.device] = None,
) -> List[Dict]:
    """
    Retrieve top_k matches from tgt_db for a single source concept.
    Returns list of dicts: [{id, label, definition, synonyms, score, remarks}, ...]
    """
    if device is None:
        device = resolve_device(cfg.device)

    label = str(src_payload.get("label", "") or "").strip()
    definition = str(src_payload.get("definition", "") or "").strip()
    synonyms = src_payload.get("synonyms", []) or []

    if query_mode == "label_only":
        qtext = f"Label: {label}"
    elif query_mode == "full_src":
        qtext = build_embedding_text(label, definition, list(synonyms), cfg.synonym_cap)
    else:
        raise ValueError(f"Unknown query_mode: {query_mode}")

    qvec = embed_texts([qtext], tokenizer, model, device, cfg.max_length)
    threshold = float(cfg.threshold)
    min_cosine = cfg.lexical_min_cosine
    fetch_k = min(cfg.faiss_fetch_k, tgt_db.count())

    spec = COLLECTIONS.get(tgt_db.cdir.name, {})
    raw_prefixes = spec.get("id_prefixes") or spec.get("id_prefix") or []
    if isinstance(raw_prefixes, str):
        raw_prefixes = [raw_prefixes]
    norm_prefixes = [normalize_prefix(p) for p in raw_prefixes]

    def ok_prefix(pid: str) -> bool:
        if not norm_prefixes:
            return True
        return any(pid.startswith(p) for p in norm_prefixes)

    # fetch from FAISS, filter to pool
    scores, idxs = tgt_db.search(qvec, fetch_k)
    pool: List[Tuple[str, float]] = []
    seen_ids: set = set()
    for s, ix in zip(scores.tolist(), idxs.tolist()):
        if ix < 0:
            continue
        if s < min_cosine:
            break
        pid = tgt_db.id_at_pos(ix)
        if not pid or not ok_prefix(pid) or pid in seen_ids:
            continue
        pool.append((pid, float(s)))
        seen_ids.add(pid)

    # rank with shared logic
    ranked = rank_pool(pool, tgt_db, label, threshold)

    # enrich results
    results: List[Dict] = []
    for pid, score, remarks in ranked[:top_k]:
        meta = tgt_db.get_payload_by_id(pid) or {}
        results.append({
            "id": pid,
            "label": meta.get("label", ""),
            "definition": meta.get("definition", ""),
            "synonyms": meta.get("synonyms", []),
            "score": score,
            "remarks": remarks,
        })
    return results


# CSV/TSV writing

def write_results_csv(
    results: List[Dict],
    out_path: str | Path,
    delimiter: str = "\t",
    gold_src_ids: Optional[set] = None,
) -> Path:
    """
    Write retrieval results to CSV/TSV.
    Each item in results is:
      {src_id, src_label, matches: [{id, label, definition, synonyms, score, rank}]}
    If gold_src_ids is provided, adds an 'in_gold' column.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    has_gold = gold_src_ids is not None
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=delimiter)
        header = ["src_id", "src_label", "tgt_id", "tgt_label", "rank", "score", "remarks"]
        if has_gold:
            header.append("in_gold")
        w.writerow(header)
        for item in results:
            src_id = item.get("src_id", "")
            src_label = item.get("src_label", "")
            for m in item.get("matches", []):
                row = [
                    src_id,
                    src_label,
                    m.get("id", ""),
                    m.get("label", ""),
                    m.get("rank", ""),
                    f"{m.get('score', 0.0):.8f}",
                    m.get("remarks", ""),
                ]
                if has_gold:
                    row.append("1" if src_id in gold_src_ids else "0")
                w.writerow(row)
    return out_path


# Gold file loading (for ablation)

def _delimiter_for(path: Path) -> str:
    return "," if path.suffix.lower() == ".csv" else "\t"


def load_gold_pairs(
    gold_path: Path,
    src_col: Optional[str] = None,
    tgt_col: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """Load gold (src_id, tgt_id) pairs from a TSV/CSV. IDs are canonicalized."""
    pairs: List[Tuple[str, str]] = []
    delim = _delimiter_for(gold_path)

    with open(gold_path, "r", encoding="utf-8", newline="") as f:
        rows = [row for row in csv.reader(f, delimiter=delim) 
                if row and not row[0].startswith("#") and any(str(c).strip() for c in row)]

    if not rows:
        return pairs

    header_l = [c.strip().lower() for c in rows[0]]
    src_i: Optional[int] = None
    tgt_i: Optional[int] = None

    if src_col and tgt_col:
        # columns explicitly given — find them by exact match
        s, t = src_col.strip().lower(), tgt_col.strip().lower()
        for i, h in enumerate(header_l):
            if src_i is None and h == s:
                src_i = i
            if tgt_i is None and h == t:
                tgt_i = i
        if src_i is None or tgt_i is None:
            raise ValueError(f"Column(s) not found in {gold_path}: src_col={src_col!r} tgt_col={tgt_col!r}. Header: {rows[0]}")
        data_rows = rows[1:]
    else:
        # heuristic: try to detect src/tgt columns
        src_keys = ("src", "source", "left", "subj", "subject")
        tgt_keys = ("tgt", "target", "right", "dest", "destination", "obj", "object")
        for i, h in enumerate(header_l):
            if src_i is None and any(k in h for k in src_keys):
                src_i = i
            if tgt_i is None and any(k in h for k in tgt_keys):
                tgt_i = i

        if src_i is not None and tgt_i is not None:
            data_rows = rows[1:]
        else:
            # no header detected, use first two columns
            src_i, tgt_i = 0, 1
            data_rows = rows

    for row in data_rows:
        if len(row) <= max(src_i, tgt_i):
            continue
        a = canonicalize_id(row[src_i])
        b = canonicalize_id(row[tgt_i])
        if a and b:
            pairs.append((a, b))

    return pairs


# Model name resolver (base vs ft)

def model_name_for(cfg: BuildConfig, model_key: str) -> str:
    return cfg.base_model_name if model_key == "base" else cfg.ft_model_path


def collection_name_for_model(base_name: str, model_key: str) -> str:
    """hp + ft -> hp, hp + base -> hp_base"""
    return base_name if model_key == "ft" else f"{base_name}_base"
