from pathlib import Path
from config import BuildConfig
from ablation_faiss import resolve_device, load_collection, load_encoder
from retrieve import fetch_top_k

cfg = BuildConfig()
project_root = Path(__file__).resolve().parents[1]

device = resolve_device(cfg.device)
tok, mdl = load_encoder(cfg.ft_model_path, device)

tgt_db = load_collection(cfg, "mp_ft", project_root)

src_payload = {
    "label": "Abnormal heart morphology",
    "definition": "Any structural anomaly of the heart.",
    "synonyms": ["Cardiac malformation"],
}

top_ids = fetch_top_k(
    cfg,
    src_payload,
    tgt_db,
    mdl,
    tok,
    top_k=50,
    query_mode="full_src",
)

print(top_ids)