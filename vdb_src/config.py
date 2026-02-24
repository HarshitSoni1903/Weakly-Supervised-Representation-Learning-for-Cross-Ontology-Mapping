from dataclasses import dataclass
from typing import Dict, Optional

# All paths are relative to project root.

@dataclass
class BuildConfig:
    # IO
    db_dir: str = "db"          # location for FAISS + metadata
    data_dir: str = "data"      # OWL files

    hp_owl_path: str = "data/hp.owl"
    mp_owl_path: str = "data/mp.owl"

    # Models
    # base_model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    ft_model_path: str = "models/sap_FT"  # local path

    # Embedding
    max_length: int = 512
    embed_batch_size: int = 64
    device: str = "auto"  # auto | cpu | cuda
    synonym_cap: int = 10

    # Retrieval
    threshold: float = 0.0
    overfetch_mult: int = 4
    max_limit_mult: int = 20  # max_limit = top_k * max_limit_mult


COLLECTIONS: Dict[str, Dict[str, str]] = {
    # "hp_base": {"source": "owl", "prefix": "hp", "model": "base", "id_prefix": "HP_"},
    # "mp_base": {"source": "owl", "prefix": "mp", "model": "base", "id_prefix": "MP_"},
    "hp_ft":   {"source": "owl", "prefix": "hp", "model": "ft",   "id_prefix": "HP_"},
    "mp_ft":   {"source": "owl", "prefix": "mp", "model": "ft",   "id_prefix": "MP_"},
}
