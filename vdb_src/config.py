from dataclasses import dataclass
from typing import Dict, Optional

# All paths are relative to project root.

@dataclass
class BuildConfig:
    # IO
    db_dir: str = "db"          # location for FAISS + metadata
    data_dir: str = "data"      # OWL files

    # Models
    base_model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
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


COLLECTIONS: Dict[str, Dict[str, object]] = {
    "hp":   {
        "source": "owl", 
        "prefix": "hp", 
        "model": "ft",
        "owl_path": "hp_enriched.owl", 
        "id_prefixes": ["HP_"]},
    "mp":   {
        "source": "owl", 
        "prefix": "mp", 
        "model": "ft",
        "owl_path": "mp_enriched.owl", 
        "id_prefixes": ["MP_"]},
    "mondo": {
        "source": "owl",
        "model": "ft",
        "owl_path": "mondo.owl",
        "id_prefixes": ["MONDO_"],
    },
    "mesh": {
        "source": "owl",
        "model": "ft",
        "owl_path": "mesh_diseas.owl",
        "id_prefixes": ["mesh_"],
    },
    "hp_base":   {
        "source": "owl", 
        "prefix": "hp", 
        "model": "base",
        "owl_path": "hp_enriched.owl", 
        "id_prefixes": ["HP_"]},
    "mp_base":   {
        "source": "owl", 
        "prefix": "mp", 
        "model": "base",
        "owl_path": "mp_enriched.owl", 
        "id_prefixes": ["MP_"]},
    "mondo_base": {
        "source": "owl",
        "model": "base",
        "owl_path": "mondo.owl",
        "id_prefixes": ["MONDO_"],
    },
    "mesh_base": {
        "source": "owl",
        "model": "base",
        "owl_path": "mesh_disease.owl",
        "id_prefixes": ["mesh_"],
    },
    "doid_base": {
        "source": "owl",
        "model": "base",
        "owl_path": "doid.owl",
        "id_prefixes": ["DOID_"],
    },
    "doid": {
        "source": "owl",
        "model": "ft",
        "owl_path": "doid.owl",
        "id_prefixes": ["DOID_"],
    },
}