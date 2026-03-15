from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

@dataclass
class BuildConfig:
    db_dir: str = "db"
    data_dir: str = "data"

    base_model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    ft_model_path: str = "models/sap_FT"

    # embedding
    max_length: int = 512
    embed_batch_size: int = 128 
    device: str = "auto"
    synonym_cap: int = 10

    # retrieval
    threshold: float = 0.9
    overfetch_mult: int = 4
    max_limit_mult: int = 20    # max_limit = top_k * max_limit_mult

    # build flags
    monitor_mode: bool = True  # prompt user with 2 samples before building
    rebuild: bool = False       # overwrite existing collections

    # logging
    log_dir: str = "logs"


def resolve_path(rel: str) -> Path:
    """Resolve a config path against PROJECT_ROOT. Absolute paths pass through."""
    p = Path(rel)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


# Collection definitions
# Each collection maps to one OWL file + one model variant.
# id_prefixes is used to filter concepts and validate retrieval results.

COLLECTIONS: Dict[str, Dict] = {
    "hp": {
        "source": "owl",
        "prefix": "hp",
        "model": "ft",
        "owl_path": "hp_enriched.owl",
        "id_prefixes": ["HP_"],
    },
    "mp": {
        "source": "owl",
        "prefix": "mp",
        "model": "ft",
        "owl_path": "mp_enriched.owl",
        "id_prefixes": ["MP_"],
    },
    "hp_base": {
        "source": "owl",
        "prefix": "hp",
        "model": "base",
        "owl_path": "hp_enriched.owl",
        "id_prefixes": ["HP_"],
    },
    "mp_base": {
        "source": "owl",
        "prefix": "mp",
        "model": "base",
        "owl_path": "mp_enriched.owl",
        "id_prefixes": ["MP_"],
    },
    "mondo": {
        "source": "owl",
        "model": "ft",
        "owl_path": "mondo.owl",
        "id_prefixes": ["MONDO_"],
    },
    "mesh": {
        "source": "owl",
        "model": "ft",
        "owl_path": "mesh_disease.owl",
        "id_prefixes": ["mesh_"],
    },
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


# Ablation study presets
# Keys are shorthand names you pass on the CLI. All params that used to be
# CLI-only arguments now live here so runs are reproducible from config alone.
# Gold file: relative to data_dir
# "src_col": column in gold file with id usually like hp:0000001
# "tgt_col": column in gold file with id usually like mp:0000001
# These columns have to coincide with the src_collection and tgt_collection prefixes, 
# so for src_collection "hp" the src_col should be the column in the gold file having have values like "hp:0000001" in that column.

# "modes" is the query mode to use for that ablation, e.g. "label_only" or "full_src". 
# Label only means only use the label of the source concept for retrieval, while full_src means use the full embedding text (label + def + synonyms).
# Full src will query the other information from the vdb to fill the embedding text, to generate the richer context. 
# For best matching, use full_src.
# "reverse" means to also run the ablation in reverse direction, i.e. swap src_collection and tgt_collection and run the same evaluation. 


ABLATIONS: Dict[str, Dict] = {
    "hp2mp": {
        "src_collection": "hp",
        "tgt_collection": "mp",
        "gold_file": "hp_mp_gold.tsv",  
        "src_col": "src_id",            
        "tgt_col": "tgt_id",            
        "ks": [1, 50, 100, 200],
        "models": ["ft"],  #["base", "ft"]
        "modes": ["full_src"],  #["label_only", "full_src"],         
        "reverse": True,
    },
    "mondo2mesh": {
        "src_collection": "mondo",
        "tgt_collection": "mesh",
        "gold_file": "gilda_mondo_mesh_predictions.sssom.tsv",
        "src_col": "subject_id",
        "tgt_col": "object_id",
        "ks": [1, 50, 100, 200],
        "models": ["base", "ft"],
        "modes": ["full_src"],
        "reverse": True,
    },
    "mondo2doid": {
        "src_collection": "mondo",
        "tgt_collection": "doid",
        "gold_file": "mondo_doid_gold.tsv",
        "src_col": "subject_id",
        "tgt_col": "object_id",
        "ks": [1, 50, 100, 200],
        "models": ["base", "ft"],
        "modes": ["label_only", "full_src"],
        "reverse": True,
    },
}


# Mapping study presets
# Full ontology mapping. Uses same study keys as ablations where possible.
# threshold: minimum cosine similarity to include a mapping (CLI overrides this)
# top_k: number of candidates per source concept
# reverse: also map tgt->src and save a separate file

MAPPINGS: Dict[str, Dict] = {
    "hp_mp": {
        "src_collection": "hp",
        "tgt_collection": "mp",
        "gold_file": "hp_mp_gold.tsv",
        "src_col": "src_id",
        "tgt_col": "tgt_id",
        "threshold": 0.9,
        "top_k": 1,
        "reverse": True,
    },
    "mondo_mesh": {
        "src_collection": "mondo",
        "tgt_collection": "mesh",
        "gold_file": "gilda_mondo_mesh_predictions.sssom.tsv",
        "src_col": "subject_id",
        "tgt_col": "object_id",
        "threshold": 0.9,
        "top_k": 1,
        "reverse": True,
    },
    "mondo_doid": {
        "src_collection": "mondo",
        "tgt_collection": "doid",
        "gold_file": "mondo_doid_gold.tsv",
        "src_col": "subject_id",
        "tgt_col": "object_id",
        "threshold": 0.9,
        "top_k": 1,
        "reverse": True,
    },
}
