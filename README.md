# New_Caps

FAISS-based cross-ontology retrieval for Human Phenotype Ontology (HP) and Mammalian Phenotype Ontology (MP), using SapBERT embeddings (finetuned model by default).

## Abstract

Ontology mapping enables semantic interoperability across independently developed biomedical ontologies by identifying semantically equivalent or closely related concepts. This capability underpins integrative analysis, cross-resource inference, and knowledge graph integration. Contemporary ontology mapping systems typically follow a three-stage architecture: (i) candidate generation, (ii) neural ranking or scoring, and (iii) mapping selection and logical repair. While recent advances have improved neural scoring models, candidate generation remains a major performance bottleneck. Because downstream matchers can only score retrieved candidates, retrieval failures impose a strict upper bound on achievable recall.

To address this gap, we propose a retrieval-centric framework for ontology mapping that strengthens candidate generation using weak semantic priors and conservative representation adaptation. Our approach builds upon SapBERT, a transformer encoder trained on ontology concept normalization that captures biomedical synonymy through supervised contrastive learning. Although SapBERT provides strong general representations, its embedding space remains static and may not optimally reflect the semantic geometry required for specific ontology mapping tasks.

To adapt representations without introducing supervision leakage, we leverage SeMRA's (Semantic Mapping Reasoning Assembler) Raw Semantic Mapping Database as a weak semantic prior. SeMRA aggregates cross-resource biomedical mappings at scale, providing high coverage but noisy alignments. Rather than treating these mappings as gold supervision, we interpret them as weak positive signals and fine-tune SapBERT using a positive-only contrastive objective with in-batch negatives (Multiple Negatives Ranking loss). This formulation encourages semantically related concepts to move closer in the embedding space while implicitly separating unrelated concepts. Gold ontology mappings are strictly reserved for evaluation.

The resulting embedding-based retrieval method is modular and model-agnostic, replacing existing candidate generation methods (like lexical TF-IDF) without modifying downstream ranking or repair components. As a proof of concept, we evaluate our framework on mapping between the Mammalian Phenotype Ontology (MP) and the Human Phenotype Ontology (HP), two independently curated but biologically related ontologies exhibiting substantial lexical and structural asymmetries. Compared to lexical retrieval (Recall@50 ~= 0.66), embedding-based candidate generation improves coverage to ~= 0.84, and weakly supervised fine-tuning further increases it to ~= 0.89. When integrated into a BERTMap-style pipeline with fixed candidate sets of size 50, these gains translate into nearly doubling top-ranked alignment accuracy (Recall@1), demonstrating that retrieval capacity, rather than scoring architecture, is the primary performance limiter.

These findings establish candidate generation as a first-class design component in ontology mapping. Weak semantic priors combined with positive-only contrastive adaptation provide a scalable and principled mechanism for raising the effective performance ceiling of ontology mapping systems across domains characterized by lexical sparsity and structural heterogeneity.

## What this repo does

- Builds vector databases from OWL ontologies (`data/hp.owl`, `data/mp.owl`).
- Stores one FAISS index per collection (`db/hp`, `db/mp`).
- Retrieves top-k cross-ontology candidates (HP -> MP or MP -> HP).
- Runs ablation/evaluation against gold mappings (`data/hp_mp_gold.tsv`).

## Code-only project map

- `vdb_src/retrieve.py`: main retrieval implementation (`fetch_top_k`, encoder/loading utilities).
- `vdb_src/build_vdb.py`: build FAISS + metadata from OWL.
- `vdb_src/wrapper.py`: batch retrieval API to CSV.
- `vdb_src/demo.py`: main demo script (batch retrieval to CSV).
- `vdb_src/ablation_faiss.py`: recall@k ablation runner.
- `vdb_src/config.py`: all runtime/build settings.
- `finetune_src/*`: finetuning and trial analysis scripts.

## Setup

```bash
cd <PROJECT_ROOT>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick demo

Use `demo.py` as the project demo. It uses batch retrieval and writes ranked candidates to CSV.

```bash
cd <PROJECT_ROOT>
source .venv/bin/activate
python vdb_src/demo.py
```

Expected behavior:
- Creates `batch_results.csv` in the project root.
- Uses the predefined `inputs` list in `vdb_src/demo.py`.

What this demo is showing:
- ID-driven retrieval: when an input has `id` (example: `HP:0001627`), missing fields are auto-filled from the source ontology collection, then used for retrieval.
- Label-only retrieval: you can provide only `label` and still generate candidates.
- Enriched text retrieval: providing `label + definition + synonyms` gives richer query text for better semantic matching.
- Ranked candidate generation: each source concept gets top-k target ontology candidates with cosine similarity scores.

## Build indexes from OWL

Build all configured collections:

```bash
python vdb_src/build_vdb.py
```

Outputs per collection:
- `index.faiss`
- `meta.jsonl`
- `id2pos.json`
- `label2pos.json`

## Retrieval usage

### 1) Batch retrieval to CSV

Run provided example:

```bash
python vdb_src/demo.py
```

This calls `batch_retrieve_to_csv(...)` in `wrapper.py` and writes `batch_results.csv`.

Demo output schema (`batch_results.csv`):
- `src_id`: source concept id if provided
- `src_label`: source concept label used in query construction
- `target_id`: retrieved candidate id from target ontology
- `rank`: candidate rank (1..k)
- `score`: similarity score
- `tgt_collection`: target collection used for retrieval (`mp` in demo)
- `query_mode`: query construction mode (`full_src` by default)

Overall logic path used by the demo:
1. Load config and encoder.
2. Load target FAISS collection.
3. For each input payload, normalize fields.
4. If `id` exists, fill missing text fields from source FAISS metadata.
5. Build query text (label-only or full text with definition/synonyms).
6. Embed query with SapBERT encoder.
7. Search target FAISS index and apply filtering/threshold rules.
8. Write ranked candidates to CSV.

### 2) Main retrieval logic

The core retrieval code is in `vdb_src/retrieve.py`, especially:
- `fetch_top_k(...)` for candidate generation
- `load_collection(...)` for FAISS collection loading
- `load_encoder(...)` for model/tokenizer setup

Common use cases supported by this demo flow:
- Candidate generation for ontology mapping pipelines before reranking/repair.
- Rapid semantic lookup of likely MP counterparts for HP phenotypes.
- Batch offline generation of candidate sets for evaluation experiments.
- Comparing query strategies (`label_only` vs `full_src`) by swapping `query_mode` in wrapper usage.

## Evaluation / ablation

Runs both directions and writes metrics under `ablation_study/results/run_*`.

```bash
python vdb_src/ablation_faiss.py --models ft
```

Main outputs:
- `run_config.json`
- per-condition `metrics.json`
- `summary.csv`

## Important specifics from config

`BuildConfig` in `vdb_src/config.py` is the central runtime/build contract used across scripts.

### Full `BuildConfig` parameters

- `db_dir: str = "db"`
  - Root folder for FAISS collections and metadata.
  - Used by: `build_vdb.py`, `retrieve.py`, `wrapper.py`, `sanity_checks.py`.
- `data_dir: str = "data"`
  - Root data folder (gold mappings and ontology inputs).
  - Used by: `ablation_faiss.py` (gold autodetection).
- `hp_owl_path: str = "data/hp.owl"`
  - HP ontology source path.
  - Used by: `build_vdb.py`.
- `mp_owl_path: str = "data/mp.owl"`
  - MP ontology source path.
  - Used by: `build_vdb.py`.
- `ft_model_path: str = "models/sap_FT"`
  - Finetuned encoder path (default encoder used by current collections).
  - Used by: `build_vdb.py`, `wrapper.py`, `demo.py`, `ablation_faiss.py`.
- `max_length: int = 512`
  - Token truncation length for encoder input.
  - Used by: `build_vdb.py` (`embed_batch`), `retrieve.py` (`embed_one`).
- `embed_batch_size: int = 64`
  - Batch size for ontology concept embedding during index build.
  - Used by: `build_vdb.py`.
- `device: str = "auto"`
  - Runtime device preference (`auto | cpu | cuda`), resolved in code as `cuda > mps > cpu` for `auto`.
  - Used by: `build_vdb.py`, `retrieve.py`, `wrapper.py`, `demo.py`, `ablation_faiss.py`.
- `synonym_cap: int = 10`
  - Max synonyms included when constructing embedding text.
  - Used by: `build_vdb.py`, `retrieve.py`.
- `threshold: float = 0.0`
  - Minimum similarity threshold for accepted retrieval candidates.
  - Used by: `retrieve.py` (including calls from wrapper/demo/ablation).
- `overfetch_mult: int = 4`
  - Initial search width multiplier relative to requested `top_k`.
  - Used by: `retrieve.py`.
- `max_limit_mult: int = 20`
  - Upper bound multiplier for iterative widening during retrieval.
  - Used by: `retrieve.py`.

### Collection:

Current active collections:
- `hp`: `{ "source": "owl", "prefix": "hp", "model": "ft", "id_prefix": "HP_" }`
- `mp`: `{ "source": "owl", "prefix": "mp", "model": "ft", "id_prefix": "MP_" }`

How these fields affect behavior:
- `prefix`: picks ontology source (`hp` uses `hp.owl`, `mp` uses `mp.owl`) in `build_vdb.py`.
- `model`: selects encoder family (`ft` currently active; `base` blocks are scaffolded but commented in config).
- `id_prefix`: retrieval post-filter to keep only target ontology IDs with expected prefix.

### Demo-relevant config behavior

For `python vdb_src/demo.py`, retrieval behavior is controlled by:
- `ft_model_path`
- `device`
- `max_length`
- `synonym_cap`
- `threshold`
- `overfetch_mult`
- `max_limit_mult`
- `db_dir`

These are consumed through `wrapper.py` -> `retrieve.py` and directly shape candidate quality, search breadth, and runtime.


## Notes

- On macOS, scripts set OpenMP-related env vars internally before Torch/FAISS imports.
- IDs are canonicalized from `HP:0001627` to `HP_0001627` internally.
