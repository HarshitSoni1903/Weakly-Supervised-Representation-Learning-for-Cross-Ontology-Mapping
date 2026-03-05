from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent
VDB_SRC = PROJECT_ROOT / "vdb_src"
if str(VDB_SRC) not in sys.path:
    sys.path.insert(0, str(VDB_SRC))


def canonicalize_id(x: str) -> str:
    return str(x).strip().replace(":", "_")


def infer_collection_from_id(src_id: str) -> str:
    return str(src_id).split(":")[0].lower()


def _clean_text(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def _split_synonyms(value: str) -> List[str]:
    text = str(value or "").strip()
    if not text:
        return []
    if ";" in text:
        return [x.strip() for x in text.split(";") if x.strip()]
    if "|" in text:
        return [x.strip() for x in text.split("|") if x.strip()]
    return [text]


DEFAULT_RUN_CONFIG: Dict[str, Any] = {
    "inputs": [{"id": "HP:0001627"}],
    "inputs_file": "",
    "inputs_format": "auto",
    "tgt_collection": "mp",
    "top_k": 50,
    "query_mode": "configured",
    "query_config": {
        "fields": ["label", "definition", "synonyms", "iri"],
        "include_field_names": True,
        "field_name_map": {
            "label": "Label",
            "definition": "Definition",
            "synonyms": "Synonyms",
            "iri": "IRI"
        },
        "list_delimiter": "; ",
    },
    "include_scores_in_pool": False,
    "out_dir": "outputs/bert_trial",
    "out_prefix": "hp_to_mp",
    "write_rows_csv": True,
    "write_pool_json": True,
    "write_bertmap_tsv": True,
    "retrieve_overrides": {
        "ft_model_path": "models/sap_FT",
        "device": "auto",
        "max_length": 512,
        "synonym_cap": 10,
        "threshold": 0.0,
        "overfetch_mult": 4,
        "max_limit_mult": 20
    },
    "stages": {
        "train": {
            "enabled": False,
            "command": "",
            "cwd": ".",
            "env": {},
            "allow_fail": False
        },
        "bertmap": {
            "enabled": False,
            "runner": "command",
            "command": "",
            "cwd": ".",
            "env": {},
            "allow_fail": False,
            "src_onto_path": "data/hp.owl",
            "tgt_onto_path": "data/mp.owl",
            "output_path": "outputs/bert_trial/bertmap_final",
            "command_template_vars": {},
            "python_api": {
                "module_candidates": [
                    "deeponto.align.bertmap",
                    "deeponto.alignment.bertmap",
                    "deeponto.align.bertmap.pipeline",
                    "deeponto.alignment.bertmap.pipeline"
                ],
                "class_candidates": ["BERTMapPipeline", "BERTMap"],
                "constructor_kwargs": {},
                "run_method_candidates": ["run", "align", "match", "run_pipeline"],
                "run_kwargs": {},
                "accept_candidate_pool_kw_candidates": [
                    "candidate_pool_path",
                    "candidate_mappings_path",
                    "candidate_path",
                    "candidate_file",
                    "candidate_pool_file"
                ]
            }
        }
    }
}


def normalize_input_item(item: Any) -> Dict[str, Any]:
    if isinstance(item, str):
        return {"id": item.strip()}
    if not isinstance(item, dict):
        raise ValueError(f"Unsupported input type: {type(item)}")

    row = dict(item)
    syn = row.get("synonyms", [])
    if isinstance(syn, str):
        syn = _split_synonyms(syn)
    elif not isinstance(syn, list):
        syn = [str(syn)]
    row["synonyms"] = [str(x).strip() for x in syn if str(x).strip()]

    if "id" in row:
        row["id"] = str(row.get("id", "") or "").strip()
    if "label" in row:
        row["label"] = str(row.get("label", "") or "").strip()
    if "definition" in row:
        row["definition"] = str(row.get("definition", "") or "").strip()
    return row


def load_inputs_from_file(path: Path, input_format: str) -> List[Dict[str, Any]]:
    fmt = input_format
    if fmt == "auto":
        suffix = path.suffix.lower()
        if suffix == ".csv":
            fmt = "csv"
        elif suffix in {".jsonl", ".ndjson"}:
            fmt = "jsonl"
        else:
            fmt = "json"

    raw_items: List[Any] = []
    if fmt == "json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            raw_items = obj
        elif isinstance(obj, dict) and isinstance(obj.get("inputs"), list):
            raw_items = obj["inputs"]
        else:
            raise ValueError("JSON input must be a list or {'inputs': [...]} object.")
    elif fmt == "jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text:
                raw_items.append(json.loads(text))
    elif fmt == "csv":
        with open(path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                parsed = dict(row)
                if "synonyms" in parsed:
                    parsed["synonyms"] = _split_synonyms(parsed.get("synonyms", ""))
                raw_items.append(parsed)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    return [normalize_input_item(x) for x in raw_items]


def merge_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = deepcopy(DEFAULT_RUN_CONFIG)

    def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                _deep_update(dst[k], v)
            else:
                dst[k] = v
        return dst

    if args.config:
        loaded = json.loads(Path(args.config).read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("Config must be a JSON object.")
        _deep_update(cfg, loaded)

    if args.inputs:
        cfg["inputs_file"] = str(args.inputs)
    if args.tgt_collection:
        cfg["tgt_collection"] = args.tgt_collection
    if args.top_k is not None:
        cfg["top_k"] = int(args.top_k)
    if args.query_mode:
        cfg["query_mode"] = args.query_mode
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    if args.out_prefix:
        cfg["out_prefix"] = args.out_prefix
    if args.include_scores_in_pool:
        cfg["include_scores_in_pool"] = True
    if args.inputs_format:
        cfg["inputs_format"] = args.inputs_format

    return cfg


def apply_retrieve_overrides(cfg: Any, overrides: Dict[str, Any]) -> None:
    for key, value in (overrides or {}).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)


def fill_payload_from_id(
    cfg: Any,
    project_root: Path,
    src_payload: Dict[str, Any],
    src_db_cache: Dict[str, object],
    load_collection_fn: Any,
) -> Dict[str, Any]:
    src_id = str(src_payload.get("id", "") or "").strip()
    if not src_id:
        return src_payload

    src_col = infer_collection_from_id(src_id)
    if src_col not in src_db_cache:
        src_db_cache[src_col] = load_collection_fn(cfg, src_col, project_root)

    src_db = src_db_cache[src_col]
    cid = canonicalize_id(src_id)
    db_payload = src_db.get_payload_by_id(cid)

    if db_payload:
        for key, value in db_payload.items():
            cur = src_payload.get(key)
            if key not in src_payload:
                src_payload[key] = value
                continue
            if isinstance(cur, list):
                if not cur:
                    src_payload[key] = value
                continue
            if not str(cur or "").strip():
                src_payload[key] = value

    if not str(src_payload.get("label", "") or "").strip():
        src_payload["label"] = cid

    return src_payload


def build_query_text(payload: Dict[str, Any], query_mode: str, query_config: Dict[str, Any]) -> str:
    if query_mode == "label_only":
        return f"Label: {_clean_text(payload.get('label', ''))}"

    if query_mode == "full_src":
        label = _clean_text(payload.get("label", ""))
        definition = _clean_text(payload.get("definition", ""))
        synonyms = payload.get("synonyms", []) or []
        syns = []
        seen = set()
        for s in synonyms:
            s2 = _clean_text(s)
            if not s2 or s2 == label or s2 in seen:
                continue
            syns.append(s2)
            seen.add(s2)
        parts = [f"Label: {label}"]
        if definition:
            parts.append(f"Definition: {definition}")
        if syns:
            parts.append("Synonyms: " + "; ".join(syns))
        return "\n".join(parts)

    if query_mode != "configured":
        raise ValueError(f"Unsupported query_mode: {query_mode}")

    fields = query_config.get("fields", ["label", "definition", "synonyms"])
    include_field_names = bool(query_config.get("include_field_names", True))
    field_name_map = query_config.get("field_name_map", {}) or {}
    list_delimiter = str(query_config.get("list_delimiter", "; ") or "; ")

    parts: List[str] = []
    for field in fields:
        raw = payload.get(field)
        text = ""
        if isinstance(raw, list):
            vals = [_clean_text(x) for x in raw if _clean_text(x)]
            text = list_delimiter.join(vals)
        elif raw is not None:
            text = _clean_text(raw)

        if not text:
            continue

        if include_field_names:
            name = str(field_name_map.get(field, field)).strip() or field
            parts.append(f"{name}: {text}")
        else:
            parts.append(text)

    if not parts:
        fallback = _clean_text(payload.get("label", ""))
        return f"Label: {fallback}" if fallback else ""

    return "\n".join(parts)


def custom_fetch_top_k(
    cfg: Any,
    qtext: str,
    tgt_db: Any,
    model: Any,
    tokenizer: Any,
    top_k: int,
) -> List[Tuple[str, float]]:
    from config import COLLECTIONS
    from retrieve import embed_one, resolve_device

    device = resolve_device(cfg.device)
    qvec = embed_one(qtext, tokenizer, model, device, cfg.max_length)

    threshold = float(cfg.threshold)
    overfetch_mult = int(cfg.overfetch_mult)
    max_limit_mult = int(cfg.max_limit_mult)

    need = int(top_k)
    limit = max(need * overfetch_mult, need)
    max_limit = max(need * max_limit_mult, need)

    tgt_prefix = None
    for name, meta in COLLECTIONS.items():
        if tgt_db.cdir.name == name:
            tgt_prefix = meta["id_prefix"]
            break
    if tgt_prefix is None:
        raise RuntimeError(f"Could not infer id_prefix for target collection: {tgt_db.cdir.name}")

    filtered: List[Tuple[str, float]] = []
    while True:
        scores, idxs = tgt_db.search(qvec, min(limit, tgt_db.count()))
        filtered = []

        for score, idx in zip(scores.tolist(), idxs.tolist()):
            if idx < 0:
                continue
            pid = tgt_db.id_at_pos(idx)
            if not pid:
                continue
            if not pid.startswith(tgt_prefix):
                continue
            if float(score) < threshold:
                continue
            filtered.append((pid, float(score)))
            if len(filtered) >= need:
                break

        if len(filtered) >= need or limit >= max_limit:
            break
        limit = min(limit * 2, max_limit)

    return filtered[:need]


def run_stage_command(name: str, stage_cfg: Dict[str, Any]) -> Dict[str, Any]:
    enabled = bool(stage_cfg.get("enabled", False))
    if not enabled:
        return {"name": name, "enabled": False, "status": "skipped"}

    command = str(stage_cfg.get("command", "") or "").strip()
    if not command:
        raise ValueError(f"Stage '{name}' is enabled but 'command' is empty")

    cwd = Path(str(stage_cfg.get("cwd", ".") or "."))
    if not cwd.is_absolute():
        cwd = PROJECT_ROOT / cwd

    env = os.environ.copy()
    for k, v in (stage_cfg.get("env", {}) or {}).items():
        env[str(k)] = str(v)

    allow_fail = bool(stage_cfg.get("allow_fail", False))
    result = subprocess.run(
        command,
        shell=True,
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
    )

    status = "ok" if result.returncode == 0 else "failed"
    stage_result = {
        "name": name,
        "enabled": True,
        "status": status,
        "return_code": int(result.returncode),
        "command": command,
        "cwd": str(cwd),
        "stdout_tail": "\n".join(result.stdout.splitlines()[-50:]),
        "stderr_tail": "\n".join(result.stderr.splitlines()[-50:]),
    }

    if result.returncode != 0 and not allow_fail:
        raise RuntimeError(f"Stage '{name}' failed with return code {result.returncode}")

    return stage_result


def _is_truthy(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}


def check_deeponto_available() -> str:
    spec = importlib.util.find_spec("deeponto")
    if spec is None:
        raise RuntimeError(
            "DeepOnto is required for bertmap stage but is not importable. "
            "Install it in this environment, then rerun."
        )
    mod = importlib.import_module("deeponto")
    return str(getattr(mod, "__version__", "unknown"))


def format_command_template(command: str, variables: Dict[str, str]) -> str:
    try:
        return command.format(**variables)
    except KeyError as e:
        missing = str(e).strip("'")
        raise ValueError(f"BERTMap command template references unknown variable: {missing}") from e


def run_bertmap_python_api(
    stage_cfg: Dict[str, Any],
    variables: Dict[str, str],
) -> Dict[str, Any]:
    py_cfg = stage_cfg.get("python_api", {}) or {}
    module_candidates = py_cfg.get("module_candidates", []) or []
    class_candidates = py_cfg.get("class_candidates", []) or []
    run_methods = py_cfg.get("run_method_candidates", []) or []
    run_kwargs = dict(py_cfg.get("run_kwargs", {}) or {})
    ctor_kwargs = dict(py_cfg.get("constructor_kwargs", {}) or {})
    cand_kw_candidates = py_cfg.get("accept_candidate_pool_kw_candidates", []) or []

    # Allow template vars in kwargs.
    for k, v in list(ctor_kwargs.items()):
        if isinstance(v, str):
            ctor_kwargs[k] = format_command_template(v, variables)
    for k, v in list(run_kwargs.items()):
        if isinstance(v, str):
            run_kwargs[k] = format_command_template(v, variables)

    src_onto = variables["src_onto_path"]
    tgt_onto = variables["tgt_onto_path"]
    output_path = variables["bertmap_output_path"]
    candidate_pool = variables["candidate_pool_json"]

    last_err: Optional[Exception] = None
    for module_name in module_candidates:
        try:
            mod = importlib.import_module(module_name)
        except Exception as e:
            last_err = e
            continue

        for class_name in class_candidates:
            cls = getattr(mod, class_name, None)
            if cls is None:
                continue

            local_ctor = dict(ctor_kwargs)
            # Common constructor names in different BERTMap wrappers.
            if "src_onto_path" not in local_ctor:
                local_ctor["src_onto_path"] = src_onto
            if "tgt_onto_path" not in local_ctor:
                local_ctor["tgt_onto_path"] = tgt_onto
            if "output_path" not in local_ctor:
                local_ctor["output_path"] = output_path
            for cand_kw in cand_kw_candidates:
                if cand_kw not in local_ctor:
                    local_ctor[cand_kw] = candidate_pool

            try:
                obj = cls(**local_ctor)
            except Exception as e:
                last_err = e
                continue

            for method_name in run_methods:
                fn = getattr(obj, method_name, None)
                if fn is None:
                    continue
                try:
                    result = fn(**run_kwargs)
                    return {
                        "name": "bertmap",
                        "enabled": True,
                        "status": "ok",
                        "runner": "python_api",
                        "module": module_name,
                        "class": class_name,
                        "method": method_name,
                        "result_repr": repr(result),
                    }
                except Exception as e:
                    last_err = e
                    continue

    raise RuntimeError(
        "Could not run DeepOnto BERTMap via python_api candidates. "
        f"Last error: {repr(last_err)}"
    )


def run_bertmap_stage(
    stage_cfg: Dict[str, Any],
    variables: Dict[str, str],
) -> Dict[str, Any]:
    enabled = bool(stage_cfg.get("enabled", False))
    if not enabled:
        return {"name": "bertmap", "enabled": False, "status": "skipped"}

    deeponto_version = check_deeponto_available()

    src_onto_path = Path(variables["src_onto_path"])
    tgt_onto_path = Path(variables["tgt_onto_path"])
    pool_path = Path(variables["candidate_pool_json"])
    if not src_onto_path.exists():
        raise FileNotFoundError(f"Missing source ontology: {src_onto_path}")
    if not tgt_onto_path.exists():
        raise FileNotFoundError(f"Missing target ontology: {tgt_onto_path}")
    if not pool_path.exists():
        raise FileNotFoundError(f"Missing candidate pool JSON: {pool_path}")

    runner = str(stage_cfg.get("runner", "command") or "command").strip().lower()
    if runner == "python_api":
        out = run_bertmap_python_api(stage_cfg=stage_cfg, variables=variables)
        out["deeponto_version"] = deeponto_version
        return out

    command = str(stage_cfg.get("command", "") or "").strip()
    if not command:
        raise ValueError("BERTMap stage enabled with runner=command but command is empty.")

    rendered = format_command_template(command, variables)
    cmd_cfg = {
        "enabled": True,
        "command": rendered,
        "cwd": stage_cfg.get("cwd", "."),
        "env": stage_cfg.get("env", {}),
        "allow_fail": stage_cfg.get("allow_fail", False),
    }
    out = run_stage_command("bertmap", cmd_cfg)
    out["runner"] = "command"
    out["deeponto_version"] = deeponto_version
    return out


def run_retrieval(
    cfg: Any,
    inputs: Iterable[Dict[str, Any]],
    tgt_collection: str,
    top_k: int,
    query_mode: str,
    query_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    from retrieve import load_collection, load_encoder, resolve_device

    project_root = PROJECT_ROOT
    device = resolve_device(cfg.device)
    tok, model = load_encoder(cfg.ft_model_path, device)
    tgt_db = load_collection(cfg, tgt_collection, project_root)

    src_db_cache: Dict[str, object] = {}
    rows: List[Dict[str, Any]] = []

    for src in inputs:
        payload = fill_payload_from_id(
            cfg=cfg,
            project_root=project_root,
            src_payload=dict(src),
            src_db_cache=src_db_cache,
            load_collection_fn=load_collection,
        )

        qtext = build_query_text(payload, query_mode=query_mode, query_config=query_config)
        if not qtext:
            continue

        candidates = custom_fetch_top_k(
            cfg=cfg,
            qtext=qtext,
            tgt_db=tgt_db,
            model=model,
            tokenizer=tok,
            top_k=top_k,
        )

        for rank, (target_id, score) in enumerate(candidates, start=1):
            rows.append(
                {
                    "src_id": str(payload.get("id", "") or "").strip(),
                    "src_label": str(payload.get("label", "") or "").strip(),
                    "target_id": target_id,
                    "rank": rank,
                    "score": float(score),
                    "tgt_collection": tgt_collection,
                    "query_mode": query_mode,
                    "query_text": qtext,
                }
            )

    return rows


def build_candidate_pool(rows: Iterable[Dict[str, Any]], include_scores: bool) -> Dict[str, Any]:
    pool: Dict[str, Any] = defaultdict(list)
    for row in rows:
        src_id = str(row.get("src_id", "") or "").strip()
        target_id = str(row.get("target_id", "") or "").strip()
        if not src_id or not target_id:
            continue

        src_key = canonicalize_id(src_id)
        target_key = canonicalize_id(target_id)
        if include_scores:
            pool[src_key].append({"target_id": target_key, "score": float(row["score"])})
        else:
            pool[src_key].append(target_key)
    return dict(pool)


def write_rows_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "src_id",
                "src_label",
                "target_id",
                "rank",
                "score",
                "tgt_collection",
                "query_mode",
                "query_text",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_bertmap_tsv(path: Path, rows: List[Dict[str, Any]]) -> int:
    skipped = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["src_id", "target_id", "score", "rank"])
        for row in rows:
            src_id = str(row.get("src_id", "") or "").strip()
            if not src_id:
                skipped += 1
                continue
            writer.writerow(
                [
                    canonicalize_id(src_id),
                    canonicalize_id(str(row["target_id"])),
                    f"{float(row['score']):.8f}",
                    int(row["rank"]),
                ]
            )
    return skipped


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-shot ontology mapping runner: optional train, top-k candidate generation, optional BERTMap stage."
    )
    parser.add_argument("--config", type=str, default="", help="Path to JSON run config.")
    parser.add_argument("--init-config", type=str, default="", help="Write default config and exit.")
    parser.add_argument("--inputs", type=str, default="", help="Input file (json/jsonl/csv).")
    parser.add_argument("--inputs-format", choices=["auto", "json", "jsonl", "csv"], default="auto")
    parser.add_argument("--tgt-collection", type=str, default="")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--query-mode", choices=["configured", "full_src", "label_only"], default="")
    parser.add_argument("--out-dir", type=str, default="")
    parser.add_argument("--out-prefix", type=str, default="")
    parser.add_argument("--include-scores-in-pool", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.init_config:
        out = Path(args.init_config)
        write_json(out, DEFAULT_RUN_CONFIG)
        print(out)
        return

    run_cfg = merge_config(args)

    input_file = str(run_cfg.get("inputs_file", "") or "").strip()
    if input_file:
        inputs = load_inputs_from_file(Path(input_file), str(run_cfg.get("inputs_format", "auto")))
    else:
        inputs = [normalize_input_item(x) for x in run_cfg.get("inputs", [])]

    from config import BuildConfig

    cfg = BuildConfig()
    apply_retrieve_overrides(cfg, run_cfg.get("retrieve_overrides", {}))

    stage_results: List[Dict[str, Any]] = []
    stages_cfg = run_cfg.get("stages", {}) or {}

    if isinstance(stages_cfg.get("train"), dict):
        stage_results.append(run_stage_command("train", stages_cfg["train"]))

    rows = run_retrieval(
        cfg=cfg,
        inputs=inputs,
        tgt_collection=str(run_cfg["tgt_collection"]),
        top_k=int(run_cfg["top_k"]),
        query_mode=str(run_cfg.get("query_mode", "configured")),
        query_config=run_cfg.get("query_config", {}) or {},
    )

    out_dir = Path(str(run_cfg["out_dir"]))
    prefix = str(run_cfg["out_prefix"])
    rows_csv_path = out_dir / f"{prefix}.candidates.csv"
    pool_json_path = out_dir / f"{prefix}.candidate_pool.json"
    tsv_path = out_dir / f"{prefix}.bertmap_candidates.tsv"
    summary_path = out_dir / f"{prefix}.run_summary.json"

    skipped_in_tsv = 0
    if bool(run_cfg.get("write_rows_csv", True)):
        write_rows_csv(rows_csv_path, rows)

    if bool(run_cfg.get("write_pool_json", True)):
        pool = build_candidate_pool(rows, bool(run_cfg.get("include_scores_in_pool", False)))
        write_json(
            pool_json_path,
            {
                "format": "deeponto_candidate_pool",
                "query_mode": str(run_cfg.get("query_mode", "configured")),
                "query_config": run_cfg.get("query_config", {}),
                "candidate_pool": pool,
                "include_scores": bool(run_cfg.get("include_scores_in_pool", False)),
            },
        )

    if bool(run_cfg.get("write_bertmap_tsv", True)):
        skipped_in_tsv = write_bertmap_tsv(tsv_path, rows)

    bertmap_stage_cfg = stages_cfg.get("bertmap") if isinstance(stages_cfg.get("bertmap"), dict) else None
    bertmap_out = ""
    if bertmap_stage_cfg:
        src_onto_path = Path(str(bertmap_stage_cfg.get("src_onto_path", "data/hp.owl")))
        tgt_onto_path = Path(str(bertmap_stage_cfg.get("tgt_onto_path", "data/mp.owl")))
        output_path = Path(str(bertmap_stage_cfg.get("output_path", out_dir / f"{prefix}.bertmap_final")))

        if not src_onto_path.is_absolute():
            src_onto_path = PROJECT_ROOT / src_onto_path
        if not tgt_onto_path.is_absolute():
            tgt_onto_path = PROJECT_ROOT / tgt_onto_path
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path

        extra_vars = {
            str(k): str(v)
            for k, v in (bertmap_stage_cfg.get("command_template_vars", {}) or {}).items()
        }
        vars_for_bertmap = {
            "project_root": str(PROJECT_ROOT),
            "out_dir": str(out_dir),
            "out_prefix": prefix,
            "rows_csv": str(rows_csv_path),
            "candidate_pool_json": str(pool_json_path),
            "bertmap_candidates_tsv": str(tsv_path),
            "src_onto_path": str(src_onto_path),
            "tgt_onto_path": str(tgt_onto_path),
            "bertmap_output_path": str(output_path),
            "top_k": str(run_cfg.get("top_k")),
            "query_mode": str(run_cfg.get("query_mode")),
        }
        vars_for_bertmap.update(extra_vars)

        stage_results.append(
            run_bertmap_stage(
                stage_cfg=bertmap_stage_cfg,
                variables=vars_for_bertmap,
            )
        )
        bertmap_out = str(output_path)

    summary = {
        "num_inputs": len(inputs),
        "num_candidate_rows": len(rows),
        "num_unique_sources_with_ids": len(build_candidate_pool(rows, False)),
        "skipped_rows_without_src_id_for_tsv": skipped_in_tsv,
        "config_used": {
            "tgt_collection": run_cfg.get("tgt_collection"),
            "top_k": run_cfg.get("top_k"),
            "query_mode": run_cfg.get("query_mode"),
            "query_config": run_cfg.get("query_config", {}),
            "retrieve_overrides": run_cfg.get("retrieve_overrides", {}),
        },
        "stage_results": stage_results,
        "outputs": {
            "rows_csv": str(rows_csv_path) if bool(run_cfg.get("write_rows_csv", True)) else "",
            "candidate_pool_json": str(pool_json_path) if bool(run_cfg.get("write_pool_json", True)) else "",
            "bertmap_tsv": str(tsv_path) if bool(run_cfg.get("write_bertmap_tsv", True)) else "",
            "bertmap_output_path": bertmap_out,
        },
    }

    write_json(summary_path, summary)
    print(summary_path)


if __name__ == "__main__":
    main()
