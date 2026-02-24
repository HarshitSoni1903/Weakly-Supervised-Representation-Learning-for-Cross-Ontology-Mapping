from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import faiss  # type: ignore

from config import BuildConfig


def _paths(cfg: BuildConfig, collection: str) -> Tuple[Path, Path, Path]:
    cdir = Path(cfg.db_dir) / collection
    return (
        cdir / "index.faiss",
        cdir / "meta.jsonl",
        cdir / "id2pos.json",
    )


def check_count(cfg: BuildConfig, collection: str) -> int:
    index_path, meta_path, _ = _paths(cfg, collection)
    if not index_path.exists():
        raise RuntimeError(f"[{collection}] missing {index_path}")
    if not meta_path.exists():
        raise RuntimeError(f"[{collection}] missing {meta_path}")

    idx = faiss.read_index(str(index_path))
    n_index = int(idx.ntotal)

    n_meta = 0
    with open(meta_path, "r", encoding="utf-8") as f:
        for _ in f:
            n_meta += 1

    print(f"[SANITY] {collection}: index={n_index} meta={n_meta}")
    if n_index <= 0:
        raise RuntimeError(f"[{collection}] index is empty")
    if n_index != n_meta:
        raise RuntimeError(f"[{collection}] index/meta count mismatch")
    return n_index


def self_retrieval_check(cfg: BuildConfig, collection: str, trials: int = 3, top_k: int = 5) -> None:
    index_path, meta_path, _ = _paths(cfg, collection)
    idx = faiss.read_index(str(index_path))

    n = int(idx.ntotal)
    if n <= 0:
        raise RuntimeError(f"[{collection}] empty")

    for _ in range(trials):
        i = random.randint(0, n - 1)
        v = idx.reconstruct(i)  # vector stored in IndexFlat*
        D, I = idx.search(v.reshape(1, -1), top_k)
        hits = list(I[0])
        if i not in hits:
            raise RuntimeError(f"[{collection}] self retrieval failed for pos={i} hits={hits}")

    print(f"[SANITY] {collection}: self-retrieval OK ({trials} trials)")


def main() -> None:
    cfg = BuildConfig()
    # discover collections by folders in db_dir
    db = Path(cfg.db_dir)
    if not db.exists():
        raise RuntimeError(f"db_dir not found: {db}")

    for cdir in sorted([p for p in db.iterdir() if p.is_dir()]):
        collection = cdir.name
        check_count(cfg, collection)
        self_retrieval_check(cfg, collection, trials=3, top_k=5)


if __name__ == "__main__":
    main()
