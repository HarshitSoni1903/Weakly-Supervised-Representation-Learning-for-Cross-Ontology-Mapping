import argparse
import logging
import random
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd


# ============================================================
# Logging
# ============================================================
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# ============================================================
# Family normalization
# ============================================================
TYPE_SUFFIX_RE = re.compile(
    r"\s+(cell|gene|protein|rna|mrna|disease|phenotype|anatomy)$",
    re.IGNORECASE,
)

def family_key(label: str) -> str:
    s = str(label).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = TYPE_SUFFIX_RE.sub("", s)
    return s.strip()


# ============================================================
# Union-Find (family components)
# ============================================================
class UnionFind:
    def __init__(self):
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


# ============================================================
# Leakage check
# ============================================================
def check_family_leakage(df: pd.DataFrame):
    fams = {}
    for split in df["split"].unique():
        fams[split] = set(df[df["split"] == split]["subject_family"]) | \
                      set(df[df["split"] == split]["object_family"])

    splits = list(fams.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            inter = fams[splits[i]] & fams[splits[j]]
            logging.info(
                "Family overlap %s∩%s = %d",
                splits[i], splits[j], len(inter)
            )
            if inter:
                raise RuntimeError("FAMILY LEAKAGE DETECTED")


# ============================================================
# Main
# ============================================================
def main():
    setup_logger()

    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--train_frac", type=float, default=0.90)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    if args.train_frac + args.val_frac > 1.0:
        raise ValueError("train_frac + val_frac must be ≤ 1.0")

    rng = random.Random(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.pairs_csv)
    df.columns = [c.lower() for c in df.columns]

    if "subject_label" not in df.columns or "object_label" not in df.columns:
        raise ValueError("CSV must contain subject_label, object_label")

    # --------------------------------------------------------
    # Family normalization
    # --------------------------------------------------------
    df["subject_family"] = df["subject_label"].map(family_key)
    df["object_family"] = df["object_label"].map(family_key)

    before = len(df)
    df = df.drop_duplicates(
        subset=["subject_family", "object_family"]
    ).reset_index(drop=True)
    logging.info("Dropped %d duplicate family-pairs", before - len(df))

    # --------------------------------------------------------
    # Build family components
    # --------------------------------------------------------
    uf = UnionFind()
    for _, r in df.iterrows():
        uf.union(r["subject_family"], r["object_family"])

    df["family_component"] = df["subject_family"].map(uf.find)

    components = df["family_component"].unique().tolist()
    rng.shuffle(components)

    n = len(components)
    n_train = int(args.train_frac * n)
    n_val = int(args.val_frac * n)

    train_comps = set(components[:n_train])
    val_comps = set(components[n_train:n_train + n_val])

    def assign_split(comp):
        if comp in train_comps:
            return "train"
        if comp in val_comps:
            return "val"
        return "unused"

    df["split"] = df["family_component"].map(assign_split)

    # --------------------------------------------------------
    # Leakage check
    # --------------------------------------------------------
    check_family_leakage(df[df["split"] != "unused"])

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    for split in ["train", "val"]:
        out = df[df["split"] == split][
            ["subject_label", "object_label"]
        ].reset_index(drop=True)
        out.to_csv(out_dir / f"{split}.csv", index=False)
        logging.info("Wrote %s.csv: %d pairs", split, len(out))

    df.to_csv(out_dir / "pairs_with_components_and_split.csv", index=False)
    logging.info(
        "Components: train=%d val=%d total=%d",
        len(train_comps), len(val_comps), n
    )


if __name__ == "__main__":
    main()