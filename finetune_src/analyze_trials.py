import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# ============================================================
# Data loading
# ============================================================
def load_gold_pairs(csv_path: Path):
    df = pd.read_csv(csv_path)
    return (
        df["mp_label"].astype(str).tolist(),
        df["hp_label"].astype(str).tolist(),
    )


# ============================================================
# Embeddings + cosine
# ============================================================
def compute_embeddings(model, mp_labels, hp_labels, batch_size=256):
    mp_emb = model.encode(
        mp_labels, convert_to_tensor=True, batch_size=batch_size
    )
    hp_emb = model.encode(
        hp_labels, convert_to_tensor=True, batch_size=batch_size
    )

    cos = torch.nn.functional.cosine_similarity(mp_emb, hp_emb, dim=1)

    return (
        cos.cpu().numpy(),
        mp_emb.cpu().numpy(),
        hp_emb.cpu().numpy(),
    )


def summarize(cos):
    return {
        "mean": float(np.mean(cos)),
        "min": float(np.min(cos)),
        "max": float(np.max(cos)),
        "var": float(np.var(cos)),
    }


# ============================================================
# Visuals
# ============================================================
def plot_histogram(cos, title, out_path):
    plt.figure()
    plt.hist(cos, bins=50)
    plt.xlabel("Cosine similarity")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_tsne(embeddings, title, out_path):
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        random_state=13,
    )
    emb_2d = tsne.fit_transform(embeddings)

    plt.figure()
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], s=5)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", required=True, type=Path)
    ap.add_argument("--gold_csv", required=True, type=Path)
    ap.add_argument(
        "--trials",
        required=True,
        type=str,
        help="Comma-separated trial numbers, e.g. 0,2,11,12",
    )
    ap.add_argument(
        "--base_model",
        default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    )
    ap.add_argument("--output_dir", required=True, type=Path)
    args = ap.parse_args()

    trial_ids = [int(x.strip()) for x in args.trials.split(",")]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    mp_labels, hp_labels = load_gold_pairs(args.gold_csv)

    # --------------------------------------------------------
    # Base SapBERT
    # --------------------------------------------------------
    logging.info("Evaluating base SapBERT")
    base_model = SentenceTransformer(args.base_model)

    base_cos, base_mp_emb, base_hp_emb = compute_embeddings(
        base_model, mp_labels, hp_labels
    )

    base_stats = summarize(base_cos)
    logging.info("BASE stats: %s", base_stats)

    plot_histogram(
        base_cos,
        "Base SapBERT – Gold Cosine Similarity",
        args.output_dir / "hist_base.png",
    )

    plot_tsne(
        np.vstack([base_mp_emb, base_hp_emb]),
        "Base SapBERT – t-SNE (Gold)",
        args.output_dir / "tsne_base.png",
    )

    # Save metrics
    rows = []
    rows.append({"model": "base", **base_stats})

    # --------------------------------------------------------
    # Trial models
    # --------------------------------------------------------
    for tid in trial_ids:
        model_dir = args.runs_dir / f"trial_{tid}"
        if not model_dir.exists():
            logging.warning("Skipping trial %d (not found)", tid)
            continue

        logging.info("Evaluating trial %d", tid)
        model = SentenceTransformer(str(model_dir))

        cos, mp_emb, hp_emb = compute_embeddings(
            model, mp_labels, hp_labels
        )

        stats = summarize(cos)
        logging.info("Trial %d stats: %s", tid, stats)

        plot_histogram(
            cos,
            f"Trial {tid} – Gold Cosine Similarity",
            args.output_dir / f"hist_trial_{tid}.png",
        )

        plot_tsne(
            np.vstack([mp_emb, hp_emb]),
            f"Trial {tid} – t-SNE (Gold)",
            args.output_dir / f"tsne_trial_{tid}.png",
        )

        rows.append({"model": f"trial_{tid}", **stats})

    # --------------------------------------------------------
    # Save summary table
    # --------------------------------------------------------
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(
        args.output_dir / "summary_metrics.csv", index=False
    )

    logging.info("Wrote summary metrics to summary_metrics.csv")


if __name__ == "__main__":
    main()