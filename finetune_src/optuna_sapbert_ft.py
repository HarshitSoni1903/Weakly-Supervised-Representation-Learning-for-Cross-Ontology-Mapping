import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer, InputExample, losses
from transformers import AdamW

import optuna


# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# ============================================================
# Data loaders
# ============================================================
def load_train_pairs(csv_path: Path):
    df = pd.read_csv(csv_path)
    return [
        InputExample(
            texts=[str(r["subject_label"]), str(r["object_label"])]
        )
        for _, r in df.iterrows()
    ]


def load_gold_pairs(csv_path: Path):
    df = pd.read_csv(csv_path)
    return (
        df["mp_label"].astype(str).tolist(),
        df["hp_label"].astype(str).tolist(),
    )


# ============================================================
# Cosine + delta computation
# ============================================================
def compute_cosines(model, mp_labels, hp_labels, batch_size=256):
    mp_emb = model.encode(
        mp_labels, convert_to_tensor=True, batch_size=batch_size
    )
    hp_emb = model.encode(
        hp_labels, convert_to_tensor=True, batch_size=batch_size
    )
    cos = torch.nn.functional.cosine_similarity(mp_emb, hp_emb, dim=1)
    return cos.cpu().numpy()


def summarize_deltas(base_cos, ft_cos):
    delta = ft_cos - base_cos
    improved = (delta > 0).sum()
    worsened = (delta < 0).sum()
    unchanged = (delta == 0).sum()

    return {
        "delta_mean": float(delta.mean()),
        "delta_std": float(delta.std()),
        "improved_frac": improved / len(delta),
        "worsened_frac": worsened / len(delta),
        "unchanged_frac": unchanged / len(delta),
    }


# ============================================================
# Training + evaluation (one trial)
# ============================================================
def run_trial(trial, args, train_examples, mp_labels, hp_labels, base_cos):

    lr = trial.suggest_float("lr", 1e-5, 5e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [256, 512])
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.05, 0.15)

    out_dir = args.output_dir / f"trial_{trial.number}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(args.model_name)
    model.max_seq_length = args.max_length

    train_loader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size,
        drop_last=True,
    )

    train_loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = int(
        len(train_loader) * args.epochs * warmup_ratio
    )

    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_class=AdamW,
        optimizer_params={"lr": lr},
        use_amp=args.fp16,
        output_path=str(out_dir),
        show_progress_bar=False,
    )

    # ---- Gold evaluation ----
    ft_cos = compute_cosines(model, mp_labels, hp_labels)

    stats = summarize_deltas(base_cos, ft_cos)

    logging.info(
        "[Trial %d] Δmean=%.5f Δstd=%.5f improved=%.2f worsened=%.2f",
        trial.number,
        stats["delta_mean"],
        stats["delta_std"],
        stats["improved_frac"],
        stats["worsened_frac"],
    )

    # Objective: maximize mean improvement, penalize variance increase
    score = stats["delta_mean"] - 0.25 * stats["delta_std"]

    return score


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_dir", required=True, type=Path)
    ap.add_argument("--gold_csv", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument(
        "--model_name",
        default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    )
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max_length", type=int, default=32)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--optuna_trials", type=int, default=10)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load training + gold data
    # --------------------------------------------------------
    train_examples = load_train_pairs(args.splits_dir / "train.csv")
    mp_labels, hp_labels = load_gold_pairs(args.gold_csv)

    logging.info("Loaded train=%d gold=%d",
                 len(train_examples), len(mp_labels))

    # --------------------------------------------------------
    # Base SapBERT evaluation (fixed)
    # --------------------------------------------------------
    base_model = SentenceTransformer(args.model_name)
    base_model.max_seq_length = args.max_length

    base_cos = compute_cosines(base_model, mp_labels, hp_labels)

    logging.info(
        "BASE SapBERT: mean=%.4f std=%.4f",
        base_cos.mean(), base_cos.std()
    )

    # --------------------------------------------------------
    # Optuna study
    # --------------------------------------------------------
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda t: run_trial(
            t, args, train_examples, mp_labels, hp_labels, base_cos
        ),
        n_trials=args.optuna_trials,
    )
    best_trial = study.best_trial
    best_model_dir = Path(best_trial.user_attrs["model_dir"])

    logging.info(
        "Best trial %d | params=%s",
        best_trial.number,
        best_trial.params,
    )

    # Reload best finetuned model
    best_model = SentenceTransformer(str(best_model_dir))

    best_cos, best_mp_emb, best_hp_emb = compute_cosines(
        best_model, mp_labels, hp_labels
    )

    logging.info(
        "BEST FT SapBERT: mean=%.4f std=%.4f",
        best_cos.mean(),
        best_cos.std(),
    )

    logging.info("Best trial: %s", study.best_trial.params)

    best_trial = study.best_trial
    best_model_dir = Path(best_trial.user_attrs["model_dir"])

    logging.info(
        "Best trial %d | params=%s",
        best_trial.number,
        best_trial.params,
    )

    # Reload best finetuned model
    best_model = SentenceTransformer(str(best_model_dir))

    best_cos, best_mp_emb, best_hp_emb = compute_cosines(
        best_model, mp_labels, hp_labels
    )

    logging.info(
        "BEST FT SapBERT: mean=%.4f std=%.4f",
        best_cos.mean(),
        best_cos.std(),
    )

if __name__ == "__main__":
    main()