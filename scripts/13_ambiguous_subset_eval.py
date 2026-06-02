#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ExperimentConfig, NOISE_PRESETS
from src.forward_model import generate_dataset
from src.modeling import evaluate_setting, precision_at_k, threshold_for_top_k, top_k_class_counts


SUMMARY_COLS = [
    "pr_auc",
    "precision_at_10pct",
    "recall_at_top10pct",
    "roc_auc",
    "brier",
    "fpr_radiation_mimic_top10pct",
    "fpr_exogenic_complex_organic_top10pct",
    "top10_positive_count",
    "top10_radiation_mimic_count",
    "top10_exogenic_count",
    "top10_noise_count",
]


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for setting, group in metrics.groupby("setting", sort=False):
        row = {"setting": setting, "model": group["model"].iloc[0], "n_seeds": len(group)}
        for col in SUMMARY_COLS:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=1)) if len(group) > 1 else 0.0
            row[col] = f"{row[f'{col}_mean']:.4f} +/- {row[f'{col}_std']:.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def subset_metrics(preds: pd.DataFrame, setting: str, model: str) -> dict:
    y = preds["y"].to_numpy()
    score = preds["score"].to_numpy()
    threshold = threshold_for_top_k(score, k_frac=0.1)
    y_pred_top = (score >= threshold).astype(int)

    out = {
        "setting": setting,
        "model": model,
        "n_subset": len(preds),
        "positive_rate_subset": float(np.mean(y)) if len(y) else np.nan,
        "pr_auc": float(average_precision_score(y, score)) if len(np.unique(y)) > 1 else np.nan,
        "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else np.nan,
        "precision_at_10pct": precision_at_k(y, score, 0.1),
        "recall_at_top10pct": float(recall_score(y, y_pred_top, zero_division=0)),
        "brier": float(brier_score_loss(y, score)),
        "fpr_radiation_mimic_top10pct": _fpr_for_class(preds, "radiation_mimic", threshold),
        "fpr_exogenic_complex_organic_top10pct": _fpr_for_class(preds, "exogenic_complex_organic", threshold),
    }
    out.update(top_k_class_counts(preds, score, 0.1))
    return out


def _fpr_for_class(preds: pd.DataFrame, class_name: str, threshold: float) -> float:
    mask = preds["z"].to_numpy() == class_name
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(preds.loc[mask, "score"].to_numpy() >= threshold))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8000)
    parser.add_argument("--rho-geo", type=float, default=0.75)
    parser.add_argument("--rho-rad", type=float, default=0.75)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=3421)
    parser.add_argument("--noise-condition", choices=NOISE_PRESETS.keys(), default="moderate")
    parser.add_argument("--smoothing-window", type=int, default=None)
    parser.add_argument("--model", choices=["rf", "hgb", "logreg"], default="logreg")
    parser.add_argument("--ambiguous-low", type=float, default=0.4)
    parser.add_argument("--ambiguous-high", type=float, default=0.7)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    rows = []
    predictions = []
    settings = ["spectral_only", "full", "context_only"]

    for i in range(args.seeds):
        seed = args.base_seed + i
        cfg = ExperimentConfig(
            seed=seed,
            noise_condition=args.noise_condition,
            smoothing_window=args.smoothing_window,
        )
        df = generate_dataset(n=args.n, rho_geo=args.rho_geo, rho_rad=args.rho_rad, cfg=cfg)

        _, spectral_preds, _ = evaluate_setting(df, setting="spectral_only", model_name=args.model, seed=seed)
        ambiguous_ids = set(
            spectral_preds.loc[
                (spectral_preds["score"] >= args.ambiguous_low)
                & (spectral_preds["score"] <= args.ambiguous_high),
                "sample_id",
            ]
        )
        if not ambiguous_ids:
            print(f"seed={seed} ambiguous subset empty")
            continue

        pred_map = {"spectral_only": spectral_preds}
        for setting in ["full", "context_only"]:
            _, preds, _ = evaluate_setting(df, setting=setting, model_name=args.model, seed=seed)
            pred_map[setting] = preds

        for setting in settings:
            sub = pred_map[setting][pred_map[setting]["sample_id"].isin(ambiguous_ids)].copy()
            metrics = subset_metrics(sub, setting=setting, model=args.model)
            metrics["seed"] = seed
            metrics["ambiguous_low"] = args.ambiguous_low
            metrics["ambiguous_high"] = args.ambiguous_high
            rows.append(metrics)
            sub["seed"] = seed
            predictions.append(sub)
            print(
                f"seed={seed} setting={setting} n={len(sub)} "
                f"pr_auc={metrics['pr_auc']:.4f} brier={metrics['brier']:.4f}"
            )

    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    metrics = pd.DataFrame(rows)
    metrics.to_csv(results / f"ambiguous_subset_metrics_by_seed{suffix}.csv", index=False)
    summary = summarize_metrics(metrics)
    extra = metrics.groupby("setting", sort=False).agg(
        n_subset_mean=("n_subset", "mean"),
        n_subset_std=("n_subset", "std"),
        positive_rate_subset_mean=("positive_rate_subset", "mean"),
        positive_rate_subset_std=("positive_rate_subset", "std"),
    ).reset_index()
    summary = summary.merge(extra, on="setting", how="left")
    summary.to_csv(results / f"ambiguous_subset_metrics{suffix}.csv", index=False)
    if predictions:
        pd.concat(predictions, ignore_index=True).to_csv(results / f"ambiguous_subset_predictions{suffix}.csv", index=False)

    print(summary[["setting", "n_subset_mean", "positive_rate_subset_mean", "pr_auc", "precision_at_10pct", "recall_at_top10pct", "brier"]])
    print(f"Saved ambiguous subset outputs to {results}")


if __name__ == "__main__":
    main()
