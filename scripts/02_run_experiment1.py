#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ExperimentConfig, NOISE_PRESETS
from src.forward_model import generate_dataset
from src.modeling import evaluate_setting, pr_curve_data
from src.plotting import save_pr_plot


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8000)
    parser.add_argument("--rho-geo", type=float, default=0.75)
    parser.add_argument("--rho-rad", type=float, default=0.75)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=3421)
    parser.add_argument("--noise-condition", choices=NOISE_PRESETS.keys(), default="moderate")
    parser.add_argument("--smoothing-window", type=int, default=None)
    parser.add_argument("--model", choices=["rf", "hgb", "logreg"], default="rf")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    settings = ["spectral_only", "spectral_geology", "spectral_radiation", "full", "context_only"]
    metrics_rows = []
    curves = {}
    all_preds = []

    seed_values = [args.base_seed + i for i in range(args.seeds)]
    for run_idx, seed in enumerate(seed_values):
        cfg = ExperimentConfig(
            seed=seed,
            noise_condition=args.noise_condition,
            smoothing_window=args.smoothing_window,
        )
        df = generate_dataset(n=args.n, rho_geo=args.rho_geo, rho_rad=args.rho_rad, cfg=cfg)

        for setting in settings:
            metrics, preds, _ = evaluate_setting(df, setting=setting, model_name=args.model, seed=seed)
            metrics["seed"] = seed
            metrics["noise_condition"] = args.noise_condition
            metrics["smoothing_window"] = cfg.effective_smoothing_window
            metrics_rows.append(metrics)
            if run_idx == 0:
                preds["seed"] = seed
                all_preds.append(preds)
                curves[setting] = pr_curve_data(preds["y"].to_numpy(), preds["score"].to_numpy())
            print(f"seed={seed} setting={setting} pr_auc={metrics['pr_auc']:.4f} top10_rad={metrics['top10_radiation_mimic_count']}")

    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    metrics_df = pd.DataFrame(metrics_rows)
    summary = summarize_metrics(metrics_df)
    summary.to_csv(results / f"experiment1_metrics{suffix}.csv", index=False)
    metrics_df.to_csv(results / f"experiment1_metrics_by_seed{suffix}.csv", index=False)
    pd.concat(all_preds, ignore_index=True).to_csv(results / f"experiment1_predictions{suffix}.csv", index=False)
    save_pr_plot(curves, str(results / f"experiment1_pr_curve{suffix}.png"))
    print(summary[["setting", "pr_auc", "precision_at_10pct", "recall_at_top10pct", "brier", "top10_radiation_mimic_count", "top10_exogenic_count"]])
    print(f"Saved outputs to {results}")


if __name__ == "__main__":
    main()
