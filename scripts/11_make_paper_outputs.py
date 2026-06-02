#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER = RESULTS / "paper"
PAPER.mkdir(parents=True, exist_ok=True)

LABELS = {
    "spectral_only": "Spectral-only",
    "spectral_geology": "Spectral + geology",
    "spectral_radiation": "Spectral + radiation",
    "full": "Full spatial-spectral",
    "context_only": "Context-only baseline",
}

SUMMARY_COLS = [
    "pr_auc",
    "roc_auc",
    "precision_at_10pct",
    "recall_at_top10pct",
    "brier",
    "top10_radiation_mimic_count",
    "top10_exogenic_count",
]


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for setting, group in metrics.groupby("setting", sort=False):
        row = {"setting": setting, "model": group["model"].iloc[0], "n_seeds": len(group)}
        for col in SUMMARY_COLS:
            mean = float(group[col].mean())
            std = float(group[col].std(ddof=1)) if len(group) > 1 else 0.0
            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std
            row[col] = f"{mean:.4f} +/- {std:.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


def combine_rf_chunks() -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = [RESULTS / f"experiment1_metrics_by_seed_rf_chunk{i}.csv" for i in range(1, 5)]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing RF chunk files: {missing}")
    by_seed = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    summary = summarize_metrics(by_seed)
    summary.to_csv(RESULTS / "experiment1_metrics_rf_20seed.csv", index=False)
    by_seed.to_csv(RESULTS / "experiment1_metrics_by_seed_rf_20seed.csv", index=False)
    return summary, by_seed


def paper_table(logreg_summary: pd.DataFrame, rf_summary: pd.DataFrame) -> None:
    rows = []
    for model_label, summary in [("Additive logistic regression", logreg_summary), ("Random forest robustness", rf_summary)]:
        context_pr = float(summary.loc[summary["setting"] == "context_only", "pr_auc_mean"].iloc[0])
        for _, row in summary.iterrows():
            rows.append(
                {
                    "Model family": model_label,
                    "Input setting": LABELS.get(row["setting"], row["setting"]),
                    "PR-AUC": row["pr_auc"],
                    "ROC-AUC": row["roc_auc"],
                    "Precision@10%": row["precision_at_10pct"],
                    "Recall@10%": row["recall_at_top10pct"],
                    "Brier": row["brier"],
                    "Rad mimic in Top10": row["top10_radiation_mimic_count"],
                    "Exogenic in Top10": row["top10_exogenic_count"],
                    "PR-AUC gap vs context-only": (
                        f"{row['pr_auc_mean'] - context_pr:.4f}" if row["setting"] != "context_only" else "0.0000"
                    ),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(PAPER / "experiment1_metric_table.csv", index=False, encoding="utf-8-sig")


def representative_pr_curve() -> None:
    preds_path = RESULTS / "experiment1_predictions.csv"
    if not preds_path.exists():
        return
    preds = pd.read_csv(preds_path)
    style = {
        "spectral_only": {"color": "#1f77b4", "lw": 2.2, "ls": "-"},
        "spectral_geology": {"color": "#2ca02c", "lw": 2.2, "ls": "-"},
        "spectral_radiation": {"color": "#ff7f0e", "lw": 2.2, "ls": "-"},
        "full": {"color": "#d62728", "lw": 2.6, "ls": "-"},
        "context_only": {"color": "#777777", "lw": 1.2, "ls": "--"},
    }
    plt.figure(figsize=(6.5, 4.5))
    for setting, group in preds.groupby("setting", sort=False):
        precision, recall, _ = precision_recall_curve(group["y"], group["score"])
        kwargs = style.get(setting, {})
        plt.plot(recall, precision, label=LABELS.get(setting, setting), **kwargs)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Representative precision-recall curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(PAPER / "experiment1_pr_curve_logreg_representative.png", dpi=250)
    plt.close()


def same_spectrum_table() -> None:
    rows = []
    for tag in ["logreg_ambiguous", "rf_ambiguous"]:
        path = RESULTS / f"same_spectrum_paired_stats_{tag}.csv"
        if path.exists():
            rows.append(pd.read_csv(path).iloc[0].to_dict())
    if rows:
        pd.DataFrame(rows).to_csv(PAPER / "same_spectrum_ambiguous_summary.csv", index=False)


def main() -> None:
    rf_summary, _rf_by_seed = combine_rf_chunks()
    logreg_summary = pd.read_csv(RESULTS / "experiment1_metrics.csv")
    paper_table(logreg_summary, rf_summary)
    representative_pr_curve()
    same_spectrum_table()
    print(f"Saved paper outputs to {PAPER}")
    print("\nRF 20-seed summary:")
    print(rf_summary[["setting", "pr_auc", "precision_at_10pct", "recall_at_top10pct", "brier"]].to_string(index=False))


if __name__ == "__main__":
    main()
