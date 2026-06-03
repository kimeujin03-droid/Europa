#!/usr/bin/env python
"""Analyse per-class score distributions from saved prediction CSVs.

Answers:
  - Which classes does ocean_organic get confused with?
  - Is radiation_mimic the primary confuser, or do noise / exogenic also leak?
  - How does full vs spectral_only change the confusion picture?
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


CLASS_ORDER = [
    "ocean_organic",
    "ocean_nonorganic",
    "radiation_mimic",
    "exogenic_complex_organic",
    "noise_artifact",
]

SETTINGS_OF_INTEREST = ["spectral_only", "full"]


def overlap_coefficient(x: np.ndarray, y: np.ndarray, bins: int = 50) -> float:
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    if hi <= lo:
        return 1.0
    hx, edges = np.histogram(x, bins=bins, range=(lo, hi), density=True)
    hy, _ = np.histogram(y, bins=edges, density=True)
    width = edges[1] - edges[0]
    return float(np.minimum(hx, hy).sum() * width)


def score_stats_by_class(preds: pd.DataFrame, setting: str) -> pd.DataFrame:
    sub = preds[preds["setting"] == setting].copy()
    rows = []
    k_frac = 0.10
    k = max(1, int(len(sub) * k_frac))
    threshold = float(np.sort(sub["score"].to_numpy())[::-1][k - 1])

    for cls in CLASS_ORDER:
        g = sub[sub["z"] == cls]
        if len(g) == 0:
            continue
        s = g["score"].to_numpy()
        frac_in_top10 = float(np.mean(s >= threshold))
        rows.append({
            "setting": setting,
            "class": cls,
            "n": len(g),
            "score_mean": float(np.mean(s)),
            "score_std": float(np.std(s, ddof=1)),
            "score_median": float(np.median(s)),
            "score_p90": float(np.percentile(s, 90)),
            "frac_in_top10pct": frac_in_top10,
        })
    return pd.DataFrame(rows)


def pairwise_overlap_vs_target(preds: pd.DataFrame, setting: str, target: str = "ocean_organic") -> pd.DataFrame:
    sub = preds[preds["setting"] == setting]
    target_scores = sub[sub["z"] == target]["score"].to_numpy()
    rows = []
    for cls in CLASS_ORDER:
        if cls == target:
            continue
        other_scores = sub[sub["z"] == cls]["score"].to_numpy()
        if len(other_scores) == 0:
            continue
        ov = overlap_coefficient(target_scores, other_scores)
        rows.append({
            "setting": setting,
            "class_vs": cls,
            f"score_overlap_vs_{target}": ov,
            "mean_other": float(np.mean(other_scores)),
            "mean_target": float(np.mean(target_scores)),
            "mean_delta": float(np.mean(target_scores)) - float(np.mean(other_scores)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default="results/experiment1_predictions_leakage_fix_v1_logreg.csv",
        help="Path to predictions CSV (relative to project root).",
    )
    parser.add_argument("--out-dir", default="results/qc")
    args = parser.parse_args()

    pred_path = ROOT / args.predictions
    if not pred_path.exists():
        print(f"[ERROR] Predictions file not found: {pred_path}")
        print("Run scripts/02_run_experiment1.py first to generate predictions.")
        sys.exit(1)

    preds = pd.read_csv(pred_path)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []
    all_overlaps = []

    for setting in SETTINGS_OF_INTEREST:
        if setting not in preds["setting"].unique():
            print(f"[WARN] Setting '{setting}' not in predictions file, skipping.")
            continue

        stats = score_stats_by_class(preds, setting)
        overlaps = pairwise_overlap_vs_target(preds, setting, target="ocean_organic")
        all_stats.append(stats)
        all_overlaps.append(overlaps)

        print(f"\n=== {setting.upper()} -- Score distribution by class ===")
        print(stats[["class", "n", "score_mean", "score_std", "score_median", "score_p90", "frac_in_top10pct"]].to_string(index=False))

        print(f"\n=== {setting.upper()} -- Score overlap vs ocean_organic ===")
        print(overlaps[["class_vs", "mean_target", "mean_other", "mean_delta", "score_overlap_vs_ocean_organic"]].to_string(index=False))

    if all_stats:
        stats_df = pd.concat(all_stats, ignore_index=True)
        stats_df.to_csv(out_dir / "class_score_stats.csv", index=False)

    if all_overlaps:
        overlap_df = pd.concat(all_overlaps, ignore_index=True)
        overlap_df.to_csv(out_dir / "class_score_overlap_vs_ocean_organic.csv", index=False)

    # Side-by-side spectral_only vs full confusion summary
    if len(all_stats) == 2:
        merged = pd.merge(
            all_stats[0][["class", "score_mean", "frac_in_top10pct"]].rename(
                columns={"score_mean": "spec_mean", "frac_in_top10pct": "spec_top10"}
            ),
            all_stats[1][["class", "score_mean", "frac_in_top10pct"]].rename(
                columns={"score_mean": "full_mean", "frac_in_top10pct": "full_top10"}
            ),
            on="class",
        )
        merged["delta_top10"] = merged["full_top10"] - merged["spec_top10"]
        print("\n=== SPECTRAL-ONLY vs FULL -- top-10% occupancy comparison ===")
        print(merged.to_string(index=False))
        merged.to_csv(out_dir / "class_top10_comparison.csv", index=False)

    print(f"\nSaved confusion QC files to {out_dir}")


if __name__ == "__main__":
    main()
