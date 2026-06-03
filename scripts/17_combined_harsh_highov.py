#!/usr/bin/env python
"""Combined condition: harsh noise (sigma=0.035) + high organic overlap (rad_so_hi=0.130).

This is the primary physically-motivated benchmark condition:
  - Harsh noise approximates actual NIMS SNR limits
  - High radiation_mimic simple_organic overlap removes the trivially-detectable
    organic-presence shortcut, forcing the model to rely on ambiguous spectral shape

Outputs:
  results/experiment1_metrics_harsh_highov_v1.csv
  results/experiment1_metrics_by_seed_harsh_highov_v1.csv
  results/experiment1_predictions_harsh_highov_v1.csv
  results/ambiguous_subset_metrics_harsh_highov_v1.csv
  results/ambiguous_subset_metrics_by_seed_harsh_highov_v1.csv
  results/qc/class_score_stats_harsh_highov_v1.csv
  results/qc/class_score_overlap_harsh_highov_v1.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, recall_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ExperimentConfig
from src.forward_model import generate_dataset
from src.modeling import (
    evaluate_setting,
    precision_at_k,
    threshold_for_top_k,
    top_k_class_counts,
)

TAG = "harsh_highov_v1"
AMBIG_LO, AMBIG_HI = 0.4, 0.7
EXP1_SETTINGS = ["spectral_only", "full", "context_only"]
CLASS_ORDER = [
    "ocean_organic",
    "ocean_nonorganic",
    "radiation_mimic",
    "exogenic_complex_organic",
    "noise_artifact",
]
SUMMARY_COLS = [
    "pr_auc",
    "precision_at_10pct",
    "recall_at_top10pct",
    "brier",
    "top10_radiation_mimic_count",
    "top10_exogenic_count",
    "top10_noise_count",
]


def make_cfg(seed: int, rad_so_hi: float) -> ExperimentConfig:
    return ExperimentConfig(
        seed=seed,
        noise_condition="harsh",
        rad_simple_organic_hi=rad_so_hi,
    )


def summarize(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    out = []
    for setting, grp in df.groupby("setting", sort=False):
        row = {"setting": setting, "n_seeds": len(grp)}
        for col in SUMMARY_COLS:
            if col not in grp.columns:
                continue
            row[f"{col}_mean"] = float(grp[col].mean())
            row[f"{col}_std"] = float(grp[col].std(ddof=1)) if len(grp) > 1 else 0.0
            row[col] = f"{row[f'{col}_mean']:.4f} +/- {row[f'{col}_std']:.4f}"
        out.append(row)
    return pd.DataFrame(out)


def overlap_coeff(x: np.ndarray, y: np.ndarray, bins: int = 50) -> float:
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if not len(x) or not len(y):
        return np.nan
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    if hi <= lo:
        return 1.0
    hx, edges = np.histogram(x, bins=bins, range=(lo, hi), density=True)
    hy, _ = np.histogram(y, bins=edges, density=True)
    return float(np.minimum(hx, hy).sum() * (edges[1] - edges[0]))


def class_confusion(preds: pd.DataFrame, setting: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = preds[preds["setting"] == setting].copy()
    k = max(1, int(len(sub) * 0.10))
    thresh = float(np.sort(sub["score"].to_numpy())[::-1][k - 1])
    target_scores = sub[sub["z"] == "ocean_organic"]["score"].to_numpy()

    stat_rows, ov_rows = [], []
    for cls in CLASS_ORDER:
        g = sub[sub["z"] == cls]
        if not len(g):
            continue
        s = g["score"].to_numpy()
        stat_rows.append({
            "setting": setting, "class": cls, "n": len(g),
            "score_mean": float(np.mean(s)),
            "score_std": float(np.std(s, ddof=1)),
            "score_median": float(np.median(s)),
            "score_p90": float(np.percentile(s, 90)),
            "frac_in_top10pct": float(np.mean(s >= thresh)),
        })
        if cls != "ocean_organic":
            ov_rows.append({
                "setting": setting, "class_vs": cls,
                "mean_target": float(np.mean(target_scores)),
                "mean_other": float(np.mean(s)),
                "mean_delta": float(np.mean(target_scores)) - float(np.mean(s)),
                "score_overlap_vs_ocean_organic": overlap_coeff(target_scores, s),
            })

    return pd.DataFrame(stat_rows), pd.DataFrame(ov_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8000)
    parser.add_argument("--rho-geo", type=float, default=0.75)
    parser.add_argument("--rho-rad", type=float, default=0.75)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=3421)
    parser.add_argument("--model", default="logreg")
    parser.add_argument("--rad-so-hi", type=float, default=0.130,
                        help="Upper bound for radiation_mimic raw simple_organic weight.")
    args = parser.parse_args()

    results = ROOT / "results"
    qc_dir = results / "qc"
    results.mkdir(exist_ok=True)
    qc_dir.mkdir(exist_ok=True)

    # Measure mixture overlap on a QC dataset before running experiments
    cfg_qc = make_cfg(seed=9999, rad_so_hi=args.rad_so_hi)
    df_qc = generate_dataset(n=4000, rho_geo=args.rho_geo, rho_rad=args.rho_rad, cfg=cfg_qc)
    ov_so = overlap_coeff(
        df_qc.loc[df_qc["z"] == "ocean_organic", "w_simple_organic"].to_numpy(),
        df_qc.loc[df_qc["z"] == "radiation_mimic", "w_simple_organic"].to_numpy(),
    )
    print(f"[QC] w_simple_organic overlap (ocean_organic vs radiation_mimic): {ov_so:.4f}")
    print(f"[QC] noise: harsh (sigma=0.035, smoothing=9), rad_so_hi={args.rad_so_hi}")

    # Experiment 1
    print(f"\n[Exp1] {args.seeds} seeds x {len(EXP1_SETTINGS)} settings ...")
    exp1_rows = []
    first_preds = []
    for i in range(args.seeds):
        seed = args.base_seed + i
        cfg = make_cfg(seed=seed, rad_so_hi=args.rad_so_hi)
        df = generate_dataset(n=args.n, rho_geo=args.rho_geo, rho_rad=args.rho_rad, cfg=cfg)
        for setting in EXP1_SETTINGS:
            m, preds, _ = evaluate_setting(df, setting=setting, model_name=args.model, seed=seed)
            m["seed"] = seed
            exp1_rows.append(m)
            if i == 0:
                preds["seed"] = seed
                first_preds.append(preds)
            print(f"  seed={seed} {setting} pr_auc={m['pr_auc']:.4f} top10_rad={m['top10_radiation_mimic_count']}")

    exp1_df = pd.DataFrame(exp1_rows)
    exp1_summary = summarize(exp1_rows)
    exp1_summary.to_csv(results / f"experiment1_metrics_{TAG}.csv", index=False)
    exp1_df.to_csv(results / f"experiment1_metrics_by_seed_{TAG}.csv", index=False)
    if first_preds:
        first_preds_df = pd.concat(first_preds, ignore_index=True)
        first_preds_df.to_csv(results / f"experiment1_predictions_{TAG}.csv", index=False)

    # Ambiguous subset
    print(f"\n[Ambig] {args.seeds} seeds ...")
    ambig_rows = []
    for i in range(args.seeds):
        seed = args.base_seed + i
        cfg = make_cfg(seed=seed, rad_so_hi=args.rad_so_hi)
        df = generate_dataset(n=args.n, rho_geo=args.rho_geo, rho_rad=args.rho_rad, cfg=cfg)

        _, spec_preds, _ = evaluate_setting(df, "spectral_only", model_name=args.model, seed=seed)
        ambig_ids = set(spec_preds.loc[
            (spec_preds["score"] >= AMBIG_LO) & (spec_preds["score"] <= AMBIG_HI),
            "sample_id",
        ])
        if not ambig_ids:
            print(f"  seed={seed} ambiguous subset empty, skipping")
            continue

        pred_map = {"spectral_only": spec_preds}
        for s in ["full", "context_only"]:
            _, p, _ = evaluate_setting(df, s, model_name=args.model, seed=seed)
            pred_map[s] = p

        for setting in EXP1_SETTINGS:
            sub = pred_map[setting][pred_map[setting]["sample_id"].isin(ambig_ids)].copy()
            y, score = sub["y"].to_numpy(), sub["score"].to_numpy()
            if len(np.unique(y)) < 2:
                continue
            thresh = threshold_for_top_k(score, 0.1)
            y_top = (score >= thresh).astype(int)
            m = {
                "setting": setting, "seed": seed, "n_subset": len(sub),
                "pr_auc": float(average_precision_score(y, score)),
                "precision_at_10pct": precision_at_k(y, score, 0.1),
                "recall_at_top10pct": float(recall_score(y, y_top, zero_division=0)),
                "brier": float(brier_score_loss(y, score)),
            }
            m.update(top_k_class_counts(sub, score, 0.1))
            ambig_rows.append(m)

        print(f"  seed={seed} ambig_n={len(ambig_ids)}")

    ambig_summary = summarize(ambig_rows)
    ambig_summary.to_csv(results / f"ambiguous_subset_metrics_{TAG}.csv", index=False)
    pd.DataFrame(ambig_rows).to_csv(results / f"ambiguous_subset_metrics_by_seed_{TAG}.csv", index=False)

    # Class confusion analysis on first-seed predictions
    if first_preds:
        all_stats, all_ovs = [], []
        for setting in ["spectral_only", "full"]:
            stats, ovs = class_confusion(first_preds_df, setting)
            all_stats.append(stats)
            all_ovs.append(ovs)
        pd.concat(all_stats).to_csv(qc_dir / f"class_score_stats_{TAG}.csv", index=False)
        pd.concat(all_ovs).to_csv(qc_dir / f"class_score_overlap_{TAG}.csv", index=False)

    # Print summary
    def pr(df: pd.DataFrame, s: str, col: str = "pr_auc") -> str:
        r = df[df["setting"] == s]
        return str(r.iloc[0][col]) if len(r) else "n/a"

    BASELINE_SPEC = "0.8951 +/- 0.0104"
    BASELINE_FULL = "0.9299 +/- 0.0075"
    BASELINE_AMBIG_SPEC = "0.3090 +/- 0.0417"
    BASELINE_AMBIG_FULL = "0.5644 +/- 0.0786"
    HARSH_SPEC = "0.7691 +/- 0.0186"
    HARSH_FULL = "0.8783 +/- 0.0118"
    HARSH_AMBIG_SPEC = "0.3258 +/- 0.0400"
    HARSH_AMBIG_FULL = "0.6579 +/- 0.0559"

    print("\n" + "=" * 72)
    print("COMBINED CONDITION SUMMARY  (harsh noise + high organic overlap)")
    print("=" * 72)
    print(f"  w_simple_organic overlap (ocean_organic vs radiation_mimic): {ov_so:.4f}")
    print(f"  noise: sigma=0.035, smoothing_window=9")
    print()
    header = f"  {'Condition':<44} {'spec PR-AUC':>14} {'full PR-AUC':>14}"
    print(header)
    print("  " + "-" * 72)
    print(f"  {'Baseline (moderate, ov=0.57)':<44} {BASELINE_SPEC:>14} {BASELINE_FULL:>14}")
    print(f"  {'Check A: harsh only (ov=0.57)':<44} {HARSH_SPEC:>14} {HARSH_FULL:>14}")
    print(f"  {'This run: harsh + high ov':<44} {pr(exp1_summary,'spectral_only'):>14} {pr(exp1_summary,'full'):>14}")
    print()
    print("  Ambiguous subset:")
    print(f"  {'Baseline':<44} {BASELINE_AMBIG_SPEC:>14} {BASELINE_AMBIG_FULL:>14}")
    print(f"  {'Check A: harsh only':<44} {HARSH_AMBIG_SPEC:>14} {HARSH_AMBIG_FULL:>14}")
    print(f"  {'This run: harsh + high ov':<44} {pr(ambig_summary,'spectral_only'):>14} {pr(ambig_summary,'full'):>14}")
    print("=" * 72)

    # Class confusion quick-print (spectral_only, seed 0)
    if first_preds:
        spec_stats, _ = class_confusion(first_preds_df, "spectral_only")
        print("\n  Class confusion (spectral_only, seed 0):")
        print("  " + spec_stats[
            ["class", "n", "score_mean", "score_p90", "frac_in_top10pct"]
        ].to_string(index=False, col_space=8))

    print(f"\n  Outputs saved to {results}/")


if __name__ == "__main__":
    main()
