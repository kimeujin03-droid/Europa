#!/usr/bin/env python
"""Sensitivity checks on noise level and radiation_mimic simple_organic overlap.

Check A — Harsh noise (sigma=0.035, smoothing_window=9):
  Runs experiment1 + ambiguous subset under --noise-condition harsh.
  Expected: overall PR-AUC drops; ambiguous-subset full-vs-spectral gap widens.

Check B — High overlap (rad_simple_organic_hi=0.130, moderate noise):
  Increases radiation_mimic w_simple_organic upper bound from 0.090 → 0.130.
  Expected: w_simple_organic overlap rises toward 0.7–0.8; overall PR-AUC < 0.85.

Usage:
  python scripts/16_sensitivity_checks.py [--seeds 20] [--skip-baseline]
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

from src.config import ExperimentConfig, NOISE_PRESETS
from src.forward_model import generate_dataset
from src.modeling import (
    evaluate_setting,
    precision_at_k,
    threshold_for_top_k,
    top_k_class_counts,
)


SUMMARY_COLS = [
    "pr_auc",
    "precision_at_10pct",
    "recall_at_top10pct",
    "brier",
    "top10_radiation_mimic_count",
    "top10_exogenic_count",
    "top10_noise_count",
]

EXP1_SETTINGS = ["spectral_only", "full", "context_only"]
AMBIG_SETTINGS = ["spectral_only", "full", "context_only"]
AMBIG_LO, AMBIG_HI = 0.4, 0.7


# ------------------------------------------------------------------ helpers --

def summarize_metrics(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    out = []
    for setting, group in df.groupby("setting", sort=False):
        row = {"setting": setting, "n_seeds": len(group)}
        for col in SUMMARY_COLS:
            if col not in group.columns:
                continue
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=1)) if len(group) > 1 else 0.0
            row[col] = f"{row[f'{col}_mean']:.4f} +/- {row[f'{col}_std']:.4f}"
        out.append(row)
    return pd.DataFrame(out)


def run_exp1_seeds(args, cfg: ExperimentConfig) -> list[dict]:
    rows = []
    for i in range(args.seeds):
        seed = args.base_seed + i
        seed_cfg = ExperimentConfig(
            seed=seed,
            noise_condition=cfg.noise_condition,
            rad_simple_organic_hi=cfg.rad_simple_organic_hi,
        )
        df = generate_dataset(n=args.n, rho_geo=args.rho_geo, rho_rad=args.rho_rad, cfg=seed_cfg)
        for setting in EXP1_SETTINGS:
            m, _, _ = evaluate_setting(df, setting=setting, model_name=args.model, seed=seed)
            m["seed"] = seed
            rows.append(m)
            print(f"  seed={seed} setting={setting} pr_auc={m['pr_auc']:.4f}")
    return rows


def run_ambig_seeds(args, cfg: ExperimentConfig) -> list[dict]:
    rows = []
    for i in range(args.seeds):
        seed = args.base_seed + i
        seed_cfg = ExperimentConfig(
            seed=seed,
            noise_condition=cfg.noise_condition,
            rad_simple_organic_hi=cfg.rad_simple_organic_hi,
        )
        df = generate_dataset(n=args.n, rho_geo=args.rho_geo, rho_rad=args.rho_rad, cfg=seed_cfg)

        _, spec_preds, _ = evaluate_setting(df, setting="spectral_only", model_name=args.model, seed=seed)
        ambig_ids = set(spec_preds.loc[
            (spec_preds["score"] >= AMBIG_LO) & (spec_preds["score"] <= AMBIG_HI),
            "sample_id",
        ])
        if not ambig_ids:
            continue

        pred_map = {"spectral_only": spec_preds}
        for s in ["full", "context_only"]:
            _, p, _ = evaluate_setting(df, setting=s, model_name=args.model, seed=seed)
            pred_map[s] = p

        for setting in AMBIG_SETTINGS:
            sub = pred_map[setting][pred_map[setting]["sample_id"].isin(ambig_ids)].copy()
            y = sub["y"].to_numpy()
            score = sub["score"].to_numpy()
            if len(np.unique(y)) < 2:
                continue
            thresh = threshold_for_top_k(score, 0.1)
            y_top = (score >= thresh).astype(int)
            m = {
                "setting": setting,
                "seed": seed,
                "n_subset": len(sub),
                "pr_auc": float(average_precision_score(y, score)),
                "precision_at_10pct": precision_at_k(y, score, 0.1),
                "recall_at_top10pct": float(recall_score(y, y_top, zero_division=0)),
                "brier": float(brier_score_loss(y, score)),
            }
            m.update(top_k_class_counts(sub, score, 0.1))
            rows.append(m)
        print(f"  seed={seed} ambig_n={len(ambig_ids)}")
    return rows


def compute_overlap(df: pd.DataFrame, cls_a: str, cls_b: str, col: str, bins: int = 50) -> float:
    x = df.loc[df["z"] == cls_a, col].to_numpy(dtype=float)
    y = df.loc[df["z"] == cls_b, col].to_numpy(dtype=float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    if hi <= lo:
        return 1.0
    hx, edges = np.histogram(x, bins=bins, range=(lo, hi), density=True)
    hy, _ = np.histogram(y, bins=edges, density=True)
    return float(np.minimum(hx, hy).sum() * (edges[1] - edges[0]))


def extract_pr_auc(summary: pd.DataFrame, setting: str) -> str:
    r = summary[summary["setting"] == setting]
    if len(r) == 0:
        return "n/a"
    return str(r.iloc[0].get("pr_auc", "n/a"))


# ------------------------------------------------------------------ main ----

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8000)
    parser.add_argument("--rho-geo", type=float, default=0.75)
    parser.add_argument("--rho-rad", type=float, default=0.75)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=3421)
    parser.add_argument("--model", default="logreg")
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Load baseline CSVs instead of re-running (faster).",
    )
    args = parser.parse_args()

    results = ROOT / "results"
    results.mkdir(exist_ok=True)

    # ---- Baseline --------------------------------------------------------
    baseline_exp1_path = results / "experiment1_metrics_leakage_fix_v1_logreg.csv"
    baseline_ambig_path = results / "ambiguous_subset_metrics_leakage_fix_v1_logreg.csv"

    if args.skip_baseline and baseline_exp1_path.exists():
        print("[Baseline] Loading from existing CSVs.")
        baseline_exp1 = pd.read_csv(baseline_exp1_path)
        baseline_ambig = pd.read_csv(baseline_ambig_path) if baseline_ambig_path.exists() else pd.DataFrame()
    else:
        print("[Baseline] Running moderate noise, rad_so_hi=0.090 …")
        cfg_base = ExperimentConfig(noise_condition="moderate", rad_simple_organic_hi=0.090)
        baseline_exp1 = summarize_metrics(run_exp1_seeds(args, cfg_base))
        baseline_ambig = summarize_metrics(run_ambig_seeds(args, cfg_base))

    # ---- Check A: harsh noise -------------------------------------------
    print("\n[Check A] Harsh noise (sigma=0.035, smoothing=9) …")
    cfg_harsh = ExperimentConfig(noise_condition="harsh", rad_simple_organic_hi=0.090)
    harsh_exp1_rows = run_exp1_seeds(args, cfg_harsh)
    harsh_ambig_rows = run_ambig_seeds(args, cfg_harsh)
    harsh_exp1 = summarize_metrics(harsh_exp1_rows)
    harsh_ambig = summarize_metrics(harsh_ambig_rows)
    harsh_exp1.to_csv(results / "experiment1_metrics_harsh_noise_v1.csv", index=False)
    harsh_ambig.to_csv(results / "ambiguous_subset_metrics_harsh_noise_v1.csv", index=False)
    pd.DataFrame(harsh_exp1_rows).to_csv(results / "experiment1_metrics_by_seed_harsh_noise_v1.csv", index=False)
    pd.DataFrame(harsh_ambig_rows).to_csv(results / "ambiguous_subset_metrics_by_seed_harsh_noise_v1.csv", index=False)

    # ---- Check B: high overlap ------------------------------------------
    print("\n[Check B] High overlap (rad_so_hi=0.130, moderate noise) …")
    cfg_hiov = ExperimentConfig(noise_condition="moderate", rad_simple_organic_hi=0.130)
    hiov_exp1_rows = run_exp1_seeds(args, cfg_hiov)
    hiov_ambig_rows = run_ambig_seeds(args, cfg_hiov)
    hiov_exp1 = summarize_metrics(hiov_exp1_rows)
    hiov_ambig = summarize_metrics(hiov_ambig_rows)
    hiov_exp1.to_csv(results / "experiment1_metrics_high_overlap_v1.csv", index=False)
    hiov_ambig.to_csv(results / "ambiguous_subset_metrics_high_overlap_v1.csv", index=False)
    pd.DataFrame(hiov_exp1_rows).to_csv(results / "experiment1_metrics_by_seed_high_overlap_v1.csv", index=False)
    pd.DataFrame(hiov_ambig_rows).to_csv(results / "ambiguous_subset_metrics_by_seed_high_overlap_v1.csv", index=False)

    # Measure actual overlap for Check B on a fresh 4000-sample dataset
    df_qc = generate_dataset(n=4000, rho_geo=0.75, rho_rad=0.75, cfg=ExperimentConfig(
        seed=9999, noise_condition="moderate", rad_simple_organic_hi=0.130
    ))
    ov_new = compute_overlap(df_qc, "ocean_organic", "radiation_mimic", "w_simple_organic")
    ov_old = 0.5685  # from previous benchmark

    # ---- Combined comparison summary ------------------------------------
    print("\n" + "=" * 72)
    print("SENSITIVITY CHECK SUMMARY")
    print("=" * 72)
    header = f"{'Condition':<38} {'Exp1 spec':>10} {'Exp1 full':>10} {'Ambig spec':>11} {'Ambig full':>10}"
    print(header)
    print("-" * 72)

    for label, exp1_df, ambig_df in [
        (f"Baseline (mod noise, ov={ov_old:.4f})", baseline_exp1, baseline_ambig),
        (f"A: harsh noise (sigma=0.035, ov={ov_old:.4f})", harsh_exp1, harsh_ambig),
        (f"B: high overlap (mod noise, ov={ov_new:.4f})", hiov_exp1, hiov_ambig),
    ]:
        e_spec = extract_pr_auc(exp1_df, "spectral_only")
        e_full = extract_pr_auc(exp1_df, "full")
        a_spec = extract_pr_auc(ambig_df, "spectral_only") if len(ambig_df) else "n/a"
        a_full = extract_pr_auc(ambig_df, "full") if len(ambig_df) else "n/a"
        print(f"  {label:<36} {e_spec:>10} {e_full:>10} {a_spec:>11} {a_full:>10}")

    print("=" * 72)
    print(f"\nCheck B new overlap coefficient: {ov_new:.4f}  (was {ov_old:.4f})")
    print(f"Results written to {results}/")


if __name__ == "__main__":
    main()
