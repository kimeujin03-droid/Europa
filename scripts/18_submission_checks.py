#!/usr/bin/env python
"""Three pre-submission checks on the main condition (harsh noise + high overlap).

Check 1 -- rho=0 corner:
  rho_geo=0, rho_rad=0 should collapse full vs spectral-only to near-zero delta,
  demonstrating that the benefit of spatial context is not leaked by construction.

Check 2 -- ambiguous threshold robustness:
  Three threshold windows (0.3-0.8, 0.4-0.7, 0.45-0.65) should all show the same
  directional result (full >> spectral-only >> context-only in the ambiguous subset).

Check 3 -- w_simple_organic overlap QC figure:
  KDE overlay of ocean_organic vs radiation_mimic w_simple_organic distributions
  under baseline (ov~0.57) and main condition (ov~0.83), saved as a publication figure.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, brier_score_loss, recall_score
from scipy.stats import gaussian_kde

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

# Main condition
NOISE = "harsh"
RAD_SO_HI_MAIN = 0.130   # high overlap
RAD_SO_HI_BASE = 0.090   # baseline overlap
N = 8000
BASE_SEED = 3421
SETTINGS = ["spectral_only", "full", "context_only"]


def make_cfg(seed: int, rho_geo: float, rho_rad: float, rad_so_hi: float) -> ExperimentConfig:
    return ExperimentConfig(seed=seed, noise_condition=NOISE, rad_simple_organic_hi=rad_so_hi)


def overlap_coeff(x: np.ndarray, y: np.ndarray, bins: int = 60) -> float:
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if not len(x) or not len(y):
        return np.nan
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    if hi <= lo:
        return 1.0
    hx, edges = np.histogram(x, bins=bins, range=(lo, hi), density=True)
    hy, _ = np.histogram(y, bins=edges, density=True)
    return float(np.minimum(hx, hy).sum() * (edges[1] - edges[0]))


def ambig_metrics(preds_map: dict, ambig_ids: set, setting: str) -> dict:
    sub = preds_map[setting][preds_map[setting]["sample_id"].isin(ambig_ids)].copy()
    y, score = sub["y"].to_numpy(), sub["score"].to_numpy()
    if len(np.unique(y)) < 2 or len(y) == 0:
        return {"setting": setting, "n_subset": len(sub), "pr_auc": np.nan,
                "precision_at_10pct": np.nan, "brier": np.nan}
    thresh = threshold_for_top_k(score, 0.1)
    y_top = (score >= thresh).astype(int)
    return {
        "setting": setting,
        "n_subset": len(sub),
        "pr_auc": float(average_precision_score(y, score)),
        "precision_at_10pct": precision_at_k(y, score, 0.1),
        "recall_at_top10pct": float(recall_score(y, y_top, zero_division=0)),
        "brier": float(brier_score_loss(y, score)),
    }


def run_seeds(seeds: list[int], rho_geo: float, rho_rad: float, rad_so_hi: float) -> list[dict]:
    rows = []
    for seed in seeds:
        cfg = ExperimentConfig(seed=seed, noise_condition=NOISE, rad_simple_organic_hi=rad_so_hi)
        df = generate_dataset(n=N, rho_geo=rho_geo, rho_rad=rho_rad, cfg=cfg)
        for setting in SETTINGS:
            m, _, _ = evaluate_setting(df, setting=setting, model_name="logreg", seed=seed)
            m["seed"] = seed
            m["rho_geo"] = rho_geo
            m["rho_rad"] = rho_rad
            rows.append(m)
    return rows


def summarize_rows(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    out = []
    for setting, grp in df.groupby("setting", sort=False):
        row = {"setting": setting, "n_seeds": len(grp)}
        for col in ["pr_auc", "brier", "precision_at_10pct"]:
            if col not in grp:
                continue
            row[f"{col}_mean"] = float(grp[col].mean())
            row[f"{col}_std"] = float(grp[col].std(ddof=1)) if len(grp) > 1 else 0.0
            row[col] = f"{row[f'{col}_mean']:.4f} +/- {row[f'{col}_std']:.4f}"
        out.append(row)
    return pd.DataFrame(out)


# ------------------------------------------------------------------ main ---

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()

    seeds = [BASE_SEED + i for i in range(args.seeds)]
    results = ROOT / "results"
    qc_dir = results / "qc"
    results.mkdir(exist_ok=True)
    qc_dir.mkdir(exist_ok=True)

    # ================================================================
    # Check 1 -- rho=0 corner
    # ================================================================
    print("=" * 64)
    print("CHECK 1 -- rho=0 corner (harsh + high overlap)")
    print("=" * 64)
    rho0_rows = run_seeds(seeds, rho_geo=0.0, rho_rad=0.0, rad_so_hi=RAD_SO_HI_MAIN)
    rho0_summary = summarize_rows(rho0_rows)
    rho0_summary.to_csv(results / "rho0_metrics_harsh_highov_v1.csv", index=False)

    # Also need the main-condition (rho=0.75) summary for comparison.
    # Load from the previous run's by-seed CSV rather than rerunning.
    main_by_seed_path = results / "experiment1_metrics_by_seed_harsh_highov_v1.csv"
    if main_by_seed_path.exists():
        main_summary = summarize_rows(pd.read_csv(main_by_seed_path).to_dict("records"))
    else:
        print("  [WARN] Main condition CSV not found; re-running rho=0.75 ...")
        main_rows = run_seeds(seeds, rho_geo=0.75, rho_rad=0.75, rad_so_hi=RAD_SO_HI_MAIN)
        main_summary = summarize_rows(main_rows)

    def pr(df: pd.DataFrame, s: str) -> str:
        r = df[df["setting"] == s]
        return str(r.iloc[0]["pr_auc"]) if len(r) else "n/a"

    print(f"\n  {'Setting':<22} {'rho=0.75 PR-AUC':>22} {'rho=0 PR-AUC':>20} {'delta':>10}")
    print("  " + "-" * 78)
    for s in SETTINGS:
        main_val = pr(main_summary, s)
        rho0_val = pr(rho0_summary, s)
        try:
            delta = f"{float(rho0_val.split()[0]) - float(main_val.split()[0]):+.4f}"
        except Exception:
            delta = "n/a"
        print(f"  {s:<22} {main_val:>22} {rho0_val:>20} {delta:>10}")

    spec_rho0 = float(pr(rho0_summary, "spectral_only").split()[0])
    full_rho0  = float(pr(rho0_summary, "full").split()[0])
    print(f"\n  rho=0 full vs spectral-only delta: {full_rho0 - spec_rho0:+.4f}")
    spec_main = float(pr(main_summary, "spectral_only").split()[0])
    full_main  = float(pr(main_summary, "full").split()[0])
    print(f"  rho=0.75 full vs spectral-only delta: {full_main - spec_main:+.4f}")

    # ================================================================
    # Check 2 -- ambiguous threshold robustness
    # ================================================================
    print("\n" + "=" * 64)
    print("CHECK 2 -- Ambiguous threshold robustness (harsh + high overlap)")
    print("=" * 64)

    thresholds = [(0.30, 0.80), (0.40, 0.70), (0.45, 0.65)]
    thresh_rows = []

    for lo, hi in thresholds:
        setting_rows = {s: [] for s in SETTINGS}
        for seed in seeds:
            cfg = ExperimentConfig(seed=seed, noise_condition=NOISE, rad_simple_organic_hi=RAD_SO_HI_MAIN)
            df = generate_dataset(n=N, rho_geo=0.75, rho_rad=0.75, cfg=cfg)

            _, spec_preds, _ = evaluate_setting(df, "spectral_only", model_name="logreg", seed=seed)
            ambig_ids = set(spec_preds.loc[
                (spec_preds["score"] >= lo) & (spec_preds["score"] <= hi), "sample_id"
            ])
            if not ambig_ids:
                continue

            pred_map = {"spectral_only": spec_preds}
            for s in ["full", "context_only"]:
                _, p, _ = evaluate_setting(df, s, model_name="logreg", seed=seed)
                pred_map[s] = p

            for s in SETTINGS:
                m = ambig_metrics(pred_map, ambig_ids, s)
                m["seed"] = seed
                m["ambig_lo"] = lo
                m["ambig_hi"] = hi
                setting_rows[s].append(m)

        for s in SETTINGS:
            if not setting_rows[s]:
                continue
            sub_df = pd.DataFrame(setting_rows[s])
            thresh_rows.append({
                "window": f"{lo}-{hi}",
                "setting": s,
                "n_seeds": len(sub_df),
                "n_subset_mean": float(sub_df["n_subset"].mean()),
                "pr_auc_mean": float(sub_df["pr_auc"].mean()),
                "pr_auc_std": float(sub_df["pr_auc"].std(ddof=1)) if len(sub_df) > 1 else 0.0,
            })
        print(f"  window {lo}-{hi}: done ({len(seeds)} seeds)")

    thresh_df = pd.DataFrame(thresh_rows)
    thresh_df.to_csv(results / "ambiguous_threshold_robustness_harsh_highov_v1.csv", index=False)

    print(f"\n  {'Window':<10} {'Setting':<22} {'n_subset':>9} {'PR-AUC':>20}")
    print("  " + "-" * 66)
    for _, row in thresh_df.iterrows():
        print(f"  {row['window']:<10} {row['setting']:<22} {row['n_subset_mean']:>9.0f} "
              f"  {row['pr_auc_mean']:.4f} +/- {row['pr_auc_std']:.4f}")

    # ================================================================
    # Check 3 -- w_simple_organic overlap QC figure
    # ================================================================
    print("\n" + "=" * 64)
    print("CHECK 3 -- w_simple_organic overlap QC figure")
    print("=" * 64)

    n_qc = 6000
    cfg_base = ExperimentConfig(seed=42, noise_condition="moderate", rad_simple_organic_hi=RAD_SO_HI_BASE)
    cfg_main = ExperimentConfig(seed=42, noise_condition=NOISE, rad_simple_organic_hi=RAD_SO_HI_MAIN)
    df_base = generate_dataset(n=n_qc, rho_geo=0.75, rho_rad=0.75, cfg=cfg_base)
    df_main = generate_dataset(n=n_qc, rho_geo=0.75, rho_rad=0.75, cfg=cfg_main)

    ov_base = overlap_coeff(
        df_base.loc[df_base["z"] == "ocean_organic", "w_simple_organic"].to_numpy(),
        df_base.loc[df_base["z"] == "radiation_mimic", "w_simple_organic"].to_numpy(),
    )
    ov_main = overlap_coeff(
        df_main.loc[df_main["z"] == "ocean_organic", "w_simple_organic"].to_numpy(),
        df_main.loc[df_main["z"] == "radiation_mimic", "w_simple_organic"].to_numpy(),
    )
    print(f"  Baseline overlap (moderate, rad_so_hi=0.09): {ov_base:.4f}")
    print(f"  Main condition overlap (harsh, rad_so_hi=0.13): {ov_main:.4f}")

    # Save overlap pair table
    ov_table = pd.DataFrame([
        {"condition": "baseline", "noise": "moderate", "rad_so_hi": 0.090,
         "class_a": "ocean_organic", "class_b": "radiation_mimic",
         "weight": "w_simple_organic", "overlap_coefficient": ov_base},
        {"condition": "main", "noise": "harsh", "rad_so_hi": 0.130,
         "class_a": "ocean_organic", "class_b": "radiation_mimic",
         "weight": "w_simple_organic", "overlap_coefficient": ov_main},
    ])
    ov_table.to_csv(qc_dir / "simple_organic_overlap_comparison.csv", index=False)

    # KDE figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    for ax, df_plot, label, ov in [
        (axes[0], df_base, f"Baseline (moderate noise)\nOverlap = {ov_base:.3f}", ov_base),
        (axes[1], df_main, f"Main condition (harsh noise)\nOverlap = {ov_main:.3f}", ov_main),
    ]:
        for cls, color, ls in [
            ("ocean_organic", "#1f77b4", "-"),
            ("radiation_mimic", "#d62728", "--"),
        ]:
            vals = df_plot.loc[df_plot["z"] == cls, "w_simple_organic"].to_numpy()
            vals = vals[np.isfinite(vals)]
            if len(vals) < 5:
                continue
            kde = gaussian_kde(vals, bw_method=0.15)
            x_grid = np.linspace(0, vals.max() * 1.15, 300)
            ax.plot(x_grid, kde(x_grid), color=color, ls=ls, lw=1.8,
                    label=cls.replace("_", " "))

        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Normalized $w_{\\rm simple\\_organic}$", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    fig.suptitle(
        "Simple-organic mixture weight distributions: ocean organic vs radiation mimic",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    fig_path = qc_dir / "simple_organic_overlap_kde_comparison.png"
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {fig_path}")

    # ================================================================
    # Final summary
    # ================================================================
    print("\n" + "=" * 64)
    print("PRE-SUBMISSION CHECK SUMMARY")
    print("=" * 64)
    print(f"\n  Check 1 (rho=0 corner)")
    print(f"    rho=0.75 delta (full - spec):  {full_main - spec_main:+.4f}")
    print(f"    rho=0     delta (full - spec):  {full_rho0 - spec_rho0:+.4f}")
    defended = abs(full_rho0 - spec_rho0) < 0.02
    print(f"    Defense holds (|delta| < 0.02): {defended}")

    print(f"\n  Check 2 (threshold robustness)")
    for lo, hi in thresholds:
        row_spec = thresh_df[(thresh_df["window"] == f"{lo}-{hi}") & (thresh_df["setting"] == "spectral_only")]
        row_full = thresh_df[(thresh_df["window"] == f"{lo}-{hi}") & (thresh_df["setting"] == "full")]
        if len(row_spec) and len(row_full):
            gap = row_full.iloc[0]["pr_auc_mean"] - row_spec.iloc[0]["pr_auc_mean"]
            print(f"    Window {lo}-{hi}:  spec={row_spec.iloc[0]['pr_auc_mean']:.4f}  "
                  f"full={row_full.iloc[0]['pr_auc_mean']:.4f}  gap={gap:+.4f}")

    print(f"\n  Check 3 (overlap QC)")
    print(f"    Baseline: {ov_base:.4f}  -->  Main: {ov_main:.4f}")
    print(f"    KDE figure: results/qc/simple_organic_overlap_kde_comparison.png")

    print("\n  All checks done.")


if __name__ == "__main__":
    main()
