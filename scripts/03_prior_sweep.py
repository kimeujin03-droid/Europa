#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ExperimentConfig
from src.forward_model import generate_dataset
from src.modeling import evaluate_setting
from src.plotting import save_prior_heatmap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["rf", "hgb", "logreg"], default="rf")
    parser.add_argument("--n", type=int, default=2500)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""

    rows = []
    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    for rho_geo in grid:
        for rho_rad in grid:
            cfg = ExperimentConfig(seed=3421 + int(rho_geo * 100) + int(rho_rad * 1000))
            df = generate_dataset(n=args.n, rho_geo=rho_geo, rho_rad=rho_rad, cfg=cfg)
            m_spec, _, _ = evaluate_setting(df, setting="spectral_only", model_name=args.model)
            m_full, _, _ = evaluate_setting(df, setting="full", model_name=args.model)
            m_context, _, _ = evaluate_setting(df, setting="context_only", model_name=args.model)
            row = {
                "model": args.model,
                "rho_geo": rho_geo,
                "rho_rad": rho_rad,
                "pr_auc_spectral": m_spec["pr_auc"],
                "pr_auc_full": m_full["pr_auc"],
                "pr_auc_context_only": m_context["pr_auc"],
                "delta_pr_auc": m_full["pr_auc"] - m_spec["pr_auc"],
                "delta_full_minus_context": m_full["pr_auc"] - m_context["pr_auc"],
                "fpr_rad_spectral": m_spec["fpr_radiation_mimic_top10pct"],
                "fpr_rad_full": m_full["fpr_radiation_mimic_top10pct"],
                "fpr_reduction_rad_mimic": m_spec["fpr_radiation_mimic_top10pct"] - m_full["fpr_radiation_mimic_top10pct"],
                "precision10_spectral": m_spec["precision_at_10pct"],
                "precision10_full": m_full["precision_at_10pct"],
                "recall10_spectral": m_spec["recall_at_top10pct"],
                "recall10_full": m_full["recall_at_top10pct"],
                "delta_precision10": m_full["precision_at_10pct"] - m_spec["precision_at_10pct"],
                "delta_recall10": m_full["recall_at_top10pct"] - m_spec["recall_at_top10pct"],
                "top10_rad_spectral": m_spec["top10_radiation_mimic_count"],
                "top10_rad_full": m_full["top10_radiation_mimic_count"],
                "top10_rad_reduction": m_spec["top10_radiation_mimic_count"] - m_full["top10_radiation_mimic_count"],
                "top10_exogenic_spectral": m_spec["top10_exogenic_count"],
                "top10_exogenic_full": m_full["top10_exogenic_count"],
            }
            rows.append(row)
            print(row)

    out = pd.DataFrame(rows)
    out.to_csv(results / f"prior_sweep_metrics{suffix}.csv", index=False)
    save_prior_heatmap(out, "fpr_reduction_rad_mimic", str(results / f"prior_sweep_heatmap_fpr_reduction{suffix}.png"))
    save_prior_heatmap(out, "delta_pr_auc", str(results / f"prior_sweep_heatmap_delta_pr_auc{suffix}.png"))
    save_prior_heatmap(out, "delta_recall10", str(results / f"prior_sweep_heatmap_delta_recall10{suffix}.png"))
    save_prior_heatmap(out, "top10_rad_reduction", str(results / f"prior_sweep_heatmap_top10_rad_reduction{suffix}.png"))
    print(f"Saved prior sweep outputs to {results}")


if __name__ == "__main__":
    main()
