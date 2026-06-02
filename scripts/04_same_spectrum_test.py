#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ExperimentConfig
from src.forward_model import generate_dataset
from src.modeling import feature_cols, build_model


def two_sided_sign_test_pvalue(deltas: np.ndarray) -> float:
    nonzero = deltas[np.abs(deltas) > 1e-12]
    n = len(nonzero)
    if n == 0:
        return 1.0
    smaller_side = int(min(np.sum(nonzero > 0), np.sum(nonzero < 0)))
    tail = sum(math.comb(n, k) for k in range(smaller_side + 1)) / (2 ** n)
    return float(min(1.0, 2.0 * tail))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=10000)
    parser.add_argument("--n-pairs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=3421)
    parser.add_argument("--rho-geo", type=float, default=0.75)
    parser.add_argument("--rho-rad", type=float, default=0.75)
    parser.add_argument("--noise-condition", choices=["clean", "moderate", "harsh"], default="moderate")
    parser.add_argument("--model", choices=["rf", "hgb", "logreg"], default="rf")
    parser.add_argument("--ambiguous-low", type=float, default=0.4)
    parser.add_argument("--ambiguous-high", type=float, default=0.7)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    cfg = ExperimentConfig(seed=args.seed, noise_condition=args.noise_condition)
    df = generate_dataset(n=args.n_train, rho_geo=args.rho_geo, rho_rad=args.rho_rad, cfg=cfg)

    cols_full = feature_cols(df, "full")
    cols_spec = feature_cols(df, "spectral_only")

    train_df, test_df = train_test_split(df, test_size=0.3, stratify=df["y"], random_state=args.seed)
    full_model = build_model(args.model, calibrate=True, seed=args.seed)
    spec_model = build_model(args.model, calibrate=True, seed=args.seed)
    full_model.fit(train_df[cols_full], train_df["y"])
    spec_model.fit(train_df[cols_spec], train_df["y"])

    # Select spectra that the spectral-only model finds ambiguous, then clone each
    # spectrum into exchange-favorable and radiation-prone contexts.
    rng = np.random.default_rng(args.seed + 2026)
    spec_scores = spec_model.predict_proba(test_df[cols_spec])[:, 1]
    candidate_df = test_df.copy()
    candidate_df["spectral_only_original_score"] = spec_scores
    ambiguous = candidate_df[
        candidate_df["spectral_only_original_score"].between(args.ambiguous_low, args.ambiguous_high)
    ].copy()
    if len(ambiguous) < args.n_pairs:
        midpoint = 0.5 * (args.ambiguous_low + args.ambiguous_high)
        ambiguous = candidate_df.assign(
            distance_to_ambiguous_midpoint=(candidate_df["spectral_only_original_score"] - midpoint).abs()
        ).sort_values("distance_to_ambiguous_midpoint")
        print(
            f"WARNING: only {len(candidate_df[candidate_df['spectral_only_original_score'].between(args.ambiguous_low, args.ambiguous_high)])} "
            f"spectra inside [{args.ambiguous_low}, {args.ambiguous_high}]. "
            "Using spectra closest to the ambiguous midpoint."
        )
    selected = ambiguous.head(args.n_pairs).copy()

    rows = []
    spec_columns = cols_spec
    for pair_id, (_, source_row) in enumerate(selected.iterrows()):
        spec_values = {col: source_row[col] for col in spec_columns}

        exchange = {
            "pair_id": pair_id,
            "context": "exchange",
            "source_sample_id": source_row["sample_id"],
            "source_z": source_row["z"],
            "source_y": source_row["y"],
            "source_spectral_only_score": source_row["spectral_only_original_score"],
            **spec_values,
            "chaos_proximity": rng.uniform(0.78, 0.98),
            "lineament_proximity": rng.uniform(0.75, 0.96),
            "ridge_proximity": rng.uniform(0.62, 0.90),
            "young_terrain": 1.0,
            "activity_proxy": rng.uniform(0.70, 0.95),
            "trailing_hemisphere": 0.0,
            "radiation_exposure": rng.uniform(0.03, 0.25),
            "sulfur_proxy": rng.uniform(0.03, 0.22),
            "rad_mimic_proxy": rng.uniform(0.02, 0.18),
        }
        radiation = {
            "pair_id": pair_id,
            "context": "radiation",
            "source_sample_id": source_row["sample_id"],
            "source_z": source_row["z"],
            "source_y": source_row["y"],
            "source_spectral_only_score": source_row["spectral_only_original_score"],
            **spec_values,
            "chaos_proximity": rng.uniform(0.02, 0.24),
            "lineament_proximity": rng.uniform(0.02, 0.26),
            "ridge_proximity": rng.uniform(0.04, 0.30),
            "young_terrain": 0.0,
            "activity_proxy": rng.uniform(0.02, 0.25),
            "trailing_hemisphere": 1.0,
            "radiation_exposure": rng.uniform(0.76, 0.98),
            "sulfur_proxy": rng.uniform(0.72, 0.96),
            "rad_mimic_proxy": rng.uniform(0.70, 0.96),
        }
        rows.extend([exchange, radiation])

    cases = pd.DataFrame(rows)
    cases["score_spectral_only"] = spec_model.predict_proba(cases[cols_spec])[:, 1]
    cases["score_full_spatial_spectral"] = full_model.predict_proba(cases[cols_full])[:, 1]

    wide = cases.pivot(index="pair_id", columns="context", values=["score_spectral_only", "score_full_spatial_spectral"])
    paired = pd.DataFrame({
        "pair_id": wide.index,
        "delta_spectral_only_exchange_minus_radiation": wide[("score_spectral_only", "exchange")] - wide[("score_spectral_only", "radiation")],
        "delta_full_exchange_minus_radiation": wide[("score_full_spatial_spectral", "exchange")] - wide[("score_full_spatial_spectral", "radiation")],
        "score_spectral_exchange": wide[("score_spectral_only", "exchange")],
        "score_spectral_radiation": wide[("score_spectral_only", "radiation")],
        "score_full_exchange": wide[("score_full_spatial_spectral", "exchange")],
        "score_full_radiation": wide[("score_full_spatial_spectral", "radiation")],
    })
    stats = {
        "model": args.model,
        "n_pairs_requested": args.n_pairs,
        "n_pairs": len(paired),
        "ambiguous_low": args.ambiguous_low,
        "ambiguous_high": args.ambiguous_high,
        "selected_source_score_mean": float(selected["spectral_only_original_score"].mean()),
        "selected_source_score_std": float(selected["spectral_only_original_score"].std(ddof=1)),
        "spectral_delta_mean": float(paired["delta_spectral_only_exchange_minus_radiation"].mean()),
        "spectral_delta_std": float(paired["delta_spectral_only_exchange_minus_radiation"].std(ddof=1)),
        "full_delta_mean": float(paired["delta_full_exchange_minus_radiation"].mean()),
        "full_delta_std": float(paired["delta_full_exchange_minus_radiation"].std(ddof=1)),
        "full_delta_sign_test_p": two_sided_sign_test_pvalue(paired["delta_full_exchange_minus_radiation"].to_numpy()),
    }

    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else f"_{args.model}_ambiguous"
    cases[["pair_id", "context", "score_spectral_only", "score_full_spatial_spectral"]].to_csv(
        results / f"same_spectrum_scores{suffix}.csv", index=False
    )
    paired.to_csv(results / f"same_spectrum_paired_deltas{suffix}.csv", index=False)
    pd.DataFrame([stats]).to_csv(results / f"same_spectrum_paired_stats{suffix}.csv", index=False)

    plt.figure(figsize=(7, 4))
    plot_data = [
        cases.loc[cases["context"] == "exchange", "score_spectral_only"],
        cases.loc[cases["context"] == "radiation", "score_spectral_only"],
        cases.loc[cases["context"] == "exchange", "score_full_spatial_spectral"],
        cases.loc[cases["context"] == "radiation", "score_full_spatial_spectral"],
    ]
    plt.boxplot(plot_data, tick_labels=["spec\nexchange", "spec\nradiation", "full\nexchange", "full\nradiation"], showfliers=False)
    jitter_rng = np.random.default_rng(args.seed)
    for x, values in enumerate(plot_data, start=1):
        plt.scatter(x + jitter_rng.normal(0, 0.035, size=len(values)), values, s=10, alpha=0.35)
    plt.ylabel("Triage score")
    plt.title("Same spectra, different context score distributions")
    plt.tight_layout()
    plt.savefig(results / f"same_spectrum_comparison{suffix}.png", dpi=200)
    plt.close()

    print(pd.DataFrame([stats]))
    print(f"Saved same-spectrum outputs to {results}")


if __name__ == "__main__":
    main()
