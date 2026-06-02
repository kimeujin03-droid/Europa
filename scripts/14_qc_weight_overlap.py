#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "qc"

DEFAULT_WEIGHTS = [
    "w_simple_organic",
    "w_ocean_salt",
    "w_sulfuric_acid_hydrate",
    "w_rad_salt",
]

PAIR_CHECKS = [
    ("ocean_organic", "radiation_mimic", "w_simple_organic"),
    ("ocean_organic", "radiation_mimic", "w_ocean_salt"),
    ("ocean_organic", "exogenic_complex_organic", "w_simple_organic"),
    ("radiation_mimic", "exogenic_complex_organic", "w_simple_organic"),
]


def overlap_coefficient(x: np.ndarray, y: np.ndarray, bins: int = 40) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan

    lo = min(float(x.min()), float(y.min()))
    hi = max(float(x.max()), float(y.max()))
    if hi <= lo:
        return 1.0

    hx, edges = np.histogram(x, bins=bins, range=(lo, hi), density=True)
    hy, _ = np.histogram(y, bins=edges, density=True)
    width = edges[1] - edges[0]
    return float(np.minimum(hx, hy).sum() * width)


def summarize_weights(df: pd.DataFrame, weight_cols: list[str]) -> pd.DataFrame:
    rows = []
    for z, group in df.groupby("z"):
        for col in weight_cols:
            values = group[col].astype(float)
            rows.append(
                {
                    "z": z,
                    "weight": col,
                    "n": len(values),
                    "mean": values.mean(),
                    "std": values.std(ddof=1),
                    "p05": values.quantile(0.05),
                    "p25": values.quantile(0.25),
                    "p50": values.quantile(0.50),
                    "p75": values.quantile(0.75),
                    "p95": values.quantile(0.95),
                    "min": values.min(),
                    "max": values.max(),
                }
            )
    return pd.DataFrame(rows)


def pair_overlap(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for z_a, z_b, col in PAIR_CHECKS:
        if col not in df.columns:
            continue
        x = df.loc[df["z"] == z_a, col].to_numpy(dtype=float)
        y = df.loc[df["z"] == z_b, col].to_numpy(dtype=float)
        rows.append(
            {
                "class_a": z_a,
                "class_b": z_b,
                "weight": col,
                "n_a": len(x),
                "n_b": len(y),
                "mean_a": float(np.mean(x)) if len(x) else np.nan,
                "mean_b": float(np.mean(y)) if len(y) else np.nan,
                "overlap_coefficient": overlap_coefficient(x, y),
            }
        )
    return pd.DataFrame(rows)


def plot_boxplots(df: pd.DataFrame, weight_cols: list[str], out_path: Path) -> None:
    classes = sorted(df["z"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.ravel()

    for ax, col in zip(axes, weight_cols):
        data = [df.loc[df["z"] == z, col].astype(float).to_numpy() for z in classes]
        ax.boxplot(data, tick_labels=classes, showfliers=False)
        ax.set_title(col)
        ax.set_ylabel("Normalized mixture weight")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(alpha=0.25, axis="y")

    for ax in axes[len(weight_cols) :]:
        ax.axis("off")

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/synthetic_dataset.csv")
    parser.add_argument("--out-dir", default="results/qc")
    args = parser.parse_args()

    dataset_path = ROOT / args.dataset
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path)
    if "z" not in df.columns:
        raise ValueError(f"{dataset_path} must contain z")

    weight_cols = [col for col in DEFAULT_WEIGHTS if col in df.columns]
    if not weight_cols:
        raise ValueError("No expected mixture-weight columns found. Regenerate the dataset after the A1 patch.")

    summary = summarize_weights(df, weight_cols)
    summary.to_csv(out_dir / "mixture_weight_summary.csv", index=False, encoding="utf-8-sig")

    overlap = pair_overlap(df)
    overlap.to_csv(out_dir / "mixture_weight_overlap_pairs.csv", index=False, encoding="utf-8-sig")

    plot_boxplots(df, weight_cols, out_dir / "mixture_weight_overlap_boxplot.png")

    print("Mixture weight pair overlap:")
    print(overlap.to_string(index=False))
    print(f"\nSaved weight QC files to {out_dir}")


if __name__ == "__main__":
    main()
