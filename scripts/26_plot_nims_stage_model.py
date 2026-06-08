#!/usr/bin/env python
"""Plot NIMS stage-model deployment curves and summary panels."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "nims_stage_model"
FIG_DIR = OUT_DIR / "figures"
SPEC_COLS = [f"spec_{j:03d}" for j in range(181)]
WAVE = np.arange(0.7, 5.2 + 0.0125, 0.025)

STAGE = "02_ocean_salt_usgs_epsomite"
BANDS = {
    "0.7-5.2 um": "0p7_5p2",
    "0.7-2.0 um": "0p7_2p0",
}
ENCOUNTERS = ["E6", "E15", "G2", "E11"]
COLORS = {
    "E6": "#1f77b4",
    "E15": "#6baed6",
    "G2": "#d62728",
    "E11": "#fb6a4a",
}
HATCHES = {
    "E6": "",
    "E15": "////",
    "G2": "",
    "E11": "////",
}


def load_summary(tag: str) -> pd.DataFrame:
    return pd.read_csv(OUT_DIR / f"nims_summary_{STAGE}_{tag}.csv")


def load_scores(enc: str, tag: str) -> pd.DataFrame:
    return pd.read_csv(OUT_DIR / f"nims_{enc.lower()}_{STAGE}_{tag}_scores.csv")


def plot_full_summary() -> Path:
    full_tag = BANDS["0.7-5.2 um"]
    short_tag = BANDS["0.7-2.0 um"]
    full = load_summary(full_tag)
    short = load_summary(short_tag)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # A. Mean spectra curves from full-range scored files.
    ax = axes[0, 0]
    for enc in ENCOUNTERS:
        df = load_scores(enc, full_tag)
        mean = df[SPEC_COLS].mean(axis=0).to_numpy()
        p10 = df[SPEC_COLS].quantile(0.10).to_numpy()
        p90 = df[SPEC_COLS].quantile(0.90).to_numpy()
        ax.plot(WAVE, mean, color=COLORS[enc], lw=1.5, label=enc)
        ax.fill_between(WAVE, p10, p90, color=COLORS[enc], alpha=0.10, linewidth=0)
    ax.axvspan(0.7, 2.0, color="#e8f3ff", alpha=0.55, label="short-wave model")
    ax.axvline(2.0, color="#777", lw=0.8, ls=":")
    ax.set_title("A. NIMS Mean Spectra")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Normalized reflectance")
    ax.set_xlim(0.7, 5.2)
    ax.legend(fontsize=8, ncol=3)

    # B. Full-model score distributions for short-wave run.
    ax = axes[0, 1]
    bins = np.linspace(0, 0.08, 42)
    for enc in ENCOUNTERS:
        df = load_scores(enc, short_tag)
        ax.hist(df["p_full"], bins=bins, histtype="step", lw=1.6, color=COLORS[enc], label=enc)
    ax.set_title("B. Real-NIMS Full-Model Probability\n(0.7-2.0 um model)")
    ax.set_xlabel("p_full")
    ax.set_ylabel("Pixel count")
    ax.legend(fontsize=8)

    # C. Context deltas for short-wave run.
    ax = axes[1, 0]
    x = np.arange(len(ENCOUNTERS))
    width = 0.25
    short_ordered = short.set_index("encounter").loc[ENCOUNTERS].reset_index()
    for offset, col, label, color in [
        (-width, "delta_geo", "geo", "#2ca25f"),
        (0.0, "delta_rad", "rad", "#756bb1"),
        (width, "delta_full", "full", "#525252"),
    ]:
        ax.bar(x + offset, short_ordered[col], width=width, color=color, alpha=0.85, label=label)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.encounter}\n{r.terrain}" for r in short_ordered.itertuples()], fontsize=8)
    ax.set_title("C. Context Decision-Value Shifts\n(0.7-2.0 um model)")
    ax.set_ylabel("Delta decision value")
    ax.legend(fontsize=8)

    # D. OOD ratio comparison full vs short.
    ax = axes[1, 1]
    full_ordered = full.set_index("encounter").loc[ENCOUNTERS].reset_index()
    width = 0.36
    ax.bar(x - width / 2, full_ordered["mean_ood_ratio"], width=width, color="#969696", label="0.7-5.2 um")
    ax.bar(x + width / 2, short_ordered["mean_ood_ratio"], width=width, color="#3182bd", label="0.7-2.0 um")
    ax.set_xticks(x)
    ax.set_xticklabels(ENCOUNTERS)
    ax.set_title("D. NIMS OOD Ratio")
    ax.set_ylabel("NIMS / synthetic internal OOD")
    ax.legend(fontsize=8)
    ax.set_yscale("log")

    fig.suptitle("NIMS Deployment With USGS Epsomite Salt Endmember", fontsize=12)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "nims_stage_model_summary.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_band_comparison_curves() -> Path:
    full = load_summary(BANDS["0.7-5.2 um"]).set_index("encounter").loc[ENCOUNTERS]
    short = load_summary(BANDS["0.7-2.0 um"]).set_index("encounter").loc[ENCOUNTERS]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    x = np.arange(len(ENCOUNTERS))

    ax = axes[0]
    ax.plot(x, full["mean_p_full"], marker="o", color="#969696", lw=1.8, label="0.7-5.2 um")
    ax.plot(x, short["mean_p_full"], marker="o", color="#3182bd", lw=1.8, label="0.7-2.0 um")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(ENCOUNTERS)
    ax.set_title("Mean p_full")
    ax.set_ylabel("probability")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(x, full["mean_dv_full"], marker="o", color="#969696", lw=1.8, label="0.7-5.2 um")
    ax.plot(x, short["mean_dv_full"], marker="o", color="#3182bd", lw=1.8, label="0.7-2.0 um")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ENCOUNTERS)
    ax.set_title("Mean decision value")
    ax.set_ylabel("DV")

    ax = axes[2]
    ax.plot(x, full["mean_ood_ratio"], marker="o", color="#969696", lw=1.8, label="0.7-5.2 um")
    ax.plot(x, short["mean_ood_ratio"], marker="o", color="#3182bd", lw=1.8, label="0.7-2.0 um")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(ENCOUNTERS)
    ax.set_title("Mean OOD ratio")
    ax.set_ylabel("ratio")

    fig.suptitle("Full-Range vs Short-Wave Real-NIMS Model Response", fontsize=11)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "nims_stage_model_full_vs_short_curves.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    outputs = [plot_full_summary(), plot_band_comparison_curves()]
    for path in outputs:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
