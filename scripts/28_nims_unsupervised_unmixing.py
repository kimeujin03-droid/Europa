#!/usr/bin/env python
"""Unsupervised PCA/NMF decomposition of real NIMS spectra.

This is model-independent: it uses only the four real NIMS score CSV spectra and
checks whether unsupervised components align with terrain/hemisphere metadata.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
NIMS_DIR = ROOT / "results" / "nims_sanity_check"
OUT_DIR = ROOT / "results" / "nims_unsupervised"
SPEC_COLS = [f"spec_{j:03d}" for j in range(181)]
WAVE = np.arange(0.7, 5.2 + 0.0125, 0.025)

NIMS_FILES = {
    "E6": NIMS_DIR / "nims_e6_scores.csv",
    "E15": NIMS_DIR / "nims_e15_scores.csv",
    "G2": NIMS_DIR / "nims_g2_scores.csv",
    "E11": NIMS_DIR / "nims_e11_scores.csv",
}
META = {
    "E6": {"hemisphere": "leading", "terrain": "chaos"},
    "E15": {"hemisphere": "leading", "terrain": "plains"},
    "G2": {"hemisphere": "trailing", "terrain": "chaos"},
    "E11": {"hemisphere": "trailing", "terrain": "plains"},
}
COLORS = {"E6": "#1f77b4", "E15": "#6baed6", "G2": "#d62728", "E11": "#fb6a4a"}
MARKERS = {"chaos": "o", "plains": "s"}


def load_nims(max_pixels_per_encounter: int = 1200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for enc, path in NIMS_FILES.items():
        df = pd.read_csv(path, usecols=SPEC_COLS)
        if len(df) > max_pixels_per_encounter:
            idx = rng.choice(len(df), size=max_pixels_per_encounter, replace=False)
            df = df.iloc[np.sort(idx)].reset_index(drop=True)
        df["encounter"] = enc
        df["hemisphere"] = META[enc]["hemisphere"]
        df["terrain"] = META[enc]["terrain"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def summarize_components(df: pd.DataFrame, pca_scores: np.ndarray, nmf_scores: np.ndarray) -> pd.DataFrame:
    out = df[["encounter", "hemisphere", "terrain"]].copy()
    for i in range(pca_scores.shape[1]):
        out[f"pca_{i+1}"] = pca_scores[:, i]
    for i in range(nmf_scores.shape[1]):
        out[f"nmf_{i+1}"] = nmf_scores[:, i]

    rows = []
    for group_col in ["encounter", "hemisphere", "terrain"]:
        for comp in [c for c in out.columns if c.startswith(("pca_", "nmf_"))]:
            stats = out.groupby(group_col)[comp].agg(["mean", "std", "count"]).reset_index()
            for _, row in stats.iterrows():
                rows.append(
                    {
                        "group_type": group_col,
                        "group": row[group_col],
                        "component": comp,
                        "mean": float(row["mean"]),
                        "std": float(row["std"]),
                        "count": int(row["count"]),
                    }
                )
    return pd.DataFrame(rows)


def separation_metrics(df: pd.DataFrame, pca_scores: np.ndarray, nmf_scores: np.ndarray) -> pd.DataFrame:
    rows = []
    reps = {
        "pca_2d": pca_scores[:, :2],
        "pca_3d": pca_scores[:, :3],
        "nmf_2d": nmf_scores[:, :2],
        "nmf_3d": nmf_scores[:, :3],
    }
    for name, arr in reps.items():
        for label_col in ["encounter", "hemisphere", "terrain"]:
            labels = df[label_col].to_numpy()
            score = silhouette_score(arr, labels) if len(np.unique(labels)) > 1 else np.nan
            rows.append({"embedding": name, "label": label_col, "silhouette": float(score)})
    return pd.DataFrame(rows)


def plot_results(
    df: pd.DataFrame,
    pca: PCA,
    pca_scores: np.ndarray,
    nmf: NMF,
    nmf_scores: np.ndarray,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    for enc in NIMS_FILES:
        sub = df["encounter"].to_numpy() == enc
        terrain = META[enc]["terrain"]
        ax.scatter(
            pca_scores[sub, 0],
            pca_scores[sub, 1],
            s=9,
            alpha=0.45,
            color=COLORS[enc],
            marker=MARKERS[terrain],
            label=f"{enc} {META[enc]['hemisphere']} {terrain}",
        )
    ax.axhline(0, color="#999", lw=0.7)
    ax.axvline(0, color="#999", lw=0.7)
    ax.set_title("A. PCA Scores")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    for i in range(3):
        ax.plot(WAVE, pca.components_[i], lw=1.3, label=f"PC{i+1}")
    ax.axvline(2.0, color="#777", lw=0.8, ls=":")
    ax.set_title("B. PCA Spectral Loadings")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Loading")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    for enc in NIMS_FILES:
        sub = df["encounter"].to_numpy() == enc
        terrain = META[enc]["terrain"]
        ax.scatter(
            nmf_scores[sub, 0],
            nmf_scores[sub, 1],
            s=9,
            alpha=0.45,
            color=COLORS[enc],
            marker=MARKERS[terrain],
            label=enc,
        )
    ax.set_title("C. NMF Abundances")
    ax.set_xlabel("NMF component 1")
    ax.set_ylabel("NMF component 2")

    ax = axes[1, 1]
    components = nmf.components_
    components = components / np.maximum(components.max(axis=1, keepdims=True), 1e-12)
    for i in range(components.shape[0]):
        ax.plot(WAVE, components[i], lw=1.3, label=f"NMF{i+1}")
    ax.axvline(2.0, color="#777", lw=0.8, ls=":")
    ax.set_title("D. NMF Component Spectra")
    ax.set_xlabel("Wavelength (um)")
    ax.set_ylabel("Normalized component")
    ax.legend(fontsize=8)

    fig.suptitle("Unsupervised Real-NIMS Spectral Structure", fontsize=12)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "nims_unsupervised_pca_nmf.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_nims()
    spectra = df[SPEC_COLS].to_numpy()

    scaled = StandardScaler().fit_transform(spectra)
    pca = PCA(n_components=5, random_state=42)
    pca_scores = pca.fit_transform(scaled)

    # NMF requires nonnegative inputs; NIMS spectra are already min-max normalized.
    nmf = NMF(n_components=4, init="nndsvda", random_state=42, max_iter=3000, tol=1e-5)
    nmf_scores = nmf.fit_transform(np.clip(spectra, 0, None))

    pixel_scores = df[["encounter", "hemisphere", "terrain"]].copy()
    for i in range(pca_scores.shape[1]):
        pixel_scores[f"pca_{i+1}"] = pca_scores[:, i]
    for i in range(nmf_scores.shape[1]):
        pixel_scores[f"nmf_{i+1}"] = nmf_scores[:, i]
    pixel_scores.to_csv(OUT_DIR / "nims_unsupervised_pixel_scores.csv", index=False)

    comp_summary = summarize_components(df, pca_scores, nmf_scores)
    comp_summary.to_csv(OUT_DIR / "nims_unsupervised_component_summary.csv", index=False)

    sep = separation_metrics(df, pca_scores, nmf_scores)
    sep.to_csv(OUT_DIR / "nims_unsupervised_separation_metrics.csv", index=False)

    fig_path = plot_results(df, pca, pca_scores, nmf, nmf_scores)
    print(sep.to_string(index=False))
    print(f"\nSaved {fig_path}")
    print(f"Saved {OUT_DIR / 'nims_unsupervised_component_summary.csv'}")


if __name__ == "__main__":
    main()
