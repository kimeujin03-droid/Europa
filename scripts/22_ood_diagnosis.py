#!/usr/bin/env python
"""OOD gap diagnosis between NIMS cubes and synthetic training data.

Steps:
  1. Per-band statistics (mean, std, SNR) for NIMS pixels and synthetic
  2. Continuum slope / curvature distribution comparison
  3. PCA decomposition -- which axes separate NIMS from synthetic cloud
  4. Concrete forward-model correction proposal
  5. (Optional) Re-run NIMS sanity check under corrected model

Outputs:
  results/qc/ood_diag_bandstats.csv
  results/qc/ood_diag_pca_loadings.csv
  results/qc/ood_diag_summary.txt
  results/qc/ood_diag_bandmeans.png
  results/qc/ood_diag_pca.png
  results/qc/ood_diag_continuum.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ExperimentConfig
from src.forward_model import generate_dataset

QC    = ROOT / "results" / "qc"
NIMS  = ROOT / "results" / "nims_sanity_check"
QC.mkdir(exist_ok=True)

NIMS_FILES = {
    "E6":  NIMS / "nims_e6_scores.csv",
    "E15": NIMS / "nims_e15_scores.csv",
    "G2":  NIMS / "nims_g2_scores.csv",
    "E11": NIMS / "nims_e11_scores.csv",
}

SPEC_COLS = [f"spec_{j:03d}" for j in range(181)]
CFG       = ExperimentConfig(seed=42, noise_condition="harsh",
                             rad_simple_organic_hi=0.130)
WAVE      = CFG.wavelengths          # 0.7–5.2 µm, 181 pts


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_nims_all() -> tuple[np.ndarray, np.ndarray]:
    """Return (nims_spectra, obs_labels) where obs_labels is int 0-3."""
    parts, labels = [], []
    for k, (enc, path) in enumerate(NIMS_FILES.items()):
        df = pd.read_csv(path, usecols=SPEC_COLS)
        parts.append(df.to_numpy())
        labels.extend([k] * len(df))
    return np.vstack(parts), np.array(labels)


def load_synth(n: int = 6000, seed: int = 42) -> np.ndarray:
    cfg = ExperimentConfig(seed=seed, noise_condition="harsh",
                           rad_simple_organic_hi=0.130)
    df  = generate_dataset(n=n, rho_geo=0.75, rho_rad=0.75, cfg=cfg)
    return df[SPEC_COLS].to_numpy()


def snr_per_pixel(spectra: np.ndarray) -> np.ndarray:
    """Estimate per-pixel SNR as mean / (std of adjacent-band diffs / sqrt2)."""
    diffs  = np.diff(spectra, axis=1)
    sigma  = diffs.std(axis=1) / np.sqrt(2)
    signal = spectra.mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(sigma > 0, signal / sigma, np.nan)
    return snr


def fit_continuum(spectra: np.ndarray, deg: int = 2) -> np.ndarray:
    """Fit per-spectrum polynomial of degree deg; return coefficients [n_pix, deg+1]."""
    x = (WAVE - WAVE.mean()) / (WAVE.max() - WAVE.min())   # normalised x in [-0.5, 0.5]
    coeffs = np.polyfit(x, spectra.T, deg).T               # (n_pix, deg+1)
    return coeffs


def thermal_correction_estimate(spectra: np.ndarray) -> np.ndarray:
    """
    Rough thermal emission estimate for Europa at ~100 K.
    Planck function (relative) convolved to band grid — used only for
    visualising the long-wavelength lift, not for hard correction.
    """
    h, c, k = 6.626e-34, 3e8, 1.38e-23
    T = 100.0
    lam = WAVE * 1e-6     # metres
    B = (2 * h * c**2 / lam**5) / (np.exp(h * c / (lam * k * T)) - 1)
    B_rel = B / B.max()
    return B_rel


# ─────────────────────────────────────────────────────────────────────────────
# 1. Band-level statistics
# ─────────────────────────────────────────────────────────────────────────────

def analyse_bandstats(nims: np.ndarray, synth: np.ndarray, obs_labels: np.ndarray) -> pd.DataFrame:
    rows = []
    # NIMS per-obs
    enc_names = list(NIMS_FILES.keys())
    for k, enc in enumerate(enc_names):
        sub = nims[obs_labels == k]
        for j in range(181):
            rows.append({"source": f"NIMS_{enc}", "band": j,
                         "wave": WAVE[j],
                         "mean": sub[:, j].mean(), "std": sub[:, j].std()})
    # synthetic (all classes pooled)
    for j in range(181):
        rows.append({"source": "synth_pool", "band": j,
                     "wave": WAVE[j],
                     "mean": synth[:, j].mean(), "std": synth[:, j].std()})
    return pd.DataFrame(rows)


def plot_bandmeans(df: pd.DataFrame, out: Path) -> None:
    enc_names = list(NIMS_FILES.keys())
    enc_colors = {"E6": "#1565c0", "E15": "#42a5f5", "G2": "#c62828", "E11": "#ef9a9a"}

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # panel A: mean spectra
    ax = axes[0]
    synth_sub = df[df["source"] == "synth_pool"]
    ax.fill_between(synth_sub["wave"],
                    synth_sub["mean"] - synth_sub["std"],
                    synth_sub["mean"] + synth_sub["std"],
                    color="gray", alpha=0.2, label="synth ±1σ")
    ax.plot(synth_sub["wave"], synth_sub["mean"],
            color="gray", lw=1.5, ls="--", label="synth mean")

    for enc in enc_names:
        sub = df[df["source"] == f"NIMS_{enc}"]
        ax.fill_between(sub["wave"],
                        sub["mean"] - sub["std"],
                        sub["mean"] + sub["std"],
                        color=enc_colors[enc], alpha=0.15)
        ax.plot(sub["wave"], sub["mean"], color=enc_colors[enc],
                lw=1.6, label=f"NIMS {enc}")

    ax.set_ylabel("Normalised reflectance (mean)")
    ax.set_title("(A) Per-band mean spectra: NIMS vs synthetic")
    ax.legend(fontsize=8, ncol=3)
    ax.set_xlim(0.7, 5.2)

    # panel B: relative difference from synthetic mean
    ax = axes[1]
    synth_mean = synth_sub["mean"].values
    for enc in enc_names:
        sub = df[df["source"] == f"NIMS_{enc}"].sort_values("band")
        delta = sub["mean"].values - synth_mean
        ax.plot(sub["wave"], delta, color=enc_colors[enc], lw=1.4, label=enc)

    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("NIMS mean − synth mean")
    ax.set_title("(B) NIMS spectral bias relative to synthetic mean")
    ax.legend(fontsize=8)

    # mark thermal emission onset
    ax.axvline(3.5, color="#888", ls=":", lw=1.0)
    axes[0].axvline(3.5, color="#888", ls=":", lw=1.0)
    for a in axes:
        a.text(3.52, a.get_ylim()[0] + 0.01 * (a.get_ylim()[1] - a.get_ylim()[0]),
               "thermal onset", fontsize=7, color="#666", rotation=90, va="bottom")

    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Continuum analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_continuum(nims: np.ndarray, synth: np.ndarray,
                      obs_labels: np.ndarray, out: Path) -> pd.DataFrame:
    enc_names = list(NIMS_FILES.keys())

    # Split on short-wave (0.7-3.4 µm) to avoid thermal contamination
    sw_mask = WAVE <= 3.4
    nims_sw  = nims[:, sw_mask]
    synth_sw = synth[:, sw_mask]
    wave_sw  = WAVE[sw_mask]

    def poly_stats(specs: np.ndarray, deg: int = 2) -> dict:
        x = (wave_sw - wave_sw.mean()) / (wave_sw[-1] - wave_sw[0])
        coeffs = np.polyfit(x, specs.T, deg).T
        return {
            "slope_mean":    float(coeffs[:, -2].mean()),
            "slope_std":     float(coeffs[:, -2].std()),
            "curv_mean":     float(coeffs[:, -3].mean()) if deg >= 2 else np.nan,
            "curv_std":      float(coeffs[:, -3].std())  if deg >= 2 else np.nan,
            "range_mean":    float((specs.max(axis=1) - specs.min(axis=1)).mean()),
            "range_std":     float((specs.max(axis=1) - specs.min(axis=1)).std()),
        }

    rows = []
    for k, enc in enumerate(enc_names):
        sub = nims_sw[obs_labels == k]
        s   = poly_stats(sub)
        s["source"] = f"NIMS_{enc}"
        rows.append(s)

    s = poly_stats(synth_sw)
    s["source"] = "synth_pool"
    rows.append(s)

    df = pd.DataFrame(rows)

    # SNR comparison
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # panel A: slope distribution
    ax = axes[0]
    ax.bar(df["source"], df["slope_mean"], yerr=df["slope_std"],
           color=["#1565c0","#42a5f5","#c62828","#ef9a9a","gray"], capsize=4)
    ax.set_title("(A) Continuum slope\n(0.7–3.4 µm, per-spectrum)")
    ax.set_ylabel("Polynomial 1st-order coeff")
    ax.tick_params(axis='x', rotation=30)

    # panel B: curvature distribution
    ax = axes[1]
    ax.bar(df["source"], df["curv_mean"], yerr=df["curv_std"],
           color=["#1565c0","#42a5f5","#c62828","#ef9a9a","gray"], capsize=4)
    ax.set_title("(B) Continuum curvature\n(0.7–3.4 µm)")
    ax.set_ylabel("Polynomial 2nd-order coeff")
    ax.tick_params(axis='x', rotation=30)

    # panel C: dynamic range distribution
    ax = axes[2]
    ax.bar(df["source"], df["range_mean"], yerr=df["range_std"],
           color=["#1565c0","#42a5f5","#c62828","#ef9a9a","gray"], capsize=4)
    ax.set_title("(C) Spectral dynamic range\n(max − min per spectrum, 0.7–3.4 µm)")
    ax.set_ylabel("Normalised range")
    ax.tick_params(axis='x', rotation=30)

    fig.suptitle("Continuum statistics: NIMS cubes vs synthetic (short-wave 0.7–3.4 µm)",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. PCA diagnosis
# ─────────────────────────────────────────────────────────────────────────────

def pca_diagnosis(nims: np.ndarray, synth: np.ndarray,
                  obs_labels: np.ndarray, out: Path) -> pd.DataFrame:
    enc_names = list(NIMS_FILES.keys())
    enc_colors = ["#1565c0", "#42a5f5", "#c62828", "#ef9a9a"]

    n_comp = 10
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(synth)

    synth_proj = pca.transform(synth)
    nims_proj  = pca.transform(nims)

    var_exp = pca.explained_variance_ratio_

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Panel A: explained variance
    ax = axes[0, 0]
    ax.bar(range(1, n_comp + 1), var_exp * 100, color="#1976d2", alpha=0.8)
    ax.set_xlabel("PC")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title("(A) Synthetic PCA explained variance")

    # Panel B-C: PC1 vs PC2 scatter, PC3 vs PC4
    for plot_idx, (pc_x, pc_y, ax) in enumerate([
        (0, 1, axes[0, 1]),
        (2, 3, axes[0, 2]),
    ]):
        ax.scatter(synth_proj[:, pc_x], synth_proj[:, pc_y],
                   color="lightgray", alpha=0.15, s=4, label="synthetic", zorder=1)
        for k, enc in enumerate(enc_names):
            sub = nims_proj[obs_labels == k]
            ax.scatter(sub[:, pc_x], sub[:, pc_y],
                       color=enc_colors[k], alpha=0.5, s=10,
                       label=f"NIMS {enc}", zorder=3)
        ax.set_xlabel(f"PC{pc_x+1} ({var_exp[pc_x]*100:.1f}%)")
        ax.set_ylabel(f"PC{pc_y+1} ({var_exp[pc_y]*100:.1f}%)")
        ax.set_title(f"({'BC'[plot_idx]}) PC{pc_x+1} vs PC{pc_y+1}")
        ax.legend(fontsize=7, markerscale=1.5)

    # Panel D: PC loadings (components)
    ax = axes[1, 0]
    for i in range(4):
        ax.plot(WAVE, pca.components_[i], lw=1.2,
                label=f"PC{i+1} ({var_exp[i]*100:.1f}%)")
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("Loading")
    ax.set_title("(D) PCA loadings (PC1–4)")
    ax.legend(fontsize=7)
    ax.axvline(3.5, color="#aaa", ls=":", lw=0.8)

    # Panel E: per-PC mean shift (NIMS - synth centroid)
    ax = axes[1, 1]
    synth_centroid = synth_proj.mean(axis=0)
    shifts = {}
    for k, enc in enumerate(enc_names):
        sub   = nims_proj[obs_labels == k]
        shift = sub.mean(axis=0) - synth_centroid
        shifts[enc] = shift
        ax.plot(range(1, n_comp + 1), shift, marker="o", ms=4, label=enc,
                color=enc_colors[k])
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("PC index")
    ax.set_ylabel("NIMS mean − synth mean (PC units)")
    ax.set_title("(E) NIMS vs synthetic centroid shift per PC")
    ax.legend(fontsize=7)

    # Panel F: per-PC OOD contribution (squared shift / synth var)
    ax = axes[1, 2]
    synth_var = synth_proj.var(axis=0)
    for k, enc in enumerate(enc_names):
        contrib = shifts[enc]**2 / (synth_var + 1e-12)
        ax.bar(np.arange(1, n_comp + 1) + k * 0.18 - 0.27, contrib,
               width=0.18, color=enc_colors[k], alpha=0.85, label=enc)
    ax.set_xlabel("PC index")
    ax.set_ylabel(r"$\Delta\mu_k^2 / \sigma^2_k$ (synth)")
    ax.set_title("(F) OOD gap contribution per PC\n(Mahalanobis-like decomposition)")
    ax.legend(fontsize=7)

    fig.suptitle("PCA diagnosis: where NIMS pixels diverge from synthetic cloud",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")

    # Return per-PC shift table
    rows = []
    for k, enc in enumerate(enc_names):
        for i in range(n_comp):
            rows.append({
                "enc": enc, "pc": i + 1,
                "shift": float(shifts[enc][i]),
                "synth_std": float(np.sqrt(synth_var[i])),
                "n_sigma": float(shifts[enc][i] / (np.sqrt(synth_var[i]) + 1e-12)),
                "var_explained": float(var_exp[i]),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 4. SNR estimation
# ─────────────────────────────────────────────────────────────────────────────

def snr_report(nims: np.ndarray, synth: np.ndarray, obs_labels: np.ndarray) -> dict:
    enc_names = list(NIMS_FILES.keys())

    # NIMS noise: std of adjacent-band differences / sqrt(2)
    diffs_nims  = np.diff(nims, axis=1)
    sigma_nims  = (diffs_nims.std(axis=1) / np.sqrt(2)).mean()
    snr_nims    = (nims.mean(axis=1) / (diffs_nims.std(axis=1) / np.sqrt(2))).mean()

    # Synthetic noise (injected σ=0.035 before smoothing+normalization → effective σ differs)
    diffs_synth = np.diff(synth, axis=1)
    sigma_synth = (diffs_synth.std(axis=1) / np.sqrt(2)).mean()
    snr_synth   = (synth.mean(axis=1) / (diffs_synth.std(axis=1) / np.sqrt(2))).mean()

    report = {
        "nims_effective_sigma": float(sigma_nims),
        "synth_effective_sigma": float(sigma_synth),
        "nims_mean_snr": float(snr_nims),
        "synth_mean_snr": float(snr_synth),
        "sigma_ratio_nims_over_synth": float(sigma_nims / (sigma_synth + 1e-12)),
    }
    # Per-observation
    for k, enc in enumerate(enc_names):
        sub = nims[obs_labels == k]
        d   = np.diff(sub, axis=1)
        report[f"sigma_{enc}"] = float(d.std(axis=1).mean() / np.sqrt(2))
        report[f"snr_{enc}"]   = float((sub.mean(axis=1) /
                                        (d.std(axis=1) / np.sqrt(2))).mean())

    # Long-wave (>3.5 µm) vs short-wave (<3.5 µm) mean reflectance ratio
    lw = WAVE > 3.5
    sw = WAVE <= 3.5
    for src, specs in [("nims", nims), ("synth", synth)]:
        report[f"{src}_lw_mean"]   = float(specs[:, lw].mean())
        report[f"{src}_sw_mean"]   = float(specs[:, sw].mean())
        report[f"{src}_lw_sw_ratio"] = float(specs[:, lw].mean() / (specs[:, sw].mean() + 1e-12))

    return report


# ─────────────────────────────────────────────────────────────────────────────
# 5. Write summary report
# ─────────────────────────────────────────────────────────────────────────────

def write_summary(snr_report: dict, pca_df: pd.DataFrame,
                  cont_df: pd.DataFrame, out: Path) -> None:
    lines = [
        "=" * 72,
        "OOD DIAGNOSIS REPORT",
        "=" * 72,
        "",
        "── Noise / SNR ─────────────────────────────────────────────────────",
        f"  NIMS effective sigma (across all obs): {snr_report['nims_effective_sigma']:.4f}",
        f"  Synth effective sigma (after smoothing+norm): {snr_report['synth_effective_sigma']:.4f}",
        f"  NIMS / synth sigma ratio: {snr_report['sigma_ratio_nims_over_synth']:.2f}×",
        f"  NIMS mean SNR: {snr_report['nims_mean_snr']:.1f}",
        f"  Synth mean SNR: {snr_report['synth_mean_snr']:.1f}",
    ]
    for enc in NIMS_FILES:
        lines.append(f"  {enc}: sigma={snr_report[f'sigma_{enc}']:.4f}  SNR={snr_report[f'snr_{enc}']:.1f}")

    lines += [
        "",
        "── Long-wave / short-wave mean ratio ───────────────────────────────",
        f"  NIMS   SW (<3.5µm) mean: {snr_report['nims_sw_mean']:.4f}",
        f"  NIMS   LW (>3.5µm) mean: {snr_report['nims_lw_mean']:.4f}",
        f"  NIMS   LW/SW ratio: {snr_report['nims_lw_sw_ratio']:.4f}",
        f"  Synth  SW mean: {snr_report['synth_sw_mean']:.4f}",
        f"  Synth  LW mean: {snr_report['synth_lw_mean']:.4f}",
        f"  Synth  LW/SW ratio: {snr_report['synth_lw_sw_ratio']:.4f}",
        f"  NIMS/synth LW/SW ratio: {snr_report['nims_lw_sw_ratio'] / (snr_report['synth_lw_sw_ratio'] + 1e-12):.2f}×",
        "(>1 means NIMS long-wave is elevated relative to synth — thermal emission signature)",
        "",
        "── PCA gap (top contributors, |n_sigma| > 2) ───────────────────────",
    ]
    top = pca_df[pca_df["n_sigma"].abs() > 2].sort_values("n_sigma", key=abs, ascending=False)
    for _, row in top.head(12).iterrows():
        lines.append(
            f"  {row['enc']}  PC{int(row['pc'])}  shift={row['shift']:+.3f}  "
            f"n_sigma={row['n_sigma']:+.1f}  var_exp={row['var_explained']*100:.1f}%"
        )

    lines += [
        "",
        "── Continuum statistics (short-wave 0.7–3.4 µm) ───────────────────",
    ]
    for _, row in cont_df.iterrows():
        lines.append(
            f"  {row['source']:<18} slope={row['slope_mean']:+.4f}±{row['slope_std']:.4f}  "
            f"curv={row['curv_mean']:+.4f}±{row['curv_std']:.4f}  "
            f"range={row['range_mean']:.4f}±{row['range_std']:.4f}"
        )

    lines += [
        "",
        "── DIAGNOSIS & FORWARD MODEL CORRECTION TARGETS ────────────────────",
    ]

    # Automated diagnosis
    sigma_ratio = snr_report["sigma_ratio_nims_over_synth"]
    lw_sw_nims  = snr_report["nims_lw_sw_ratio"]
    lw_sw_synth = snr_report["synth_lw_sw_ratio"]
    lw_ratio    = lw_sw_nims / (lw_sw_synth + 1e-12)

    if sigma_ratio > 1.5:
        lines.append(f"  [NOISE]  NIMS sigma {sigma_ratio:.1f}× larger than synthetic.")
        lines.append(f"           → Increase effective_noise_sigma in forward model,")
        lines.append(f"             or reduce smoothing_window to match NIMS noise texture.")
    elif sigma_ratio < 0.5:
        lines.append(f"  [NOISE]  NIMS sigma {sigma_ratio:.1f}× SMALLER than synthetic.")
        lines.append(f"           → Reduce noise_sigma or increase smoothing_window.")
    else:
        lines.append(f"  [NOISE]  sigma ratio {sigma_ratio:.2f} — noise level roughly matched.")

    if lw_ratio > 1.3:
        lines.append(f"  [THERMAL] NIMS LW/SW ratio {lw_ratio:.2f}× synthetic.")
        lines.append(f"           → Long-wave NIMS flux elevated (thermal emission at >3.5 µm).")
        lines.append(f"             Recommended fix: truncate to ≤3.5 µm for OOD/model training,")
        lines.append(f"             or add a thermal emission additive term in generate_spectrum.")
    else:
        lines.append(f"  [THERMAL] LW/SW ratio {lw_ratio:.2f} — no strong thermal offset detected.")

    # Identify dominant PCA axes
    dominant_pcs = pca_df[pca_df["n_sigma"].abs() > 3].groupby("pc").agg(
        n_enc=("enc", "count"),
        mean_nsigma=("n_sigma", lambda x: x.abs().mean()),
        var_exp=("var_explained", "first")
    ).sort_values("mean_nsigma", ascending=False)

    if len(dominant_pcs):
        top_pc = int(dominant_pcs.index[0])
        lines.append(f"  [PCA]    Dominant OOD axis: PC{top_pc} "
                     f"(affects {dominant_pcs.iloc[0]['n_enc']}/4 obs, "
                     f"mean |n_sigma|={dominant_pcs.iloc[0]['mean_nsigma']:.1f}, "
                     f"var_exp={dominant_pcs.iloc[0]['var_exp']*100:.1f}%)")
        # PC1 → overall level; PC2+ → shape
        if top_pc == 1:
            lines.append("           → PC1-dominated gap: overall reflectance LEVEL mismatch.")
            lines.append("             Fix: check if NIMS spectra and synth span same dynamic range.")
        else:
            lines.append(f"           → PC{top_pc}-dominated gap: spectral SHAPE mismatch.")
            lines.append("             Fix: review endmember shapes in critical wavelength bands.")

    lines.append("")
    lines.append("=" * 72)

    txt = "\n".join(lines)
    out.write_text(txt, encoding="utf-8")
    sys.stdout.buffer.write(("\n" + txt + "\n").encode("utf-8", errors="replace"))


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading NIMS spectra ...")
    nims, obs_labels = load_nims_all()
    print(f"  NIMS total pixels: {len(nims)}")

    print("Generating synthetic spectra ...")
    synth = load_synth(n=6000, seed=42)
    print(f"  Synthetic: {len(synth)}")

    print("\n[1] Band-level statistics ...")
    bdf = analyse_bandstats(nims, synth, obs_labels)
    bdf.to_csv(QC / "ood_diag_bandstats.csv", index=False)
    plot_bandmeans(bdf, QC / "ood_diag_bandmeans.png")

    print("\n[2] Continuum analysis ...")
    cdf = analyse_continuum(nims, synth, obs_labels, QC / "ood_diag_continuum.png")
    cdf.to_csv(QC / "ood_diag_continuum.csv", index=False)

    print("\n[3] PCA diagnosis ...")
    pca_df = pca_diagnosis(nims, synth, obs_labels, QC / "ood_diag_pca.png")
    pca_df.to_csv(QC / "ood_diag_pca_loadings.csv", index=False)

    print("\n[4] SNR / noise estimation ...")
    snr = snr_report(nims, synth, obs_labels)
    for k, v in snr.items():
        print(f"  {k}: {v:.4f}")

    print("\n[5] Summary report ...")
    write_summary(snr, pca_df, cdf, QC / "ood_diag_summary.txt")


if __name__ == "__main__":
    main()
