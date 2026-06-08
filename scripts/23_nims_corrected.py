#!/usr/bin/env python
"""Forward-model correction and OOD gap reduction for NIMS deployment test.

Diagnosis (script 22) found three root causes for OOD = 29-46×:
  A) Spectral shape: H2O ice absorptions in NIMS are much deeper than
     synthetic proxy endmembers.  After per-pixel min-max normalisation
     NIMS spectra fall to ~0 by 3 um; synthetic stays at 0.7-0.9.
  B) Dynamic range: NIMS E6/G2/E11 use the full [0,1] range; synthetic
     only [0.3, 1.0].
  C) Noise: NIMS effective sigma ~0.027 vs synthetic ~0.008 (3.2x).

Correction strategy:
  1. Wavelength truncation to SW only (test several cuts, select best).
  2. Add a long-wave absorption envelope to synthetic generate_spectrum
     so the LW region decays to match NIMS absorption depth.
  3. Noise recalibration: noise_sigma matched to NIMS target SNR.

This script:
  Step 0 -- OOD scan across wavelength cuts (0.7-2.0/2.5/3.0/3.5/full)
            to find which cut reduces OOD the most.
  Step 1 -- Apply the best correction (wave truncation + noise recal).
  Step 2 -- Re-run full NIMS sanity check on corrected model.
  Step 3 -- Report new OOD distances and re-computed delta-DV values.

Outputs:
  results/qc/ood_wave_scan.csv
  results/qc/ood_wave_scan.png
  results/nims_sanity_check/nims_obs_summary_corrected.csv
  results/qc/ood_corrected_comparison.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ExperimentConfig
from src.forward_model import generate_dataset
from src.modeling import evaluate_setting, precision_at_k, threshold_for_top_k

QC   = ROOT / "results" / "qc"
NIMS = ROOT / "results" / "nims_sanity_check"
QC.mkdir(exist_ok=True)

NIMS_FILES = {
    "E6":  NIMS / "nims_e6_scores.csv",
    "E15": NIMS / "nims_e15_scores.csv",
    "G2":  NIMS / "nims_g2_scores.csv",
    "E11": NIMS / "nims_e11_scores.csv",
}
BASE_CFG = ExperimentConfig(seed=42, noise_condition="harsh",
                            rad_simple_organic_hi=0.130)
WAVE_FULL = BASE_CFG.wavelengths          # 181 pts, 0.7-5.2 um

WAVE_CUTS = [2.0, 2.5, 3.0, 3.5, WAVE_FULL[-1]]  # upper limit in um


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def wave_mask(wave_max_um: float) -> np.ndarray:
    return WAVE_FULL <= wave_max_um + 1e-9


def load_nims_spectra(mask: np.ndarray) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return (spectra, enc_list, per-pixel enc index)."""
    all_cols = [f"spec_{j:03d}" for j in range(181)]
    sel_cols = [c for c, include in zip(all_cols, mask) if include]
    parts, labels = [], []
    enc_names = list(NIMS_FILES.keys())
    for k, (enc, path) in enumerate(NIMS_FILES.items()):
        df = pd.read_csv(path, usecols=sel_cols)
        parts.append(df.to_numpy())
        labels.extend([k] * len(df))
    return np.vstack(parts), enc_names, np.array(labels)


def ood_distances(nims: np.ndarray, synth: np.ndarray,
                  n_components: int = 5) -> np.ndarray:
    """Return per-NIMS-pixel OOD distance (nearest-synth Euclidean in PCA-5 space)."""
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(synth)
    synth_p = pca.transform(synth)
    nims_p  = pca.transform(nims)

    # Vectorised nearest-neighbour search (chunks for memory)
    chunk_size = 500
    dists = np.empty(len(nims_p))
    for start in range(0, len(nims_p), chunk_size):
        end  = min(start + chunk_size, len(nims_p))
        diff = nims_p[start:end, np.newaxis, :] - synth_p[np.newaxis, :, :]
        dists[start:end] = np.sqrt((diff ** 2).sum(-1)).min(axis=1)
    return dists


def synth_dataset(n: int = 4000, seed: int = 42, mask: np.ndarray | None = None,
                  noise_sigma: float | None = None) -> np.ndarray:
    """Generate synthetic spectra, optionally with a custom noise sigma."""
    all_cols = [f"spec_{j:03d}" for j in range(181)]
    if noise_sigma is not None:
        cfg = ExperimentConfig(seed=seed, noise_condition="harsh",
                               noise_sigma=noise_sigma, smoothing_window=9,
                               rad_simple_organic_hi=0.130)
    else:
        cfg = ExperimentConfig(seed=seed, noise_condition="harsh",
                               rad_simple_organic_hi=0.130)
    df = generate_dataset(n=n, rho_geo=0.75, rho_rad=0.75, cfg=cfg)
    arr = df[all_cols].to_numpy()
    if mask is not None:
        arr = arr[:, mask]
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# Step 0 -- wavelength cut scan
# ─────────────────────────────────────────────────────────────────────────────

def step0_wave_scan() -> pd.DataFrame:
    print("=" * 64)
    print("STEP 0 -- OOD scan across wavelength cuts")
    print("=" * 64)

    synth_full = synth_dataset(n=4000, seed=42)
    rows = []
    for wmax in WAVE_CUTS:
        msk = wave_mask(wmax)
        n_bands = msk.sum()
        nims, enc_names, obs_labels = load_nims_spectra(msk)
        s_masked = synth_full[:, msk]

        dists = ood_distances(nims, s_masked)
        per_obs = {}
        for k, enc in enumerate(enc_names):
            sub = dists[obs_labels == k]
            per_obs[enc] = float(sub.mean())

        synth_ref = ood_distances(s_masked[:500], s_masked[500:1000])
        row = {"wave_max_um": wmax, "n_bands": int(n_bands),
               "synth_internal_ood": float(synth_ref.mean())}
        for enc in enc_names:
            row[f"ood_{enc}"] = per_obs[enc]
            row[f"ratio_{enc}"] = per_obs[enc] / (row["synth_internal_ood"] + 1e-9)
        rows.append(row)
        mean_ratio = np.mean([row[f"ratio_{enc}"] for enc in enc_names])
        print(f"  wave_max={wmax:.1f} um  n_bands={n_bands:3d}  "
              f"synth_ref={row['synth_internal_ood']:.2f}  "
              f"mean_ratio={mean_ratio:.1f}x  "
              + "  ".join(f"{enc}={per_obs[enc]:.1f}" for enc in enc_names))

    df = pd.DataFrame(rows)
    df.to_csv(QC / "ood_wave_scan.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 4.5))
    enc_colors = {"E6": "#1565c0", "E15": "#42a5f5", "G2": "#c62828", "E11": "#ef9a9a"}
    for enc in enc_names:
        ax.plot(df["wave_max_um"], df[f"ratio_{enc}"],
                marker="o", color=enc_colors[enc], lw=1.6, label=enc)
    ax.axhline(1, color="gray", ls="--", lw=0.8, label="synth reference (=1)")
    ax.set_xlabel("Wavelength upper limit (µm)")
    ax.set_ylabel("NIMS OOD / synth internal OOD")
    ax.set_title("OOD gap vs wavelength cut\n"
                 "(lower = better alignment with synthetic distribution)")
    ax.legend()
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(QC / "ood_wave_scan.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  plot saved: ood_wave_scan.png")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 -- identify best wave cut + noise recalibration
# ─────────────────────────────────────────────────────────────────────────────

def step1_best_params(scan_df: pd.DataFrame) -> tuple[float, float]:
    enc_names = list(NIMS_FILES.keys())
    scan_df = scan_df.copy()
    scan_df["mean_ratio"] = scan_df[[f"ratio_{e}" for e in enc_names]].mean(axis=1)

    # best wave cut: minimum mean_ratio
    best_row = scan_df.sort_values("mean_ratio").iloc[0]
    best_wmax = float(best_row["wave_max_um"])
    print(f"\n  Best wave cut: {best_wmax} um  (mean OOD ratio {best_row['mean_ratio']:.1f}x)")

    # Estimate noise sigma needed to match NIMS SNR
    # NIMS effective sigma ~ 0.027 (from diagnosis), synthetic after smoothing ~0.008
    # Target: match synth sigma to NIMS sigma (3.2x increase)
    # With 9-band smoothing, injected sigma -> effective sigma = sigma/sqrt(window)
    # window=9 -> factor=3. To get effective sigma=0.027: inject sigma=0.027*3=0.081
    target_sigma = 0.081
    print(f"  Noise correction: sigma {BASE_CFG.effective_noise_sigma:.4f} -> {target_sigma:.4f}")
    print(f"  (target effective sigma ~0.027 to match NIMS SNR)")
    return best_wmax, target_sigma


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 -- full corrected NIMS sanity check
# ─────────────────────────────────────────────────────────────────────────────

NIMS_META = {
    "E6":  {"clon_w": 141.8, "hemisphere": "leading",  "terrain": "chaos",
             "geo_unit": "chm",
             "chaos_proximity": 0.80, "lineament_proximity": 0.70,
             "ridge_proximity": 0.70, "young_terrain": 0.80, "activity_proxy": 0.80,
             "trailing_hemisphere": 0.0, "radiation_exposure": 0.19,
             "sulfur_proxy": 0.15, "rad_mimic_proxy": 0.05},
    "E15": {"clon_w": 105.6, "hemisphere": "leading",  "terrain": "plains",
             "geo_unit": "pr",
             "chaos_proximity": 0.10, "lineament_proximity": 0.20,
             "ridge_proximity": 0.20, "young_terrain": 0.20, "activity_proxy": 0.20,
             "trailing_hemisphere": 0.0, "radiation_exposure": 0.02,
             "sulfur_proxy": 0.02, "rad_mimic_proxy": 0.01},
    "G2":  {"clon_w": 291.1, "hemisphere": "trailing", "terrain": "chaos",
             "geo_unit": "chl",
             "chaos_proximity": 0.75, "lineament_proximity": 0.65,
             "ridge_proximity": 0.65, "young_terrain": 0.75, "activity_proxy": 0.75,
             "trailing_hemisphere": 1.0, "radiation_exposure": 0.97,
             "sulfur_proxy": 0.90, "rad_mimic_proxy": 0.70},
    "E11": {"clon_w": 219.0, "hemisphere": "trailing", "terrain": "plains",
             "geo_unit": "pr",
             "chaos_proximity": 0.10, "lineament_proximity": 0.20,
             "ridge_proximity": 0.20, "young_terrain": 0.20, "activity_proxy": 0.20,
             "trailing_hemisphere": 1.0, "radiation_exposure": 0.81,
             "sulfur_proxy": 0.70, "rad_mimic_proxy": 0.30},
}

GEO_COLS = ["chaos_proximity", "lineament_proximity", "ridge_proximity",
            "young_terrain", "activity_proxy"]
RAD_COLS = ["trailing_hemisphere", "radiation_exposure", "sulfur_proxy", "rad_mimic_proxy"]
NEUTRAL_GEO = {c: 0.5 for c in GEO_COLS}
NEUTRAL_RAD = {c: 0.5 for c in RAD_COLS}


def compute_delta_dv(nims_scores: pd.DataFrame,
                     wave_mask_arr: np.ndarray,
                     noise_sigma: float,
                     n_synth: int = 8000,
                     seed: int = 42) -> list[dict]:
    """Train model on corrected synthetic data, compute ΔDV for 4 NIMS obs."""
    all_spec_cols = [f"spec_{j:03d}" for j in range(181)]
    sel_spec_cols = [c for c, m in zip(all_spec_cols, wave_mask_arr) if m]

    # Training data
    cfg = ExperimentConfig(seed=seed, noise_condition="harsh",
                           noise_sigma=noise_sigma, smoothing_window=9,
                           rad_simple_organic_hi=0.130)
    train_df = generate_dataset(n=n_synth, rho_geo=0.75, rho_rad=0.75, cfg=cfg)

    # Feature sets
    geo_cols = GEO_COLS
    rad_cols = RAD_COLS
    all_feats = sel_spec_cols + geo_cols + rad_cols

    X_all = train_df[all_feats].to_numpy()
    y_all = train_df["y"].to_numpy()
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=0.30, stratify=y_all, random_state=seed)

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("clf",   LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
    ])
    pipe.fit(X_tr, y_tr)

    ap_val = average_precision_score(y_val, pipe.predict_proba(X_val)[:, 1])
    print(f"    val PR-AUC (corrected model, full): {ap_val:.4f}")

    # OOD reference on validation spectra
    val_spec = X_val[:, :len(sel_spec_cols)]
    pca5 = PCA(n_components=5, random_state=42)
    pca5.fit(val_spec)
    val_proj = pca5.transform(val_spec)
    ref_chunk = val_proj[:500]
    ref2_chunk = val_proj[500:1000]
    diff_ref = ref_chunk[:, np.newaxis, :] - ref2_chunk[np.newaxis, :, :]
    synth_internal_ood = float(np.sqrt((diff_ref**2).sum(-1)).min(axis=1).mean())

    # Per-NIMS-obs ΔDV computation
    def build_row(meta: dict, context_mode: str) -> np.ndarray:
        """Build feature row with real or neutral context."""
        spec_vals = nims_scores.get(meta.get("enc", ""), np.zeros(len(sel_spec_cols)))
        if context_mode == "real":
            geo_vals = [meta[c] for c in geo_cols]
            rad_vals = [meta[c] for c in rad_cols]
        elif context_mode == "neut_geo":
            geo_vals = [NEUTRAL_GEO[c] for c in geo_cols]
            rad_vals = [meta[c] for c in rad_cols]
        elif context_mode == "neut_rad":
            geo_vals = [meta[c] for c in geo_cols]
            rad_vals = [NEUTRAL_RAD[c] for c in rad_cols]
        else:  # neut_all
            geo_vals = [NEUTRAL_GEO[c] for c in geo_cols]
            rad_vals = [NEUTRAL_RAD[c] for c in rad_cols]
        return np.array(list(spec_vals) + geo_vals + rad_vals)

    def dv_from_row(row: np.ndarray) -> float:
        row2d = row.reshape(1, -1)
        scaled = pipe.named_steps["scale"].transform(row2d)
        return float(pipe.named_steps["clf"].decision_function(scaled)[0])

    summary_rows = []
    for enc, meta in NIMS_META.items():
        nims_spec_mean = nims_scores[enc]    # mean NIMS spectrum (selected bands)

        r_full     = build_row({**meta, "enc": enc}, "real")
        r_neut_geo = build_row({**meta, "enc": enc}, "neut_geo")
        r_neut_rad = build_row({**meta, "enc": enc}, "neut_rad")
        r_neut_all = build_row({**meta, "enc": enc}, "neut_all")

        dv_full     = dv_from_row(r_full)
        dv_neut_geo = dv_from_row(r_neut_geo)
        dv_neut_rad = dv_from_row(r_neut_rad)
        dv_neut_all = dv_from_row(r_neut_all)

        delta_geo  = dv_full - dv_neut_geo
        delta_rad  = dv_full - dv_neut_rad
        delta_full = dv_full - dv_neut_all

        # OOD for this observation's mean spec
        nims_pca = pca5.transform(nims_spec_mean.reshape(1, -1))
        diff_ood = nims_pca[:, np.newaxis, :] - val_proj[np.newaxis, :500, :]
        ood_dist = float(np.sqrt((diff_ood**2).sum(-1)).min())
        ood_ratio = ood_dist / (synth_internal_ood + 1e-9)

        # S_spec (spectral-only score for mean NIMS pixel)
        r_spec_only = np.array(list(nims_spec_mean) + [0.5] * (len(geo_cols) + len(rad_cols)))
        # use spectral only model
        X_spec = train_df[sel_spec_cols].to_numpy()
        pipe_spec = Pipeline([("scale", StandardScaler()),
                              ("clf",   LogisticRegression(max_iter=2000,
                                                            class_weight="balanced", C=1.0))])
        # build spec-only X
        X_spec_tr, X_spec_val, y_spec_tr, _ = train_test_split(
            X_spec, y_all, test_size=0.30, stratify=y_all, random_state=seed)
        pipe_spec.fit(X_spec_tr, y_spec_tr)
        s_spec = float(pipe_spec.predict_proba(nims_spec_mean.reshape(1, -1))[:, 1])

        row = {
            "enc": enc,
            "hemisphere": meta["hemisphere"],
            "terrain": meta["terrain"],
            "geo_unit": meta["geo_unit"],
            "delta_geo_corrected":  round(delta_geo,  4),
            "delta_rad_corrected":  round(delta_rad,  4),
            "delta_full_corrected": round(delta_full, 4),
            "ood_dist_corrected":   round(ood_dist,   2),
            "ood_ratio_corrected":  round(ood_ratio,  1),
            "s_spec_corrected":     round(s_spec,     4),
        }
        print(f"  {enc}: dv_geo={delta_geo:+.3f}  dv_rad={delta_rad:+.3f}  "
              f"dv_full={delta_full:+.3f}  OOD={ood_dist:.1f} ({ood_ratio:.1f}x)  "
              f"S_spec={s_spec:.4f}")
        summary_rows.append(row)

    return summary_rows, synth_internal_ood, pca5


def step2_corrected_run(best_wmax: float, target_sigma: float) -> None:
    print("\n" + "=" * 64)
    print(f"STEP 2 -- corrected NIMS sanity check")
    print(f"  wave_max={best_wmax} um  noise_sigma={target_sigma:.4f}")
    print("=" * 64)

    mask = wave_mask(best_wmax)
    n_bands = mask.sum()
    all_spec_cols = [f"spec_{j:03d}" for j in range(181)]
    sel_spec_cols = [c for c, m in zip(all_spec_cols, mask) if m]
    print(f"  Using {n_bands} spectral bands (0.7-{best_wmax} um)")

    # Load NIMS mean spectra per obs (selected bands)
    enc_names = list(NIMS_FILES.keys())
    nims_all, _, obs_labels = __import__("scripts.22_ood_diagnosis",
                                          fromlist=["load_nims_spectra"]) if False else (None, None, None)
    # Direct load
    nims_means = {}
    nims_all_arr = []
    obs_idx_arr  = []
    for k, (enc, path) in enumerate(NIMS_FILES.items()):
        df = pd.read_csv(path, usecols=sel_spec_cols)
        arr = df.to_numpy()
        nims_means[enc] = arr.mean(axis=0)
        nims_all_arr.append(arr)
        obs_idx_arr.extend([k] * len(arr))
    nims_all_arr = np.vstack(nims_all_arr)
    obs_labels   = np.array(obs_idx_arr)

    # Compute OOD with corrected synth (noise recal)
    synth_corr = synth_dataset(n=4000, seed=42, mask=mask, noise_sigma=target_sigma)
    dists = ood_distances(nims_all_arr, synth_corr)
    synth_ref = ood_distances(synth_corr[:500], synth_corr[500:1000])
    print(f"\n  OOD distances (corrected model, {n_bands} bands):")
    print(f"  synth internal ref: {synth_ref.mean():.2f}")
    for k, enc in enumerate(enc_names):
        sub   = dists[obs_labels == k]
        ratio = sub.mean() / (synth_ref.mean() + 1e-9)
        print(f"  {enc}: mean={sub.mean():.2f}  ratio={ratio:.1f}x  "
              f"(was {[150.3, 93.7, 148.9, 138.2][k]:.1f})")

    # Delta-DV with corrected model
    print("\n  Computing corrected delta-DV ...")
    summary_rows, synth_ood_ref, pca5 = compute_delta_dv(
        nims_means, mask, noise_sigma=target_sigma)

    # Load original summary for comparison
    orig_path = NIMS / "nims_obs_summary.csv"
    orig_df = pd.read_csv(orig_path) if orig_path.exists() else pd.DataFrame()

    # Save corrected summary
    out_df = pd.DataFrame(summary_rows)
    # rename "enc" to "encounter" to match original summary column
    out_df = out_df.rename(columns={"enc": "encounter"})
    if len(orig_df):
        merged = orig_df.merge(out_df, on="encounter", suffixes=("_orig", ""))
    else:
        merged = out_df
    merged.to_csv(NIMS / "nims_obs_summary_corrected.csv", index=False)
    print(f"  saved nims_obs_summary_corrected.csv")

    # Comparison plot
    _plot_comparison(summary_rows, orig_df, synth_ref.mean())


def _plot_comparison(rows: list[dict], orig_df: pd.DataFrame,
                     synth_ref_ood: float) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    enc_colors = {"E6": "#1565c0", "E15": "#42a5f5", "G2": "#c62828", "E11": "#ef9a9a"}
    encs = [r["enc"] for r in rows]

    dv_pairs = [("delta_geo", "delta_geo_corrected",   r"$\Delta DV_{\rm geo}$"),
                ("delta_rad", "delta_rad_corrected",   r"$\Delta DV_{\rm rad}$"),
                ("delta_full", "delta_full_corrected", r"$\Delta DV_{\rm full}$")]

    for ax, (orig_col, corr_col, label) in zip(axes, dv_pairs):
        x     = np.arange(len(encs))
        width = 0.35
        orig_vals = orig_df[orig_col].values if orig_col in orig_df.columns else [0]*4
        corr_vals = [r[corr_col] for r in rows]
        colors    = [enc_colors[e] for e in encs]

        bars_o = ax.bar(x - width/2, orig_vals, width=width, alpha=0.45,
                        color=colors, edgecolor=colors, lw=1.2, label="original (181 bands)")
        bars_c = ax.bar(x + width/2, corr_vals, width=width, alpha=0.85,
                        color=colors, edgecolor=colors, lw=1.2, hatch="////",
                        label="corrected (short-wave)")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(encs, fontsize=8.5)
        ax.set_title(label)
        ax.set_ylabel("ΔDV")
        if ax == axes[0]:
            ax.legend(fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("ΔDV: original vs corrected forward model\n"
                 "(hatched bars = corrected short-wave model)", fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(QC / "ood_corrected_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved ood_corrected_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    scan_df = step0_wave_scan()
    best_wmax, target_sigma = step1_best_params(scan_df)
    step2_corrected_run(best_wmax, target_sigma)
    print("\nDone.")


if __name__ == "__main__":
    main()
