#!/usr/bin/env python
"""Generate all paper figures for the Europa AI triage paper.

Fig 2 -- Representative synthetic spectra by class with absorption feature annotations.
Fig 3 -- Ambiguous subset effect: score histograms + PR curves (spectral-only vs full).
Fig 4 -- rho sweep heatmap: delta PR-AUC as a function of (rho_geo, rho_rad).
Fig 5 -- NIMS 2x2 deployment sanity check figure.

Outputs: results/paper/figures/fig{2,3,4,5}.pdf  (also .png at 300 dpi)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ExperimentConfig
from src.forward_model import generate_spectrum
from src.endmember_loader import load_processed_endmembers

FIGDIR = ROOT / "results" / "paper" / "figures"
RESULTS = ROOT / "results"
FIGDIR.mkdir(parents=True, exist_ok=True)

# ── color/style constants ────────────────────────────────────────────────────
CLASS_COLORS = {
    "ocean_organic":            "#1565c0",
    "ocean_nonorganic":         "#42a5f5",
    "radiation_mimic":          "#c62828",
    "exogenic_complex_organic": "#6d4c41",
}
CLASS_LABELS = {
    "ocean_organic":            r"ocean organic ($z_1$)",
    "ocean_nonorganic":         r"ocean non-organic ($z_2$)",
    "radiation_mimic":          r"radiation mimic ($z_3$)",
    "exogenic_complex_organic": r"exogenic complex organic ($z_4$)",
}
SETTING_COLORS = {"spectral_only": "#e65100", "full": "#2e7d32"}
SETTING_LABELS = {"spectral_only": "spectral-only", "full": "full (spectral + spatial)"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
})


# ═══════════════════════════════════════════════════════════════════════════
# Fig 2 — Representative synthetic spectra
# ═══════════════════════════════════════════════════════════════════════════

def fig2_spectra() -> None:
    print("[Fig 2] Generating synthetic spectra ...")
    cfg = ExperimentConfig(seed=0, noise_condition="moderate")
    wave = cfg.wavelengths          # 181 points, 0.7–5.2 µm
    lib  = load_processed_endmembers(wave)

    n_reps = 40
    classes = ["ocean_organic", "ocean_nonorganic", "radiation_mimic", "exogenic_complex_organic"]

    # Generate n_reps spectra per class, collect mean ± std
    means, stds = {}, {}
    for z in classes:
        specs = []
        for seed in range(n_reps):
            rng = np.random.default_rng(seed + 1000)
            spec, _ = generate_spectrum(z, rng, cfg, lib)
            specs.append(spec)
        arr = np.vstack(specs)
        means[z] = arr.mean(axis=0)
        stds[z]  = arr.std(axis=0)

    # Compute spectral OVL between ocean_organic and radiation_mimic
    # via histogram intersection on mean spectra (as rough visual OVL)
    # Use score OVL from predictions instead (displayed as annotation).
    OVL_DISPLAY = 0.825  # from paper main results harsh+highov condition

    # Key absorption feature annotations (wavelength_um, label, y_frac)
    FEATURES = [
        (1.50, r"H$_2$O 1.5 $\mu$m", 0.79),
        (2.00, r"H$_2$O 2.0 $\mu$m", 0.74),
        (3.00, r"H$_2$O 3.0 $\mu$m", 0.55),
        (3.50, r"H$_2$O$_2$/C-H 3.5 $\mu$m", 0.35),
        (3.78, r"H$_2$SO$_4 \cdot n$H$_2$O", 0.22),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    ls_map = {
        "ocean_organic":            "-",
        "ocean_nonorganic":         "--",
        "radiation_mimic":          "-.",
        "exogenic_complex_organic": ":",
    }
    lw_map = {
        "ocean_organic":  2.0,
        "ocean_nonorganic": 1.6,
        "radiation_mimic": 2.0,
        "exogenic_complex_organic": 1.6,
    }

    for z in classes:
        c  = CLASS_COLORS[z]
        ls = ls_map[z]
        lw = lw_map[z]
        ax.plot(wave, means[z], color=c, ls=ls, lw=lw,
                label=CLASS_LABELS[z], zorder=3)
        ax.fill_between(wave,
                        means[z] - stds[z],
                        means[z] + stds[z],
                        color=c, alpha=0.10, zorder=2)

    # Shade ambiguous zone (z1 vs z3 overlap) between 3.2–3.8 µm
    ax.axvspan(3.2, 3.9, color="#ffd54f", alpha=0.18, label=r"spectral ambiguity zone ($z_1$/$z_3$)", zorder=1)

    # Absorption feature arrows
    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    for wf, label, yfrac in FEATURES:
        y_arrow = ymin + yfrac * yrange
        ax.annotate(
            label,
            xy=(wf, y_arrow),
            xytext=(wf + 0.08, y_arrow + 0.022 * yrange),
            fontsize=7.5,
            color="#333333",
            arrowprops=dict(arrowstyle="-|>", color="#555555",
                            lw=0.8, mutation_scale=8),
        )
        ax.axvline(wf, color="#aaaaaa", ls=":", lw=0.7, zorder=0)

    # OVL annotation
    ax.text(
        0.98, 0.97,
        f"spectral score OVL($z_1$,$z_3$) = {OVL_DISPLAY:.3f}\n(harsh noise, high overlap condition)",
        ha="right", va="top", transform=ax.transAxes,
        fontsize=8, color="#555555",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85),
    )

    ax.set_xlabel(r"Wavelength ($\mu$m)")
    ax.set_ylabel("Reflectance (normalised, a.u.)")
    ax.set_xlim(0.7, 5.2)
    ax.set_title("Fig. 2 — Representative synthetic spectra by hidden source class\n"
                 r"(mean ± 1$\sigma$ over 40 realisations, moderate noise)", pad=8)
    ax.legend(loc="upper right", framealpha=0.92, ncol=1)
    ax.set_yticks([])

    fig.tight_layout()
    _save(fig, "fig2_spectra")
    print("       done.")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 3 — Ambiguous subset effect
# ═══════════════════════════════════════════════════════════════════════════

def fig3_ambiguous() -> None:
    print("[Fig 3] Building ambiguous subset figure ...")

    preds_path = RESULTS / "experiment1_predictions_harsh_highov_v1.csv"
    if not preds_path.exists():
        print(f"       SKIP: {preds_path} not found.")
        return

    preds = pd.read_csv(preds_path)

    # Identify ambiguous sample_ids using spectral_only scores
    spec_preds = preds[preds["setting"] == "spectral_only"].copy()
    AMBIG_LO, AMBIG_HI = 0.4, 0.7
    ambig_ids = set(
        spec_preds.loc[(spec_preds["score"] >= AMBIG_LO) & (spec_preds["score"] <= AMBIG_HI),
                       "sample_id"]
    )
    full_preds = preds[preds["setting"] == "full"].copy()

    spec_ambig = spec_preds[spec_preds["sample_id"].isin(ambig_ids)].copy()
    full_ambig = full_preds[full_preds["sample_id"].isin(ambig_ids)].copy()

    n_ambig = len(ambig_ids)
    n_pos   = int(spec_ambig["y"].sum())
    print(f"       ambig n={n_ambig}, positives={n_pos}")

    # PR curves
    prec_s, rec_s, _ = precision_recall_curve(spec_ambig["y"], spec_ambig["score"])
    prec_f, rec_f, _ = precision_recall_curve(full_ambig["y"], full_ambig["score"])
    auc_s = average_precision_score(spec_ambig["y"], spec_ambig["score"])
    auc_f = average_precision_score(full_ambig["y"], full_ambig["score"])

    # Score distributions by true class — ALL samples (for Panel A overview)
    oc_all = spec_preds.loc[spec_preds["z"] == "ocean_organic",   "score"]
    rm_all = spec_preds.loc[spec_preds["z"] == "radiation_mimic", "score"]
    # Ambiguous subset re-scored by full model
    oc_f = full_ambig.loc[full_ambig["z"] == "ocean_organic",   "score"]
    rm_f = full_ambig.loc[full_ambig["z"] == "radiation_mimic", "score"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))

    # ── panel A: full score distribution, spectral-only ──────────────────
    ax = axes[0]
    bins = np.linspace(0, 1, 28)
    ax.hist(oc_all, bins=bins, color=CLASS_COLORS["ocean_organic"],
            alpha=0.65, label=r"$z_1$ ocean organic", density=True)
    ax.hist(rm_all, bins=bins, color=CLASS_COLORS["radiation_mimic"],
            alpha=0.65, label=r"$z_3$ radiation mimic", density=True)
    ax.axvspan(AMBIG_LO, AMBIG_HI, color="#ffd54f", alpha=0.30,
               label=f"ambiguous zone [0.4–0.7]\n(n={n_ambig})")
    ax.set_xlabel("Score (spectral-only)")
    ax.set_ylabel("Density")
    ax.set_title("(A) Spectral-only — full score distribution\n"
                 r"(all $z_1$/$z_3$ samples; ambiguous zone shaded)")
    ax.legend(fontsize=7.5)

    # ── panel B: full model re-scores the ambiguous subset ───────────────
    ax = axes[1]
    ax.hist(oc_f, bins=bins, color=CLASS_COLORS["ocean_organic"],
            alpha=0.65, label=r"$z_1$ ocean organic", density=True)
    ax.hist(rm_f, bins=bins, color=CLASS_COLORS["radiation_mimic"],
            alpha=0.65, label=r"$z_3$ radiation mimic", density=True)
    ax.axvspan(AMBIG_LO, AMBIG_HI, color="#ffd54f", alpha=0.20,
               label="spectral-only ambiguous zone\n(reference)")
    ax.set_xlabel("Score (full model)")
    ax.set_ylabel("Density")
    ax.set_title("(B) Full model — ambiguous-subset samples re-scored\n"
                 r"(spatial context disperses $z_1$/$z_3$ toward poles)")
    ax.legend(fontsize=7.5)

    # ── panel C: PR curves ───────────────────────────────────────────────
    ax = axes[2]
    baseline = n_pos / n_ambig
    ax.axhline(baseline, color="#999", ls=":", lw=1.0, label=f"random (prevalence {baseline:.2f})")
    ax.plot(rec_s, prec_s, color=SETTING_COLORS["spectral_only"], lw=1.8,
            label=f"spectral-only  AP={auc_s:.3f}")
    ax.plot(rec_f, prec_f, color=SETTING_COLORS["full"],          lw=1.8,
            label=f"full model     AP={auc_f:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("(C) PR curve — ambiguous subset")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)

    fig.suptitle(
        "Fig. 3 — Spatial context is most effective on ambiguous candidates "
        "(spectral-only score 0.4–0.7, harsh+high-overlap condition, seed 3421)",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    _save(fig, "fig3_ambiguous")
    print("       done.")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 4 — rho sweep heatmap
# ═══════════════════════════════════════════════════════════════════════════

def fig4_rho_heatmap() -> None:
    print("[Fig 4] Building rho sweep heatmap ...")

    sweep_path = RESULTS / "prior_sweep_metrics_leakage_fix_v1_logreg.csv"
    if not sweep_path.exists():
        print(f"       SKIP: {sweep_path} not found.")
        return

    df = pd.read_csv(sweep_path)
    pivot = df.pivot(index="rho_geo", columns="rho_rad", values="delta_pr_auc")
    rho_vals = sorted(pivot.index.tolist())
    pivot = pivot.loc[rho_vals, sorted(pivot.columns.tolist())]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # ── panel A: delta PR-AUC heatmap ───────────────────────────────────
    ax = axes[0]
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto",
                   vmin=0, vmax=pivot.values.max(), origin="lower")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\Delta$PR-AUC (full $-$ spectral-only)", fontsize=9)

    ax.set_xticks(range(len(rho_vals)))
    ax.set_yticks(range(len(rho_vals)))
    ax.set_xticklabels([f"{v:.2f}" for v in sorted(pivot.columns)])
    ax.set_yticklabels([f"{v:.2f}" for v in rho_vals])
    ax.set_xlabel(r"$\rho_{\rm rad}$ (radiation context strength)")
    ax.set_ylabel(r"$\rho_{\rm geo}$ (geo context strength)")
    ax.set_title(r"(A) $\Delta$PR-AUC = full $-$ spectral-only")

    for i in range(len(rho_vals)):
        for j in range(len(rho_vals)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=7.5, color="black" if val < 0.04 else "white")

    # ── panel B: PR-AUC spectral vs full scatter ─────────────────────────
    ax = axes[1]
    scatter = ax.scatter(
        df["pr_auc_spectral"], df["pr_auc_full"],
        c=df["rho_geo"] + df["rho_rad"],
        cmap="plasma", s=50, alpha=0.8, zorder=3,
    )
    cbar2 = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar2.set_label(r"$\rho_{\rm geo} + \rho_{\rm rad}$", fontsize=9)
    lo = min(df["pr_auc_spectral"].min(), df["pr_auc_full"].min()) - 0.01
    hi = max(df["pr_auc_spectral"].max(), df["pr_auc_full"].max()) + 0.01
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="no gain")
    ax.set_xlabel("PR-AUC spectral-only")
    ax.set_ylabel("PR-AUC full model")
    ax.set_title("(B) Gain at each (rho_geo, rho_rad) setting")
    ax.legend(fontsize=8)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)

    fig.suptitle(
        r"Fig. 4 — Spatial-context gain ($\Delta$PR-AUC) across "
        r"$(\rho_{\rm geo}, \rho_{\rm rad})$ grid "
        "(logreg, leakage-fix condition, 5 seeds each)",
        fontsize=10, y=1.01,
    )
    fig.tight_layout()
    _save(fig, "fig4_rho_heatmap")
    print("       done.")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 5 — NIMS 2×2 deployment
# ═══════════════════════════════════════════════════════════════════════════

# Observation metadata (West longitude)
NIMS_OBS = [
    {"enc": "E6",  "clon_w": 141.8, "hemisphere": "leading",  "terrain": "chaos",
     "geo_unit": "chm", "n": 111,   "dv_geo": +3.473, "dv_rad": +0.182, "dv_full": +3.655},
    {"enc": "E15", "clon_w": 105.6, "hemisphere": "leading",  "terrain": "plains",
     "geo_unit": "pr",  "n": 865,   "dv_geo": -4.445, "dv_rad": +0.284, "dv_full": -4.161},
    {"enc": "G2",  "clon_w": 291.1, "hemisphere": "trailing", "terrain": "chaos",
     "geo_unit": "chl", "n":  53,   "dv_geo": +2.786, "dv_rad": -2.393, "dv_full": +0.393},
    {"enc": "E11", "clon_w": 219.0, "hemisphere": "trailing", "terrain": "plains",
     "geo_unit": "pr",  "n": 3431,  "dv_geo": -4.445, "dv_rad": -2.304, "dv_full": -6.749},
]


def _obs_color(obs: dict) -> str:
    return "#1565c0" if obs["hemisphere"] == "leading" else "#c62828"


def _obs_hatch(obs: dict) -> str:
    return "" if obs["terrain"] == "chaos" else "////"


def fig5_nims() -> None:
    print("[Fig 5] Building NIMS 2×2 figure ...")

    fig = plt.figure(figsize=(12, 5))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1.05, 0.95], wspace=0.10)
    ax_map = fig.add_subplot(gs[0])
    ax_dv  = fig.add_subplot(gs[1])

    # ── panel A: Europa longitude map ───────────────────────────────────
    _draw_europa_map(ax_map)

    # ── panel B: ΔDV bar chart ──────────────────────────────────────────
    _draw_dv_bars(ax_dv)

    # Shared legend
    legend_elems = [
        mpatches.Patch(fc="#1565c0", label="leading hemisphere"),
        mpatches.Patch(fc="#c62828", label="trailing hemisphere"),
        mpatches.Patch(fc="#bbbbbb", hatch="",    label="chaos terrain (chm/chl)"),
        mpatches.Patch(fc="#bbbbbb", hatch="////", label="plains terrain (pr)"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.07), fontsize=8.5, framealpha=0.9)

    fig.suptitle(
        "Fig. 5 — Galileo/NIMS 2×2 spatial-context deployment sanity check\n"
        r"(leading/trailing $\times$ chaos/plains; $\Delta DV$ decomposition)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig5_nims")
    print("       done.")


def _draw_europa_map(ax: plt.Axes) -> None:
    """Simple cylindrical-projection Europa longitude map with 4 obs markers."""
    LON_MIN, LON_MAX = 0, 360

    # Background: leading (0–180°W) vs trailing (180–360°W)
    ax.axvspan(0,   180, color="#bbdefb", alpha=0.40, zorder=0, label="leading")
    ax.axvspan(180, 360, color="#ffcdd2", alpha=0.40, zorder=0, label="trailing")

    # Radiation apex and nadir
    for lon, lbl, c in [(270, "radiation\napex (270°W)", "#c62828"),
                        ( 90, "radiation\nnadir (90°W)",  "#1565c0")]:
        ax.axvline(lon, color=c, ls="--", lw=1.0, alpha=0.6, zorder=1)
        ax.text(lon, 18, lbl, ha="center", va="bottom", fontsize=7.5,
                color=c, style="italic")

    # Leading/trailing boundary
    ax.axvline(180, color="#888888", ls="-", lw=0.8, alpha=0.5)

    # Plot 4 observations
    LAT_JITTER = {0: 0, 1: -6, 2: 0, 3: 6}     # slight lat offset for readability
    for k, obs in enumerate(NIMS_OBS):
        lon = obs["clon_w"]
        lat = LAT_JITTER[k]
        c   = _obs_color(obs)
        mk  = "o" if obs["terrain"] == "chaos" else "D"
        ec  = "white"
        ax.plot(lon, lat, marker=mk, ms=11, color=c, mec=ec, mew=1.4,
                zorder=5, clip_on=False)
        # Label offset
        dx = +8 if lon < 180 else -8
        dy = +4 if lat >= 0 else -5
        ax.annotate(
            f"{obs['enc']}\n({obs['geo_unit']}, n={obs['n']:,})",
            xy=(lon, lat), xytext=(lon + dx, lat + dy + 2),
            fontsize=7.5, color=c, ha="center",
            arrowprops=dict(arrowstyle="-", color=c, lw=0.8),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=c, alpha=0.85, lw=0.8),
        )

    # Decoration
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(-25, 25)
    ax.set_xlabel("West Longitude (°W)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("(A) Observation footprints on Europa\n"
                 "● chaos terrain   ◆ plains terrain")
    ax.set_xticks(range(0, 361, 45))
    ax.set_yticks([-20, -10, 0, 10, 20])
    ax.axhline(0, color="#aaaaaa", lw=0.6, ls=":")

    # Hemisphere labels
    ax.text( 90, -22, "leading hemisphere",  ha="center", fontsize=8.5,
             color="#1565c0", weight="bold")
    ax.text(270, -22, "trailing hemisphere", ha="center", fontsize=8.5,
             color="#c62828", weight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _draw_dv_bars(ax: plt.Axes) -> None:
    """Grouped bar chart: ΔDV_geo, ΔDV_rad, ΔDV_full for 4 observations."""
    metrics = ["dv_geo", "dv_rad", "dv_full"]
    labels  = [r"$\Delta DV_{\rm geo}$", r"$\Delta DV_{\rm rad}$",
               r"$\Delta DV_{\rm full}$"]
    n_met   = len(metrics)
    n_obs   = len(NIMS_OBS)
    w       = 0.18          # bar width
    offsets = np.linspace(-(n_met - 1) * w / 2, (n_met - 1) * w / 2, n_met)

    x = np.arange(n_obs)
    for j, (met, lbl) in enumerate(zip(metrics, labels)):
        vals    = [obs[met] for obs in NIMS_OBS]
        colors  = [_obs_color(obs) for obs in NIMS_OBS]
        hatches = [_obs_hatch(obs) for obs in NIMS_OBS]
        bars = ax.bar(x + offsets[j], vals, width=w,
                      color=[mcolors.to_rgba(c, alpha=0.70) for c in colors],
                      edgecolor=colors, linewidth=1.2,
                      hatch=hatches, label=lbl, zorder=3)
        # Value labels just outside each bar
        for bar, v in zip(bars, vals):
            pad = 0.22 if abs(v) > 0.5 else 0.10
            ypos = v + pad * np.sign(v)
            va   = "bottom" if v >= 0 else "top"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    ypos, f"{v:+.2f}", ha="center", va=va,
                    fontsize=6.5, color="#222222")

    ax.axhline(0, color="black", lw=0.8)
    hemi_abbr = {"leading": "Lead.", "trailing": "Trail."}
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{o['enc']}\n{hemi_abbr[o['hemisphere']]}\n{o['geo_unit']}" for o in NIMS_OBS],
        fontsize=8.5,
    )
    ax.set_ylabel(r"$\Delta DV$ (decision value shift from neutral baseline)")
    ax.set_title("(B) Decision-value decomposition\n"
                 r"($\Delta DV_{\rm geo}$, $\Delta DV_{\rm rad}$, $\Delta DV_{\rm full}$)")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_xlim(-0.5, n_obs - 0.5)
    yabs = max(abs(v) for obs in NIMS_OBS for v in [obs["dv_geo"], obs["dv_rad"], obs["dv_full"]])
    ylim = yabs + 1.8
    ax.set_ylim(-ylim, ylim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ═══════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════

def _save(fig: plt.Figure, name: str) -> None:
    for ext in ("pdf", "png"):
        p = FIGDIR / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        print(f"       saved  {p.relative_to(ROOT)}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print(f"Output directory: {FIGDIR}")
    fig2_spectra()
    fig3_ambiguous()
    fig4_rho_heatmap()
    fig5_nims()
    print("\nAll figures done.")


if __name__ == "__main__":
    main()
