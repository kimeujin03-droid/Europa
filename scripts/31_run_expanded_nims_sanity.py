#!/usr/bin/env python
"""Run expanded NIMS sanity scoring on preselected observation cohorts.

Selection is read from data/manifest/selected_nims_observations_{8,12}.csv.
Those manifests encode only pre-deployment quality criteria; model scores are
not used to choose observations.
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
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
sys.path.insert(0, str(ROOT))

from src.config import ExperimentConfig


def load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nims20 = load_script_module("nims20", ROOT / "scripts" / "20_nims_sanity_check.py")
stage25 = load_script_module("stage25", ROOT / "scripts" / "25_run_nims_with_stage_endmembers.py")

OUT_DIR = ROOT / "results" / "nims_expanded_sanity"
RAW_DIR = ROOT / "data" / "raw" / "nims_pds"
SPEC_COLS = [f"spec_{j:03d}" for j in range(181)]
GEO_COLS = stage25.GEO_COLS
RAD_COLS = stage25.RAD_COLS
NEUTRAL_GEO = stage25.NEUTRAL_GEO
NEUTRAL_RAD = stage25.NEUTRAL_RAD


def safe_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_").lower()


def iqr(values: np.ndarray) -> float:
    q75, q25 = np.nanpercentile(values, [75, 25])
    return float(q75 - q25)


def ood_gate(median_ood_ratio: float) -> str:
    if median_ood_ratio <= 10.0:
        return "score_reportable"
    if median_ood_ratio <= 25.0:
        return "context_only"
    return "ood_only"


def context_from_cube(qub_bytes: bytes, n: int, gdf: object | None, terrain_hint: str) -> tuple[pd.DataFrame, pd.DataFrame, float, str]:
    label_text, _ = nims20.extract_attached_label(qub_bytes)
    clon_w = nims20.extract_center_lon(label_text)
    if clon_w is None:
        raise ValueError("Could not parse observation center longitude")

    q_real = nims20.assign_radiation_context_obs(clon_w, n)
    if gdf is not None:
        try:
            g_real, geo_unit = nims20.geo_lookup_center(clon_w, n, gdf)
            return g_real.reset_index(drop=True), q_real.reset_index(drop=True), float(clon_w), str(geo_unit)
        except Exception as exc:
            print(f"  [WARN] geologic lookup failed, using terrain hint: {exc}")

    unit = "chm" if terrain_hint == "chaos" else "pr"
    g_vals = nims20.UNIT_GFEATURES.get(unit, nims20.UNIT_GFEATURES["__default__"])
    return pd.DataFrame([g_vals] * n), q_real.reset_index(drop=True), float(clon_w), unit


def build_context(g_real: pd.DataFrame, q_real: pd.DataFrame, mode: str) -> pd.DataFrame:
    n = len(g_real)
    if mode == "real":
        return pd.concat([g_real[GEO_COLS].reset_index(drop=True), q_real[RAD_COLS].reset_index(drop=True)], axis=1)
    if mode == "neutral_all":
        return pd.DataFrame([{**NEUTRAL_GEO, **NEUTRAL_RAD}] * n)
    if mode == "neutral_geo":
        return pd.concat([pd.DataFrame([NEUTRAL_GEO] * n), q_real[RAD_COLS].reset_index(drop=True)], axis=1)
    if mode == "neutral_rad":
        return pd.concat([g_real[GEO_COLS].reset_index(drop=True), pd.DataFrame([NEUTRAL_RAD] * n)], axis=1)
    raise ValueError(mode)


def score_observation(
    row: pd.Series,
    mask: np.ndarray,
    spec_model,
    full_model,
    val_df: pd.DataFrame,
    gdf: object | None,
) -> tuple[pd.DataFrame, dict]:
    path = RAW_DIR / row["file"]
    qub_bytes = nims20.download_bytes(str(row["url"]), path)
    df_nims, _ = nims20.read_nims_qub(qub_bytes, ExperimentConfig().wavelengths)
    selected = [col for col, include in zip(SPEC_COLS, mask) if include]
    spec = df_nims[selected].to_numpy()
    n = len(df_nims)

    g_real, q_real, clon_w, geo_unit = context_from_cube(qub_bytes, n, gdf, row["terrain"])
    ctx_real = build_context(g_real, q_real, "real")
    ctx_neut_all = build_context(g_real, q_real, "neutral_all")
    ctx_neut_geo = build_context(g_real, q_real, "neutral_geo")
    ctx_neut_rad = build_context(g_real, q_real, "neutral_rad")

    def full_score(ctx: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = np.hstack([spec, ctx[GEO_COLS + RAD_COLS].to_numpy()])
        return full_model.predict_proba(x)[:, 1], full_model.decision_function(x)

    p_spec = spec_model.predict_proba(spec)[:, 1]
    dv_spec = spec_model.decision_function(spec)
    p_full, dv_full = full_score(ctx_real)
    _, dv_neut_all = full_score(ctx_neut_all)
    _, dv_neut_geo = full_score(ctx_neut_geo)
    _, dv_neut_rad = full_score(ctx_neut_rad)

    synth_spec = val_df[selected].to_numpy()
    ood_dist, ood_ref = stage25.ood_distances(spec, synth_spec)
    ood_ratio = ood_dist / (ood_ref + 1e-9)

    scored = df_nims[selected + ["lat_w", "lon_w"]].copy()
    scored["group"] = row["group"]
    scored["obs_id"] = row["obs_id"]
    scored["hemisphere"] = row["hemisphere"]
    scored["terrain"] = row["terrain"]
    scored["geo_unit"] = geo_unit
    scored["clon_w"] = clon_w
    scored["p_spec"] = p_spec
    scored["p_full"] = p_full
    scored["dv_spec"] = dv_spec
    scored["dv_full"] = dv_full
    scored["delta_geo"] = dv_full - dv_neut_geo
    scored["delta_rad"] = dv_full - dv_neut_rad
    scored["delta_full"] = dv_full - dv_neut_all
    scored["ood_dist"] = ood_dist
    scored["ood_ratio"] = ood_ratio

    median_ood = float(np.nanmedian(ood_ratio))
    gate = ood_gate(median_ood)
    summary = {
        "group": row["group"],
        "obs_id": row["obs_id"],
        "volume": row["volume"],
        "file": row["file"],
        "product": row["product"],
        "selection_source": row["selection_source"],
        "hemisphere": row["hemisphere"],
        "terrain": row["terrain"],
        "geo_unit": geo_unit,
        "clon_w": clon_w,
        "n_pixels": n,
        "median_p_spec": float(np.nanmedian(p_spec)),
        "iqr_p_spec": iqr(p_spec),
        "median_p_full": float(np.nanmedian(p_full)),
        "iqr_p_full": iqr(p_full),
        "median_dv_spec": float(np.nanmedian(dv_spec)),
        "median_dv_full": float(np.nanmedian(dv_full)),
        "iqr_dv_full": iqr(dv_full),
        "median_delta_geo": float(np.nanmedian(scored["delta_geo"])),
        "median_delta_rad": float(np.nanmedian(scored["delta_rad"])),
        "median_delta_full": float(np.nanmedian(scored["delta_full"])),
        "median_ood_ratio": median_ood,
        "iqr_ood_ratio": iqr(ood_ratio),
        "synth_internal_ood": float(ood_ref),
        "ood_gate": gate,
        "report_p_full": float(np.nanmedian(p_full)) if gate == "score_reportable" else np.nan,
        "screen_ood_ratio_0p7_2p0": row.get("screen_ood_ratio_0p7_2p0", np.nan),
    }
    return scored, summary


def save_summary_plot(summary: pd.DataFrame, out_path: Path, title: str) -> None:
    order = ["leading_chaos", "leading_plains", "trailing_chaos", "trailing_plains"]
    colors = {"score_reportable": "#2E7D32", "context_only": "#B26A00", "ood_only": "#8A1C1C"}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    for ax, col, ylabel in [
        (axes[0], "median_ood_ratio", "Median OOD ratio"),
        (axes[1], "median_delta_full", "Median full-context DV shift"),
        (axes[2], "median_p_full", "Median model score"),
    ]:
        x = np.arange(len(summary))
        row_colors = [colors[g] for g in summary["ood_gate"]]
        ax.bar(x, summary[col], color=row_colors, edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(summary["obs_id"], rotation=55, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        for idx, group in enumerate(summary["group"]):
            ax.text(idx, ax.get_ylim()[1] * 0.98, group.replace("_", "\n"), ha="center", va="top", fontsize=6)
    axes[0].axhline(10, color="#2E7D32", ls="--", lw=1)
    axes[0].axhline(25, color="#8A1C1C", ls="--", lw=1)
    fig.suptitle(title)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def run_unsupervised(summary: pd.DataFrame, pixel_frames: list[pd.DataFrame], selected: list[str], cohort_tag: str) -> None:
    rows = []
    spectra = []
    for obs_id, pix in zip(summary["obs_id"], pixel_frames):
        spectra.append(pix[selected].mean(axis=0).to_numpy())
        rows.append({
            "obs_id": obs_id,
            "group": pix["group"].iloc[0],
            "hemisphere": pix["hemisphere"].iloc[0],
            "terrain": pix["terrain"].iloc[0],
        })
    meta = pd.DataFrame(rows)
    x = np.vstack(spectra)
    x_scaled = StandardScaler().fit_transform(x)

    pca = PCA(n_components=min(4, len(x) - 1, x.shape[1]), random_state=42)
    pc = pca.fit_transform(x_scaled)
    for i in range(pc.shape[1]):
        meta[f"pc{i + 1}"] = pc[:, i]

    nmf_components = min(4, len(x), x.shape[1])
    nmf = NMF(n_components=nmf_components, init="nndsvda", random_state=42, max_iter=5000, tol=1e-5)
    nmf_score = nmf.fit_transform(np.clip(x, 0, None))
    for i in range(nmf_score.shape[1]):
        meta[f"nmf{i + 1}"] = nmf_score[:, i]

    metrics = []
    for label in ["hemisphere", "terrain", "group"]:
        labels = meta[label].to_numpy()
        if len(np.unique(labels)) > 1 and len(np.unique(labels)) < len(labels):
            metrics.append({
                "cohort": cohort_tag,
                "embedding": "PCA_observation_mean",
                "label": label,
                "silhouette": float(silhouette_score(pc[:, : min(2, pc.shape[1])], labels)),
            })
            metrics.append({
                "cohort": cohort_tag,
                "embedding": "NMF_observation_mean",
                "label": label,
                "silhouette": float(silhouette_score(nmf_score[:, : min(2, nmf_score.shape[1])], labels)),
            })

    meta.to_csv(OUT_DIR / f"{cohort_tag}_unsupervised_observation_scores.csv", index=False)
    pd.DataFrame(metrics).to_csv(OUT_DIR / f"{cohort_tag}_unsupervised_observation_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 5.0), constrained_layout=True)
    markers = {"chaos": "o", "plains": "s"}
    colors = {"leading": "#1F77B4", "trailing": "#D62728"}
    for _, row in meta.iterrows():
        ax.scatter(row["pc1"], row["pc2"], marker=markers[row["terrain"]], color=colors[row["hemisphere"]], s=70)
        ax.text(row["pc1"], row["pc2"], f" {row['obs_id']}", fontsize=8, va="center")
    ax.axhline(0, color="0.75", lw=0.8)
    ax.axvline(0, color="0.75", lw=0.8)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_title(f"{cohort_tag}: observation-mean PCA")
    fig.savefig(OUT_DIR / f"{cohort_tag}_unsupervised_observation_pca.png", dpi=220)
    plt.close(fig)


def run_cohort(manifest_path: Path, args: argparse.Namespace, spec_model, full_model, val_df, mask: np.ndarray, gdf: object | None) -> pd.DataFrame:
    cohort_tag = manifest_path.stem.replace("selected_nims_observations_", "selected_observations_")
    cohort_dir = OUT_DIR / cohort_tag
    cohort_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_path)

    summaries = []
    pixel_frames = []
    selected = [col for col, include in zip(SPEC_COLS, mask) if include]
    for _, row in manifest.iterrows():
        print(f"\n[{cohort_tag}] scoring {row['obs_id']} {row['file']} ({row['group']})")
        scored, summary = score_observation(row, mask, spec_model, full_model, val_df, gdf)
        out_name = f"nims_{safe_tag(row['obs_id'])}_{safe_tag(row['file'])}_scores.csv"
        scored.to_csv(cohort_dir / out_name, index=False)
        summaries.append(summary)
        pixel_frames.append(scored)
        print(
            f"  {row['obs_id']}: median p_full={summary['median_p_full']:.4f}, "
            f"OOD={summary['median_ood_ratio']:.1f}x, gate={summary['ood_gate']}"
        )

    summary = pd.DataFrame(summaries)
    summary["cohort"] = cohort_tag
    summary_path = OUT_DIR / f"{cohort_tag}_summary.csv"
    summary.to_csv(summary_path, index=False)
    pd.concat(pixel_frames, ignore_index=True).to_csv(OUT_DIR / f"{cohort_tag}_pixel_scores.csv", index=False)
    save_summary_plot(summary, OUT_DIR / f"{cohort_tag}_observation_summary.png", cohort_tag)
    run_unsupervised(summary, pixel_frames, selected, cohort_tag)
    print(f"Saved {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage-dir",
        default="data/processed/endmembers_ablation/02_ocean_salt_usgs_epsomite",
        help="Processed endmember stage directory.",
    )
    parser.add_argument("--wave-min", type=float, default=0.7)
    parser.add_argument("--wave-max", type=float, default=2.0)
    parser.add_argument("--n-synth", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=3421)
    parser.add_argument("--skip-geology-map", action="store_true")
    parser.add_argument(
        "--manifests",
        nargs="+",
        default=[
            "data/manifest/selected_nims_observations_8.csv",
            "data/manifest/selected_nims_observations_12.csv",
        ],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    stage_dir = Path(args.stage_dir)
    if not stage_dir.is_absolute():
        stage_dir = ROOT / stage_dir
    mask = stage25.wave_mask(args.wave_min, args.wave_max)
    print(f"Training staged model: {stage_dir} band={args.wave_min:.1f}-{args.wave_max:.1f} um")
    train_df = stage25.generate_training(stage_dir, args.n_synth, args.seed, mask)
    spec_model, full_model, val_df, metrics = stage25.fit_models(train_df, args.seed)
    pd.DataFrame([metrics]).to_csv(OUT_DIR / "expanded_model_validation_metrics.csv", index=False)
    print(
        "Validation: "
        f"PR-AUC spectral={metrics['val_pr_auc_spectral']:.4f}, full={metrics['val_pr_auc_full']:.4f}; "
        f"ROC-AUC spectral={metrics['val_roc_auc_spectral']:.4f}, full={metrics['val_roc_auc_full']:.4f}"
    )

    gdf = None if args.skip_geology_map else nims20.load_usgs_geomap(ROOT / "data" / "raw")
    all_summaries = []
    for manifest in args.manifests:
        all_summaries.append(run_cohort(ROOT / manifest, args, spec_model, full_model, val_df, mask, gdf))
    combined = pd.concat(all_summaries, ignore_index=True)
    for key, value in metrics.items():
        combined[key] = value
    combined["stage_dir"] = str(stage_dir.relative_to(ROOT) if stage_dir.is_relative_to(ROOT) else stage_dir)
    combined["wave_min_um"] = args.wave_min
    combined["wave_max_um"] = args.wave_max
    combined.to_csv(OUT_DIR / "expanded_nims_sanity_combined_summary.csv", index=False)


if __name__ == "__main__":
    main()
