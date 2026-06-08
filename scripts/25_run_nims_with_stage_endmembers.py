#!/usr/bin/env python
"""Apply a staged endmember model to existing NIMS real-data score CSVs.

This reuses the NIMS spectra already produced by scripts/20_nims_sanity_check.py
and trains a synthetic model with a selected staged endmember directory, e.g.
data/processed/endmembers_ablation/02_ocean_salt_usgs_epsomite.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.forward_model as forward_model
from src.config import ExperimentConfig
from src.endmember_loader import load_processed_endmembers


NIMS_DIR = ROOT / "results" / "nims_sanity_check"
OUT_DIR = ROOT / "results" / "nims_stage_model"
SPEC_COLS = [f"spec_{j:03d}" for j in range(181)]
GEO_COLS = ["chaos_proximity", "lineament_proximity", "ridge_proximity", "young_terrain", "activity_proxy"]
RAD_COLS = ["trailing_hemisphere", "radiation_exposure", "sulfur_proxy", "rad_mimic_proxy"]
NEUTRAL_GEO = {col: 0.5 for col in GEO_COLS}
NEUTRAL_RAD = {col: 0.5 for col in RAD_COLS}

NIMS_FILES = {
    "E6": NIMS_DIR / "nims_e6_scores.csv",
    "E15": NIMS_DIR / "nims_e15_scores.csv",
    "G2": NIMS_DIR / "nims_g2_scores.csv",
    "E11": NIMS_DIR / "nims_e11_scores.csv",
}

NIMS_META = {
    "E6": {
        "hemisphere": "leading",
        "terrain": "chaos",
        "geo_unit": "chm",
        "chaos_proximity": 0.80,
        "lineament_proximity": 0.70,
        "ridge_proximity": 0.70,
        "young_terrain": 0.80,
        "activity_proxy": 0.80,
        "trailing_hemisphere": 0.0,
        "radiation_exposure": 0.190795802321223,
        "sulfur_proxy": 0.1717162220891007,
        "rad_mimic_proxy": 0.0,
    },
    "E15": {
        "hemisphere": "leading",
        "terrain": "plains",
        "geo_unit": "pr",
        "chaos_proximity": 0.10,
        "lineament_proximity": 0.20,
        "ridge_proximity": 0.20,
        "young_terrain": 0.20,
        "activity_proxy": 0.20,
        "trailing_hemisphere": 0.0,
        "radiation_exposure": 0.018418716601170948,
        "sulfur_proxy": 0.016576844941053855,
        "rad_mimic_proxy": 0.0,
    },
    "G2": {
        "hemisphere": "trailing",
        "terrain": "chaos",
        "geo_unit": "chl",
        "chaos_proximity": 0.75,
        "lineament_proximity": 0.65,
        "ridge_proximity": 0.65,
        "young_terrain": 0.75,
        "activity_proxy": 0.75,
        "trailing_hemisphere": 1.0,
        "radiation_exposure": 0.9664767674127446,
        "sulfur_proxy": 0.8698290906714702,
        "rad_mimic_proxy": 0.7,
    },
    "E11": {
        "hemisphere": "trailing",
        "terrain": "plains",
        "geo_unit": "pr",
        "chaos_proximity": 0.10,
        "lineament_proximity": 0.20,
        "ridge_proximity": 0.20,
        "young_terrain": 0.20,
        "activity_proxy": 0.20,
        "trailing_hemisphere": 1.0,
        "radiation_exposure": 0.8146601955249189,
        "sulfur_proxy": 0.733194175972427,
        "rad_mimic_proxy": 0.7,
    },
}


def wave_mask(wave_min: float, wave_max: float) -> np.ndarray:
    wavelengths = ExperimentConfig().wavelengths
    return (wavelengths >= wave_min - 1e-9) & (wavelengths <= wave_max + 1e-9)


def generate_training(stage_dir: Path, n: int, seed: int, mask: np.ndarray) -> pd.DataFrame:
    original_available = forward_model.processed_endmembers_available
    original_loader = forward_model.load_processed_endmembers
    try:
        forward_model.processed_endmembers_available = lambda: True
        forward_model.load_processed_endmembers = lambda wavelengths: load_processed_endmembers(
            wavelengths, base_dir=stage_dir
        )
        cfg = ExperimentConfig(seed=seed, noise_condition="harsh", rad_simple_organic_hi=0.130)
        df = forward_model.generate_dataset(n=n, rho_geo=0.75, rho_rad=0.75, cfg=cfg)
    finally:
        forward_model.processed_endmembers_available = original_available
        forward_model.load_processed_endmembers = original_loader

    selected = [col for col, include in zip(SPEC_COLS, mask) if include]
    return df[["sample_id", "z", "y", *selected, *GEO_COLS, *RAD_COLS]].copy()


def fit_models(train_df: pd.DataFrame, seed: int) -> tuple[Pipeline, Pipeline, pd.DataFrame, dict]:
    spec_cols = [col for col in train_df.columns if col.startswith("spec_")]
    full_cols = spec_cols + GEO_COLS + RAD_COLS
    train, val = train_test_split(train_df, test_size=0.30, stratify=train_df["y"], random_state=seed)

    def fit(cols: list[str]) -> Pipeline:
        pipe = Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, random_state=seed)),
            ]
        )
        pipe.fit(train[cols].to_numpy(), train["y"].to_numpy())
        return pipe

    spec_model = fit(spec_cols)
    full_model = fit(full_cols)
    spec_score = spec_model.predict_proba(val[spec_cols].to_numpy())[:, 1]
    full_score = full_model.predict_proba(val[full_cols].to_numpy())[:, 1]
    metrics = {
        "val_pr_auc_spectral": float(average_precision_score(val["y"], spec_score)),
        "val_roc_auc_spectral": float(roc_auc_score(val["y"], spec_score)),
        "val_pr_auc_full": float(average_precision_score(val["y"], full_score)),
        "val_roc_auc_full": float(roc_auc_score(val["y"], full_score)),
    }
    return spec_model, full_model, val, metrics


def ood_distances(nims: np.ndarray, synth: np.ndarray, n_components: int = 5) -> tuple[np.ndarray, float]:
    n_components = min(n_components, nims.shape[1], synth.shape[1], len(synth) - 1)
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(synth)
    synth_proj = pca.transform(synth)
    nims_proj = pca.transform(nims)

    distances = np.empty(len(nims_proj), dtype=float)
    for start in range(0, len(nims_proj), 500):
        end = min(start + 500, len(nims_proj))
        diff = nims_proj[start:end, np.newaxis, :] - synth_proj[np.newaxis, :, :]
        distances[start:end] = np.sqrt((diff**2).sum(axis=2)).min(axis=1)

    ref = np.empty(500, dtype=float)
    left = synth_proj[:500]
    right = synth_proj[500:1000]
    for start in range(0, len(left), 100):
        end = min(start + 100, len(left))
        diff = left[start:end, np.newaxis, :] - right[np.newaxis, :, :]
        ref[start:end] = np.sqrt((diff**2).sum(axis=2)).min(axis=1)
    return distances, float(ref.mean())


def build_context(meta: dict, n: int, mode: str) -> pd.DataFrame:
    if mode == "real":
        values = {col: meta[col] for col in GEO_COLS + RAD_COLS}
    elif mode == "neutral_all":
        values = {**NEUTRAL_GEO, **NEUTRAL_RAD}
    elif mode == "neutral_geo":
        values = {**NEUTRAL_GEO, **{col: meta[col] for col in RAD_COLS}}
    elif mode == "neutral_rad":
        values = {**{col: meta[col] for col in GEO_COLS}, **NEUTRAL_RAD}
    else:
        raise ValueError(mode)
    return pd.DataFrame([values] * n)


def score_nims(
    enc: str,
    path: Path,
    mask: np.ndarray,
    spec_model: Pipeline,
    full_model: Pipeline,
    val_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    selected = [col for col, include in zip(SPEC_COLS, mask) if include]
    raw = pd.read_csv(path, usecols=selected)
    meta = NIMS_META[enc]
    n = len(raw)
    spec = raw[selected].to_numpy()

    ctx_real = build_context(meta, n, "real")
    ctx_neut_all = build_context(meta, n, "neutral_all")
    ctx_neut_geo = build_context(meta, n, "neutral_geo")
    ctx_neut_rad = build_context(meta, n, "neutral_rad")

    def full_score(ctx: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = np.hstack([spec, ctx[GEO_COLS + RAD_COLS].to_numpy()])
        return full_model.predict_proba(x)[:, 1], full_model.decision_function(x)

    p_spec = spec_model.predict_proba(spec)[:, 1]
    dv_spec = spec_model.decision_function(spec)
    p_real, dv_real = full_score(ctx_real)
    _, dv_neut_all = full_score(ctx_neut_all)
    _, dv_neut_geo = full_score(ctx_neut_geo)
    _, dv_neut_rad = full_score(ctx_neut_rad)

    synth_spec = val_df[selected].to_numpy()
    ood, ref = ood_distances(spec, synth_spec)

    scored = raw.copy()
    scored["encounter"] = enc
    scored["hemisphere"] = meta["hemisphere"]
    scored["terrain"] = meta["terrain"]
    scored["geo_unit"] = meta["geo_unit"]
    scored["p_spec"] = p_spec
    scored["p_full"] = p_real
    scored["dv_spec"] = dv_spec
    scored["dv_full"] = dv_real
    scored["delta_geo"] = dv_real - dv_neut_geo
    scored["delta_rad"] = dv_real - dv_neut_rad
    scored["delta_full"] = dv_real - dv_neut_all
    scored["ood_dist"] = ood
    scored["ood_ratio"] = ood / (ref + 1e-9)

    summary = {
        "encounter": enc,
        "hemisphere": meta["hemisphere"],
        "terrain": meta["terrain"],
        "geo_unit": meta["geo_unit"],
        "n_pixels": n,
        "mean_p_spec": float(np.mean(p_spec)),
        "mean_p_full": float(np.mean(p_real)),
        "median_p_full": float(np.median(p_real)),
        "mean_dv_spec": float(np.mean(dv_spec)),
        "mean_dv_full": float(np.mean(dv_real)),
        "delta_geo": float(scored["delta_geo"].iloc[0]),
        "delta_rad": float(scored["delta_rad"].iloc[0]),
        "delta_full": float(scored["delta_full"].iloc[0]),
        "mean_ood_dist": float(np.mean(ood)),
        "mean_ood_ratio": float(np.mean(ood / (ref + 1e-9))),
        "synth_internal_ood": ref,
        "top1pct_mean_p_full": float(np.mean(np.sort(p_real)[-max(1, n // 100) :])),
    }
    return scored, summary


def run(args: argparse.Namespace) -> None:
    stage_dir = Path(args.stage_dir)
    if not stage_dir.is_absolute():
        stage_dir = ROOT / stage_dir
    if not stage_dir.exists():
        raise FileNotFoundError(stage_dir)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mask = wave_mask(args.wave_min, args.wave_max)
    band_tag = f"{args.wave_min:.1f}_{args.wave_max:.1f}".replace(".", "p")
    stage_tag = stage_dir.name

    print(f"Training stage={stage_dir} band={args.wave_min:.1f}-{args.wave_max:.1f} um")
    train_df = generate_training(stage_dir, args.n_synth, args.seed, mask)
    spec_model, full_model, val_df, metrics = fit_models(train_df, args.seed)
    print(
        "Validation: "
        f"PR-AUC spectral={metrics['val_pr_auc_spectral']:.4f}, full={metrics['val_pr_auc_full']:.4f}; "
        f"ROC-AUC spectral={metrics['val_roc_auc_spectral']:.4f}, full={metrics['val_roc_auc_full']:.4f}"
    )

    summaries = []
    for enc, path in NIMS_FILES.items():
        scored, summary = score_nims(enc, path, mask, spec_model, full_model, val_df)
        out_path = OUT_DIR / f"nims_{enc.lower()}_{stage_tag}_{band_tag}_scores.csv"
        scored.to_csv(out_path, index=False)
        summaries.append(summary)
        print(
            f"{enc}: p_full={summary['mean_p_full']:.4f}, "
            f"dv_full={summary['mean_dv_full']:+.2f}, "
            f"dG={summary['delta_geo']:+.2f}, dR={summary['delta_rad']:+.2f}, "
            f"OOD={summary['mean_ood_ratio']:.1f}x"
        )

    summary_df = pd.DataFrame(summaries)
    for key, value in metrics.items():
        summary_df[key] = value
    summary_df["stage_dir"] = str(stage_dir.relative_to(ROOT) if stage_dir.is_relative_to(ROOT) else stage_dir)
    summary_df["wave_min_um"] = args.wave_min
    summary_df["wave_max_um"] = args.wave_max
    summary_path = OUT_DIR / f"nims_summary_{stage_tag}_{band_tag}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage-dir",
        default="data/processed/endmembers_ablation/02_ocean_salt_usgs_epsomite",
        help="Processed endmember stage directory.",
    )
    parser.add_argument("--wave-min", type=float, default=0.7)
    parser.add_argument("--wave-max", type=float, default=5.2)
    parser.add_argument("--n-synth", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=3421)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
