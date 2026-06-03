#!/usr/bin/env python
"""Experiment 4: Galileo/NIMS real-data spatial-context deployment test.

Uses two NIMS observations covering contrasting hemispheres:
  LEADING:  go_1108/europa/e6e002ci.qub  (clon=142W, E6 encounter)
  TRAILING: go_1105/europa/g2e002ci.qub  (clon=291W, G2 encounter)

Geologic context (G features) is derived from the USGS Global Geologic Map
of Europa (Klemaszewski & Leonard 2024, SIM 3513) via spatial join.

Radiation context (Q features) is derived from longitude (POSITIVE_LONGITUDE_DIRECTION=WEST).

Tests whether ΔS_geo and ΔS_rad shift decision function values in the
physically expected direction relative to a shuffled null baseline.

This is NOT a biosignature validation.  No life-label ground truth exists.

Outputs:
  results/nims_sanity_check/
    nims_leading_scores.csv
    nims_trailing_scores.csv
    nims_obs_summary.csv
    nims_ood_pca.csv
    nims_sanity_check_figure.png
"""
from __future__ import annotations

import re
import struct
import sys
import urllib.request
import zipfile
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu  # used for supplementary stats if needed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ExperimentConfig
from src.forward_model import generate_dataset
from src.modeling import feature_cols

# ---- Main synthetic condition ----
NOISE = "harsh"
RAD_SO_HI = 0.130

# ---- NIMS observations ----
PDS_BASE = "https://planetarydata.jpl.nasa.gov/img/data/go-e-nims-4-mosaic-v1.0"
NIMS_CUBES = [
    {"vol": "go_1108", "file": "e6e002ci.qub",  "hemisphere": "leading",
     "clon_w": 141.8, "encounter": "E6",  "terrain_type": "chaos"},
    {"vol": "go_1114", "file": "11e001ci.qub",  "hemisphere": "trailing",
     "clon_w": 219.0, "encounter": "E11", "terrain_type": "plains"},
    {"vol": "go_1105", "file": "g2e002ci.qub",  "hemisphere": "trailing",
     "clon_w": 291.1, "encounter": "G2",  "terrain_type": "chaos"},
    {"vol": "go_1116", "file": "15e001ci.qub",  "hemisphere": "leading",
     "clon_w": 105.6, "encounter": "E15", "terrain_type": "plains"},
]

# ---- Synthetic wavelength grid ----
WAVE_MIN, WAVE_MAX, WAVE_STEP = 0.7, 5.2, 0.025
N_SYNTH_BANDS = round((WAVE_MAX - WAVE_MIN) / WAVE_STEP) + 1  # 181

# ---- USGS Geologic Map ----
USGS_GIS_URL = "https://pubs.usgs.gov/sim/3513/sim3513_gis.zip"

# Geologic unit → G feature vector
# Derived from USGS SIM 3513 unit descriptions
# chaos_prox, lineament_prox, ridge_prox, young_terrain, activity_proxy
UNIT_GFEATURES = {
    # Chaos material (young, active, ocean-exchange-favorable).
    # In generate_spatial_features(), ocean_organic gets HIGH values for
    # chaos_proximity, lineament_proximity, ridge_proximity via beta_near(high=True).
    # So chaos terrain must encode HIGH values to be organic-favoring in the model.
    "chM":  {"chaos_proximity": 0.90, "lineament_proximity": 0.80, "ridge_proximity": 0.80, "young_terrain": 0.90, "activity_proxy": 0.90},
    "chh":  {"chaos_proximity": 0.85, "lineament_proximity": 0.75, "ridge_proximity": 0.75, "young_terrain": 0.85, "activity_proxy": 0.85},
    "chm":  {"chaos_proximity": 0.80, "lineament_proximity": 0.70, "ridge_proximity": 0.70, "young_terrain": 0.80, "activity_proxy": 0.80},
    "chl":  {"chaos_proximity": 0.75, "lineament_proximity": 0.65, "ridge_proximity": 0.65, "young_terrain": 0.75, "activity_proxy": 0.75},
    "c":    {"chaos_proximity": 0.75, "lineament_proximity": 0.65, "ridge_proximity": 0.65, "young_terrain": 0.70, "activity_proxy": 0.70},
    "ce":   {"chaos_proximity": 0.70, "lineament_proximity": 0.60, "ridge_proximity": 0.60, "young_terrain": 0.65, "activity_proxy": 0.65},
    # Band material (lineament/ridge rich but not actively chaotic)
    "b":    {"chaos_proximity": 0.50, "lineament_proximity": 0.80, "ridge_proximity": 0.75, "young_terrain": 0.60, "activity_proxy": 0.60},
    # Plains material (ancient, low activity, far from chaos)
    "pr":   {"chaos_proximity": 0.10, "lineament_proximity": 0.20, "ridge_proximity": 0.20, "young_terrain": 0.20, "activity_proxy": 0.20},
    "cpr":  {"chaos_proximity": 0.25, "lineament_proximity": 0.35, "ridge_proximity": 0.30, "young_terrain": 0.35, "activity_proxy": 0.30},
    "cpre": {"chaos_proximity": 0.20, "lineament_proximity": 0.30, "ridge_proximity": 0.30, "young_terrain": 0.35, "activity_proxy": 0.30},
    "nd":   {"chaos_proximity": 0.50, "lineament_proximity": 0.50, "ridge_proximity": 0.50, "young_terrain": 0.50, "activity_proxy": 0.50},
    "__default__": {"chaos_proximity": 0.50, "lineament_proximity": 0.50, "ridge_proximity": 0.50, "young_terrain": 0.50, "activity_proxy": 0.50},
}


# ================================================================ PDS utils ==

def download_bytes(url: str, out_path: Path, timeout: int = 180) -> bytes:
    if out_path.exists():
        return out_path.read_bytes()
    print(f"  Downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    print(f"  Saved {len(data)/1024:.1f} KB -> {out_path}")
    return data


def extract_attached_label(qub_bytes: bytes) -> tuple[str, int]:
    text = qub_bytes[:60000].decode("latin-1", errors="replace")
    m = re.search(r"RECORD_BYTES\s*=\s*(\d+)", text)
    record_bytes = int(m.group(1)) if m else 512
    m = re.search(r"^\s*\^QUBE\s*=\s*(\d+)", text, re.MULTILINE)
    qube_record = int(m.group(1)) if m else None
    end_m = re.search(r"\nEND\s*\n", text)
    label_text = text[: end_m.end()] if end_m else text[:5000]
    if qube_record is not None:
        qube_offset = (qube_record - 1) * record_bytes
    else:
        m2 = re.search(r"LABEL_RECORDS\s*=\s*(\d+)", text)
        label_records = int(m2.group(1)) if m2 else 46
        qube_offset = label_records * record_bytes
    return label_text, qube_offset


def extract_center_lon(label_text: str) -> float | None:
    """Return average sub-spacecraft longitude from PDS label (West, 0-360)."""
    s = re.search(r"START_SUB_SPACECRAFT_LONGITUDE\s*=\s*([\d.]+)", label_text)
    e = re.search(r"STOP_SUB_SPACECRAFT_LONGITUDE\s*=\s*([\d.]+)", label_text)
    if s and e:
        return (float(s.group(1)) + float(e.group(1))) / 2.0
    if s:
        return float(s.group(1))
    m = re.search(r"CENTER_LONGITUDE\s*=\s*([\d.]+)", label_text)
    return float(m.group(1)) if m else None


def parse_band_bin_center(label_text: str) -> np.ndarray | None:
    m = re.search(r"BAND_BIN_CENTER\s*=\s*\(([^)]+)\)", label_text, re.DOTALL)
    if not m:
        return None
    vals = [float(x.strip()) for x in m.group(1).split(",") if x.strip()]
    return np.array(vals)


def vax_real_to_float(raw4: bytes) -> float:
    b1, b2, b3, b4 = raw4[0], raw4[1], raw4[2], raw4[3]
    w1 = (b2 << 8) | b1
    w2 = (b4 << 8) | b3
    sign = (w1 >> 15) & 1
    exp  = (w1 >> 7) & 0xFF
    mant = ((w1 & 0x7F) << 16) | w2
    if exp == 0:
        return 0.0 if sign == 0 else float("nan")
    ieee_exp = exp - 1
    ieee_bits = (sign << 31) | (ieee_exp << 23) | mant
    return struct.unpack(">f", struct.pack(">I", ieee_bits))[0]


def vax_array_to_float(raw: bytes, count: int, byte_offset: int = 0) -> np.ndarray:
    result = np.empty(count, dtype=np.float32)
    for i in range(count):
        o = byte_offset + i * 4
        result[i] = vax_real_to_float(raw[o: o + 4])
    return result


def read_nims_qub(qub_bytes: bytes, synth_wave: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """Parse NIMS attached-label QUB → DataFrame with spectra + lat/lon."""
    label_text, qube_offset = extract_attached_label(qub_bytes)
    print(f"  QUBE binary at byte {qube_offset}")

    m = re.search(r"CORE_ITEMS\s*=\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", label_text)
    if not m:
        raise ValueError("CORE_ITEMS not found")
    n_samp, n_lines, n_bands = int(m.group(1)), int(m.group(2)), int(m.group(3))
    n_pix = n_samp * n_lines

    ms = re.search(r"SUFFIX_ITEMS\s*=\s*\(\s*\d+\s*,\s*\d+\s*,\s*(\d+)\s*\)", label_text)
    n_band_suffix = int(ms.group(1)) if ms else 0

    nims_wave = parse_band_bin_center(label_text)
    if nims_wave is None or len(nims_wave) != n_bands:
        raise ValueError("BAND_BIN_CENTER parse failed")

    print(f"  {n_samp}x{n_lines} pixels, {n_bands} bands [{nims_wave[0]:.4f}-{nims_wave[-1]:.4f}um], {n_band_suffix} suffix planes")

    binary = qub_bytes[qube_offset:]
    n_core = n_samp * n_lines * n_bands
    if len(binary) < n_core * 4:
        raise ValueError(f"Binary too short: {len(binary)} < {n_core*4}")

    # Core: AXIS_NAME = (SAMPLE, LINE, BAND), Fortran order (sample fastest)
    core_flat = vax_array_to_float(binary, n_core, byte_offset=0)
    core_3d = core_flat.reshape((n_samp, n_lines, n_bands), order="F")
    core_bsq = core_3d.transpose(2, 1, 0)   # (n_bands, n_lines, n_samp)
    core_arr = core_bsq.reshape(n_bands, n_pix)  # (n_bands, n_pix)

    # Suffix backplanes 0=lat, 1=lon
    lat = np.full(n_pix, np.nan)
    lon = np.full(n_pix, np.nan)
    suf_start = n_core * 4
    if n_band_suffix >= 2 and len(binary) >= suf_start + 2 * n_pix * 4:
        lat_f = vax_array_to_float(binary, n_pix, byte_offset=suf_start)
        lon_f = vax_array_to_float(binary, n_pix, byte_offset=suf_start + n_pix * 4)
        lat = lat_f.reshape((n_samp, n_lines), order="F").T.reshape(-1)
        lon = lon_f.reshape((n_samp, n_lines), order="F").T.reshape(-1)
    else:
        print("  [WARN] Latitude/longitude backplanes missing")

    core_arr[np.abs(core_arr) > 1e30] = np.nan
    lat[np.abs(lat) > 1e10] = np.nan
    lon[np.abs(lon) > 1e10] = np.nan

    valid_frac = np.isfinite(core_arr).mean(axis=0)
    valid_mask = valid_frac > 0.7
    print(f"  Valid pixels: {valid_mask.sum()} / {n_pix}")
    if valid_mask.sum() == 0:
        raise ValueError("No valid pixels")

    core_valid = core_arr[:, valid_mask]
    lat_v = lat[valid_mask]
    lon_v = lon[valid_mask]

    # Min-max normalize per pixel
    r_min = np.nanmin(core_valid, axis=0, keepdims=True)
    r_max = np.nanmax(core_valid, axis=0, keepdims=True)
    denom = np.where(r_max - r_min > 1e-10, r_max - r_min, 1.0)
    core_norm = (core_valid - r_min) / denom

    # Interpolate to synthetic grid
    n_valid = core_norm.shape[1]
    synth = np.full((n_valid, len(synth_wave)), np.nan)
    for i in range(n_valid):
        spec = core_norm[:, i]
        finite = np.isfinite(spec)
        if finite.sum() < 10:
            continue
        # Edge extrapolation handles slight wavelength boundary mismatches
        synth[i] = np.interp(synth_wave, nims_wave[finite], spec[finite],
                             left=spec[finite][0], right=spec[finite][-1])

    keep = np.isfinite(synth).all(axis=1)
    synth = synth[keep]
    lat_f2 = lat_v[keep]
    lon_f2 = lon_v[keep]
    print(f"  After interpolation: {len(synth)} pixels")

    spec_cols = [f"spec_{j:03d}" for j in range(len(synth_wave))]
    df = pd.DataFrame(synth, columns=spec_cols)
    df["lat_w"] = lat_f2      # planetocentric latitude
    df["lon_w"] = lon_f2      # west longitude (POSITIVE_LONGITUDE_DIRECTION=WEST)
    return df, nims_wave


# ================================================================ Geologic map ==

def load_usgs_geomap(raw_dir: Path) -> object:
    """Download + cache USGS GIS zip, return GeoDataFrame of geologic units."""
    import geopandas as gpd

    zip_path = raw_dir / "sim3513_gis.zip"
    shp_path = raw_dir / "sim3513_Europa_GIS" / "shapefiles" / "GeoUnits_FINAL.shp"

    if not shp_path.exists():
        download_bytes(USGS_GIS_URL, zip_path, timeout=300)
        print("  Extracting GIS shapefile ...")
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                if "GeoUnits_FINAL" in name and not name.startswith("__"):
                    zf.extract(name, raw_dir)

    gdf = gpd.read_file(str(shp_path))
    print(f"  USGS GeoUnits loaded: {len(gdf)} polygons, CRS: {gdf.crs.name}")
    return gdf


def geo_lookup_center(clon_w: float, n: int, gdf: object) -> tuple[pd.DataFrame, str]:
    """Look up geologic unit at observation center (lat=0, known central lon).

    The NIMS lat/lon suffix backplanes are unreliable for these cubes.
    We use the observation's known central longitude and equatorial latitude as
    a representative coordinate for the whole observation.

    clon_w : central west longitude (degrees W, 0–360)
    Returns (G feature DataFrame for all n pixels, unit code string)
    """
    import geopandas as gpd
    from shapely.geometry import Point

    lon_east_360 = (360.0 - clon_w) % 360.0
    lon_east_180 = lon_east_360 if lon_east_360 <= 180.0 else lon_east_360 - 360.0

    center = gpd.GeoDataFrame(
        {"idx": [0]},
        geometry=[Point(lon_east_180, 0.0)],
        crs=gdf.crs,
    )
    joined = gpd.sjoin(center, gdf[["Geounits", "geometry"]], how="left", predicate="within")
    unit = "__default__"
    if len(joined) > 0:
        raw = joined.iloc[0].get("Geounits", "__default__")
        if isinstance(raw, str):
            unit = raw
    g_vals = UNIT_GFEATURES.get(unit, UNIT_GFEATURES["__default__"])
    print(f"  Geologic center lookup: lon_east={lon_east_180:.1f}°E → unit={unit!r}")
    return pd.DataFrame([g_vals] * n), unit


# ================================================================ Radiation context ==

def assign_radiation_context_obs(clon_w: float, n: int) -> pd.DataFrame:
    """Radiation Q features from the observation's known central west longitude.

    Uses clon_w (the cube's central longitude, degrees W) to assign a uniform
    radiation context to all n pixels in the observation.  This is more reliable
    than per-pixel backplane coordinates, which are corrupted in these cubes.

    Trailing hemisphere: clon_w ∈ [180, 360)°W (radiation apex at 270°W).
    """
    lon_w_mod = clon_w % 360.0
    trailing = 1.0 if lon_w_mod >= 180.0 else 0.0
    rad_exp = float(np.clip(0.5 * (1.0 + np.cos(np.deg2rad(lon_w_mod) - np.deg2rad(270.0))),
                            0.0, 1.0))
    q = {
        "trailing_hemisphere": trailing,
        "radiation_exposure":  rad_exp,
        "sulfur_proxy":        rad_exp * 0.9,
        "rad_mimic_proxy":     trailing * 0.7,
    }
    print(f"  Radiation context: trailing={trailing:.0f}, rad_exp={rad_exp:.3f}")
    return pd.DataFrame([q] * n)


# ================================================================ Model ==

def train_model(seed: int = 3421) -> tuple[Pipeline, Pipeline]:
    cfg = ExperimentConfig(seed=seed, noise_condition=NOISE,
                           rad_simple_organic_hi=RAD_SO_HI)
    df = generate_dataset(n=8000, rho_geo=0.75, rho_rad=0.75, cfg=cfg)
    train_df, test_df = train_test_split(df, test_size=0.30, stratify=df["y"],
                                         random_state=seed)
    y_train = train_df["y"].to_numpy()

    def fit(setting: str) -> Pipeline:
        cols = feature_cols(train_df, setting)
        X = train_df[cols].to_numpy()
        pipe = Pipeline([("scale", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000,
                                                     class_weight="balanced",
                                                     random_state=seed))])
        pipe.fit(X, y_train)
        return pipe

    return fit("spectral_only"), fit("full"), test_df


# ================================================================ Delta DV ==

GEO_COLS = ["chaos_proximity", "lineament_proximity", "ridge_proximity",
            "young_terrain", "activity_proxy"]
RAD_COLS = ["trailing_hemisphere", "radiation_exposure", "sulfur_proxy", "rad_mimic_proxy"]
NEUTRAL_GEO = {c: 0.5 for c in GEO_COLS}
NEUTRAL_RAD = {"trailing_hemisphere": 0.0, "radiation_exposure": 0.5,
               "sulfur_proxy": 0.45, "rad_mimic_proxy": 0.0}


def compute_delta_dvs(
    df_spectra: pd.DataFrame,
    g_real: pd.DataFrame,
    q_real: pd.DataFrame,
    m_spec: Pipeline,
    m_full: Pipeline,
) -> pd.DataFrame:
    """Compute decomposed context modulation for each pixel vs neutral baseline.

    ΔDV_geo  = DV(G_real, Q_neutral) − DV(G_neutral, Q_neutral)
    ΔDV_rad  = DV(G_neutral, Q_real) − DV(G_neutral, Q_neutral)
    ΔDV_full = DV(G_real, Q_real)    − DV(G_neutral, Q_neutral)

    For logistic regression these deltas are constant across pixels when g_real
    and q_real are uniform (observation-level context), so per-pixel variation
    comes entirely from the spectral score dv_spec.
    """
    spec_cols = [c for c in df_spectra.columns if c.startswith("spec_")]
    n = len(df_spectra)
    X_spec = df_spectra[spec_cols].to_numpy()

    g_neut = pd.DataFrame([NEUTRAL_GEO] * n)
    q_neut = pd.DataFrame([NEUTRAL_RAD] * n)

    def dv_full_fn(g: pd.DataFrame, q: pd.DataFrame) -> np.ndarray:
        X = np.hstack([X_spec, g[GEO_COLS].to_numpy(), q[RAD_COLS].to_numpy()])
        return m_full.decision_function(X)

    dv_spec   = m_spec.decision_function(X_spec)
    dv_neut   = dv_full_fn(g_neut, q_neut)     # fully neutral baseline
    dv_geo_r  = dv_full_fn(g_real, q_neut)     # real G + neutral Q
    dv_rad_r  = dv_full_fn(g_neut, q_real)     # neutral G + real Q
    dv_full_r = dv_full_fn(g_real, q_real)     # real G + real Q

    result = df_spectra[spec_cols + ["lat_w", "lon_w"]].copy()
    result["dv_spec"]     = dv_spec
    result["dv_neut"]     = dv_neut
    result["dv_geo_real"] = dv_geo_r
    result["dv_rad_real"] = dv_rad_r
    result["dv_full_real"]= dv_full_r
    result["delta_geo"]   = dv_geo_r  - dv_neut   # geo context vs neutral
    result["delta_rad"]   = dv_rad_r  - dv_neut   # rad context vs neutral
    result["delta_full"]  = dv_full_r - dv_neut   # combined context vs neutral
    for col in GEO_COLS:
        result[col] = g_real[col].to_numpy()
    for col in RAD_COLS:
        result[col] = q_real[col].to_numpy()
    return result


# ================================================================ OOD diagnosis ==

def pca_ood_distance(
    X_nims: np.ndarray,
    X_synth: np.ndarray,
    n_components: int = 5,
) -> np.ndarray:
    """Return L2 distance from each NIMS pixel to the nearest synthetic test point in PCA space."""
    pca = PCA(n_components=n_components)
    # Fit on synthetic, transform both
    X_all = np.vstack([X_synth, X_nims])
    # Normalize first
    mu = X_synth.mean(axis=0)
    sd = X_synth.std(axis=0) + 1e-8
    X_synth_n = (X_synth - mu) / sd
    X_nims_n  = (X_nims  - mu) / sd

    pca.fit(X_synth_n)
    P_synth = pca.transform(X_synth_n)
    P_nims  = pca.transform(X_nims_n)

    # Nearest neighbor distance
    dists = []
    for pn in P_nims:
        d = np.linalg.norm(P_synth - pn, axis=1).min()
        dists.append(d)
    return np.array(dists)


# ================================================================ Figure ==

def make_figure(
    summary: list[dict],
    dfs_by_obs: dict,
    out_path: Path,
) -> None:
    """4-panel figure for the NIMS spatial-context deployment test (n observations).

    Panels A-C: bar charts of ΔDV values (observation-level constants for LogReg).
    Panel D: per-pixel dv_spec histograms + OOD annotation.
    Color: blue=leading, red=trailing. Hatch: solid=chaos, ////=plains.
    """
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))

    lead_col  = "#1f77b4"
    trail_col = "#d62728"

    def obs_style(row: dict) -> tuple[str, str]:
        col   = lead_col  if row["hemisphere"] == "leading" else trail_col
        hatch = "////"    if row.get("terrain_type", "chaos") == "plains" else ""
        return col, hatch

    n = len(summary)
    x_pos = np.arange(n) * 1.1

    tick_labels = [
        f"{r['encounter']}\n{r['hemisphere']}\n{r['geo_unit']}\n(n={r['n_pixels']})"
        for r in summary
    ]

    def bar_panel(ax, key, ylabel, title):
        for i, row in enumerate(summary):
            col, hatch = obs_style(row)
            v = row.get(key, 0.0)
            ax.bar(x_pos[i], v, width=0.7, color=col, alpha=0.75,
                   edgecolor="black", lw=0.8, hatch=hatch)
            ax.text(x_pos[i], v + 0.02 * np.sign(v + 1e-9),
                    f"{v:+.3f}", ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=7.5, fontweight="bold")
        ax.axhline(0, color="black", lw=0.9, ls="--", alpha=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(tick_labels, fontsize=6.5)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=8)
        ax.grid(alpha=0.2, axis="y")
        ax.set_xlim(x_pos[0] - 0.7, x_pos[-1] + 0.7)

    bar_panel(axes[0], "delta_rad",
              r"$\Delta DV_{\rm rad}$ (real Q − neutral Q)",
              "A. Radiation-context shift\n(+: organic-favoring, −: suppressed)")

    bar_panel(axes[1], "delta_geo",
              r"$\Delta DV_{\rm geo}$ (real G − neutral G)",
              "B. Geologic-context shift\n(USGS SIM 3513 center lookup)")

    bar_panel(axes[2], "delta_full",
              r"$\Delta DV_{\rm full}$ (real G+Q − neutral G+Q)",
              "C. Combined context shift\n(geo + rad)")

    # --- Panel D: per-pixel dv_spec histograms + OOD annotation ---
    ax = axes[3]
    all_dv = np.concatenate([
        df["dv_spec"].to_numpy() for df in dfs_by_obs.values() if len(df) and "dv_spec" in df
    ])
    bins_dv = np.linspace(all_dv.min() - 0.2, all_dv.max() + 0.2, 30)
    ood_lines = []
    for row in summary:
        fname = row["obs_file"]
        df = dfs_by_obs.get(fname, pd.DataFrame())
        if not len(df) or "dv_spec" not in df:
            continue
        col, hatch = obs_style(row)
        label = f"{row['encounter']} {row['hemisphere']}"
        ax.hist(df["dv_spec"].to_numpy(), bins=bins_dv,
                color=col, alpha=0.50, label=label,
                hatch=hatch if hatch else None, edgecolor=col if hatch else None)
        if "ood_dist" in df:
            ood_lines.append(f"  {row['encounter']} μ={df['ood_dist'].mean():.1f}")
    ax.set_xlabel("Spectral-only decision function $DV_{\\rm spec}$", fontsize=8)
    ax.set_ylabel("Pixel count", fontsize=8)
    ax.set_title("D. Per-pixel spectral score\n(OOD: NIMS >> synthetic)", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)
    if ood_lines:
        txt = "OOD dist (PCA-NN):\n" + "\n".join(ood_lines)
        ax.text(0.97, 0.97, txt, transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color="gray",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    # Legend patches for color/hatch key
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=lead_col,  edgecolor="black", label="Leading"),
        Patch(facecolor=trail_col, edgecolor="black", label="Trailing"),
        Patch(facecolor="white",   edgecolor="black", label="Chaos (solid)"),
        Patch(facecolor="white",   edgecolor="black", hatch="////", label="Plains (hatch)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=7.5,
               framealpha=0.8, bbox_to_anchor=(0.5, -0.06))

    fig.suptitle(
        "Galileo/NIMS real-data spatial-context deployment test  "
        "(n=4, 2×2: hemisphere × terrain)\n"
        "Qualitative behavioral check — no ground-truth life labels exist",
        fontsize=9, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {out_path}")


# ================================================================ Main ==

def main() -> None:
    out_dir = ROOT / "results" / "nims_sanity_check"
    raw_dir = ROOT / "data" / "raw" / "nims_pds"
    geo_dir = ROOT / "data" / "raw" / "usgs_europa_geomap"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    geo_dir.mkdir(parents=True, exist_ok=True)

    synth_wave = np.arange(N_SYNTH_BANDS) * WAVE_STEP + WAVE_MIN

    # ---- Load USGS geologic map ----
    print("[Step 1] Loading USGS Global Geologic Map of Europa (SIM 3513) ...")
    try:
        gdf = load_usgs_geomap(geo_dir)
    except Exception as e:
        print(f"  [WARN] USGS map unavailable ({e}). Geologic features will be neutral.")
        gdf = None

    # ---- Train model ----
    print("\n[Step 2] Training logistic regression on synthetic main condition ...")
    m_spec, m_full, test_df = train_model(seed=3421)
    spec_cols = [c for c in test_df.columns if c.startswith("spec_")]
    X_synth_test = test_df[spec_cols].to_numpy()

    # ---- Process each NIMS observation ----
    dfs_by_obs = {}   # keyed by fname (avoids hemisphere key collision with n>2)

    for cube_info in NIMS_CUBES:
        vol          = cube_info["vol"]
        fname        = cube_info["file"]
        hemi         = cube_info["hemisphere"]
        encounter    = cube_info["encounter"]
        terrain_type = cube_info.get("terrain_type", "chaos")
        url          = f"{PDS_BASE}/{vol}/europa/{fname}"

        print(f"\n[Step 3] Processing {fname} ({hemi}, {encounter}, {terrain_type}) ...")

        try:
            qub_bytes = download_bytes(url, raw_dir / fname)
        except Exception as e:
            print(f"  [ERROR] Download failed: {e}")
            continue

        try:
            label_text, _ = extract_attached_label(qub_bytes)
            label_clon = extract_center_lon(label_text)
            if label_clon is not None:
                print(f"  PDS label CENTER_LONGITUDE = {label_clon:.1f}°W")
                if abs(label_clon - cube_info["clon_w"]) > 5.0:
                    print(f"  [WARN] clon_w mismatch: hardcoded={cube_info['clon_w']}°W, "
                          f"label={label_clon:.1f}°W — using label value")
                    cube_info = dict(cube_info, clon_w=label_clon)
            df_nims, nims_wave = read_nims_qub(qub_bytes, synth_wave)
        except Exception as e:
            print(f"  [ERROR] Parse failed: {e}")
            continue

        if len(df_nims) < 5:
            print(f"  [SKIP] Too few valid pixels: {len(df_nims)}")
            continue

        clon_w = cube_info["clon_w"]

        # Radiation context from known cube central longitude (suffix backplanes unreliable)
        q_real = assign_radiation_context_obs(clon_w, len(df_nims))

        # Geologic context from USGS center lookup (avoids unreliable per-pixel suffix coords)
        if gdf is not None:
            g_real, g_unit = geo_lookup_center(clon_w, len(df_nims), gdf)
        else:
            g_real = pd.DataFrame([NEUTRAL_GEO] * len(df_nims))
            g_unit = "neutral"

        # Compute ΔDV vs neutral baseline
        df_scores = compute_delta_dvs(df_nims, g_real, q_real, m_spec, m_full)
        df_scores["obs_file"]     = fname
        df_scores["encounter"]    = encounter
        df_scores["hemisphere"]   = hemi
        df_scores["terrain_type"] = terrain_type
        df_scores["geo_unit"]     = g_unit

        # OOD distance
        X_nims_spec = df_nims[[c for c in df_nims.columns if c.startswith("spec_")]].to_numpy()
        ood_dists = pca_ood_distance(X_nims_spec, X_synth_test)
        df_scores["ood_dist"] = ood_dists

        df_scores.to_csv(out_dir / f"nims_{encounter.lower()}_scores.csv", index=False)
        dfs_by_obs[fname] = df_scores

        # delta values are uniform per obs (linear model + uniform context)
        print(f"  delta_geo:  {df_scores['delta_geo'].iloc[0]:+.4f}  (uniform: geo_unit={g_unit!r})")
        print(f"  delta_rad:  {df_scores['delta_rad'].iloc[0]:+.4f}  (uniform: clon_w={clon_w}°W)")
        print(f"  delta_full: {df_scores['delta_full'].iloc[0]:+.4f}")
        print(f"  OOD dist:   mean={ood_dists.mean():.2f}")

    if not dfs_by_obs:
        print("[ERROR] No NIMS data processed. Exiting.")
        return

    # ---- Observation-level summary ----
    print("\n[Results] Observation-level summary:")
    summary_rows = []
    for cube_info in NIMS_CUBES:
        fname = cube_info["file"]
        df_s = dfs_by_obs.get(fname)
        if df_s is None:
            continue
        row = {
            "obs_file":     df_s["obs_file"].iloc[0],
            "encounter":    df_s["encounter"].iloc[0],
            "hemisphere":   df_s["hemisphere"].iloc[0],
            "terrain_type": df_s["terrain_type"].iloc[0],
            "geo_unit":     df_s["geo_unit"].iloc[0],
            "n_pixels":     len(df_s),
            "delta_geo":    float(df_s["delta_geo"].iloc[0]),
            "delta_rad":    float(df_s["delta_rad"].iloc[0]),
            "delta_full":   float(df_s["delta_full"].iloc[0]),
            "chaos_proximity":     float(df_s["chaos_proximity"].iloc[0]),
            "trailing_hemisphere": float(df_s["trailing_hemisphere"].iloc[0]),
            "radiation_exposure":  float(df_s["radiation_exposure"].iloc[0]),
            "mean_dv_spec":        float(df_s["dv_spec"].mean()),
            "std_dv_spec":         float(df_s["dv_spec"].std()),
            "mean_ood_dist":       float(df_s["ood_dist"].mean()),
            "std_ood_dist":        float(df_s["ood_dist"].std()),
        }
        summary_rows.append(row)
        print(f"  {row['encounter']:5s} {row['hemisphere']:10s} {row['obs_file']:18s}: "
              f"n={row['n_pixels']:3d}, geo_unit={row['geo_unit']:4s}, "
              f"ΔDV_geo={row['delta_geo']:+.3f}, "
              f"ΔDV_rad={row['delta_rad']:+.3f}, "
              f"ΔDV_full={row['delta_full']:+.3f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "nims_obs_summary.csv", index=False)

    # Synth internal OOD baseline
    X_s = X_synth_test[:200]
    internal_dists = pca_ood_distance(X_s[:100], X_s[100:])
    print(f"\n  Synthetic internal OOD baseline: mean={internal_dists.mean():.2f}  (NIMS should be much larger)")

    # ---- Figure ----
    print("\n[Step 4] Generating figure ...")
    if len(summary_rows) >= 1:
        make_figure(summary_rows, dfs_by_obs, out_dir / "nims_sanity_check_figure.png")
    else:
        print("  [SKIP] No observations processed")

    print(f"\nOutputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
