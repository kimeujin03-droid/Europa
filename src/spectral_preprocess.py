from __future__ import annotations

import numpy as np
import pandas as pd


def clean_spectrum_table(df: pd.DataFrame) -> pd.DataFrame:
    required = {"wavelength_um", "value"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Spectrum CSV missing required columns: {sorted(missing)}")

    out = df.copy()
    out["wavelength_um"] = pd.to_numeric(out["wavelength_um"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["wavelength_um", "value"])
    out = out.sort_values("wavelength_um")
    out = out.drop_duplicates(subset=["wavelength_um"], keep="first")
    return out


def proxy_reflectance_from_intensity(values: np.ndarray, depth: float = 0.08) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values - np.nanmin(values)
    max_value = np.nanmax(values)
    if max_value <= 0:
        return np.ones_like(values)
    normalized = values / max_value
    return 1.0 - depth * normalized


def minmax_normalize_reflectance(values: np.ndarray, low: float = 0.35, high: float = 1.0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    if vmax <= vmin:
        return np.full_like(values, high)
    scaled = (values - vmin) / (vmax - vmin)
    return low + (high - low) * scaled


def interpolate_to_grid(
    df: pd.DataFrame,
    target_wavelengths: np.ndarray,
    intensity_depth: float = 0.08,
    normalize_reflectance: bool = True,
) -> np.ndarray:
    clean = clean_spectrum_table(df)
    kind = str(clean["kind"].iloc[0]).lower() if "kind" in clean.columns and len(clean) else "reflectance"
    x = clean["wavelength_um"].to_numpy(dtype=float)
    y = clean["value"].to_numpy(dtype=float)

    if "absorb" in kind or "intensity" in kind:
        y = proxy_reflectance_from_intensity(y, depth=intensity_depth)
    elif normalize_reflectance:
        y = minmax_normalize_reflectance(y)

    if target_wavelengths.min() < x.min() or target_wavelengths.max() > x.max():
        left = y[0]
        right = y[-1]
    else:
        left = None
        right = None

    interpolated = np.interp(target_wavelengths, x, y, left=left, right=right)
    return np.clip(interpolated, 0.02, 1.5)
