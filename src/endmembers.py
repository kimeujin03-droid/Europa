"""Toy endmember spectra for first-pass synthetic experiments.

중요: 이 파일의 Gaussian absorption spectrum은 논문 최종용 실험실 spectrum이 아닙니다.
초기 코드 검증용 toy forward model입니다. 최종 실험에서는 RELAB/PDS/USGS/PAHdb 등에서
가져온 laboratory spectra로 교체해야 합니다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


def gaussian_absorption(wavelengths: np.ndarray, center: float, width: float, depth: float) -> np.ndarray:
    return depth * np.exp(-0.5 * ((wavelengths - center) / width) ** 2)


def continuum(wavelengths: np.ndarray, slope: float = 0.0, intercept: float = 1.0) -> np.ndarray:
    x = wavelengths - wavelengths.mean()
    return intercept + slope * x


def clip_reflectance(r: np.ndarray) -> np.ndarray:
    return np.clip(r, 0.02, 1.5)


def toy_endmember_library(wavelengths: np.ndarray) -> dict[str, np.ndarray]:
    """Return normalized toy reflectance spectra on a NIMS-like wavelength grid."""
    w = wavelengths

    # Water ice: broad absorptions around ~1.5, 2.0, 3.0 µm.
    ice = continuum(w, slope=-0.02, intercept=1.0)
    for c, wid, dep in [(1.5, 0.08, 0.25), (2.0, 0.10, 0.35), (3.05, 0.22, 0.55), (4.5, 0.18, 0.12)]:
        ice -= gaussian_absorption(w, c, wid, dep)
    ice = clip_reflectance(ice)

    # Hydrated salt / ocean salt: hydration bands and weak structure.
    ocean_salt = continuum(w, slope=0.005, intercept=0.85)
    for c, wid, dep in [(1.45, 0.10, 0.12), (1.95, 0.11, 0.16), (2.4, 0.08, 0.06), (3.05, 0.18, 0.22)]:
        ocean_salt -= gaussian_absorption(w, c, wid, dep)
    ocean_salt = clip_reflectance(ocean_salt)

    # Simple organics: intentionally weak C-H-like structure for hard-positive tests.
    simple_org = continuum(w, slope=0.015, intercept=0.9)
    for c, wid, dep in [(2.95, 0.06, 0.040), (3.38, 0.065, 0.090), (3.48, 0.065, 0.070), (4.65, 0.08, 0.050)]:
        simple_org -= gaussian_absorption(w, c, wid, dep)
    simple_org = clip_reflectance(simple_org)

    # Tholin/PAH-like complex organics: hard negative with overlapping 3.3-3.5 um C-H-like bands.
    tholin_pah = continuum(w, slope=-0.00, intercept=0.75) + 0.10 * np.tanh((w - 1.4) / 0.8)
    for c, wid, dep in [(1.65, 0.10, 0.04), (3.34, 0.075, 0.100), (3.43, 0.075, 0.090), (3.52, 0.070, 0.060), (4.25, 0.10, 0.035)]:
        tholin_pah -= gaussian_absorption(w, c, wid, dep)
    tholin_pah = clip_reflectance(tholin_pah)

    # Sulfuric acid hydrate proxy: broad/distorted hydration features.
    sulfuric_acid_hydrate = continuum(w, slope=-0.035, intercept=0.95)
    for c, wid, dep in [(1.35, 0.12, 0.08), (1.75, 0.14, 0.10), (2.05, 0.13, 0.11), (3.05, 0.25, 0.35)]:
        sulfuric_acid_hydrate -= gaussian_absorption(w, c, wid, dep)
    sulfuric_acid_hydrate = clip_reflectance(sulfuric_acid_hydrate)

    # SO2 / sulfur product group proxy.
    sulfur_so2 = continuum(w, slope=-0.015, intercept=0.82)
    for c, wid, dep in [(2.0, 0.05, 0.05), (3.95, 0.08, 0.08), (4.05, 0.08, 0.08)]:
        sulfur_so2 -= gaussian_absorption(w, c, wid, dep)
    sulfur_so2 = clip_reflectance(sulfur_so2)

    # H2O2 proxy: feature near 3.5 µm.
    h2o2 = continuum(w, slope=-0.01, intercept=0.92)
    for c, wid, dep in [(3.5, 0.08, 0.04)]:
        h2o2 -= gaussian_absorption(w, c, wid, dep)
    h2o2 = clip_reflectance(h2o2)

    # Radiation-altered salt mimic parameter.
    rad_salt = continuum(w, slope=-0.03, intercept=0.88)
    for c, wid, dep in [(1.7, 0.14, 0.08), (2.15, 0.11, 0.08), (3.05, 0.21, 0.24), (3.5, 0.10, 0.06)]:
        rad_salt -= gaussian_absorption(w, c, wid, dep)
    rad_salt = clip_reflectance(rad_salt)

    return {
        "ice": ice,
        "ocean_salt": ocean_salt,
        "simple_organic": simple_org,
        "tholin_pah": tholin_pah,
        "sulfuric_acid_hydrate": sulfuric_acid_hydrate,
        "sulfur_so2": sulfur_so2,
        "h2o2": h2o2,
        "rad_salt": rad_salt,
    }


def load_csv_spectrum(path: str | Path, wavelengths: np.ndarray, y_col: str | None = None) -> np.ndarray:
    """Load a 2-column CSV spectrum and interpolate it onto the experiment grid.

    Expected columns include `wavelength_um` plus a reflectance/intensity column.
    This is a generic helper for endmember replacement.
    """
    df = pd.read_csv(path)
    if "wavelength_um" not in df.columns:
        raise ValueError("CSV must contain a 'wavelength_um' column.")
    if y_col is None:
        candidates = [c for c in df.columns if c != "wavelength_um"]
        if not candidates:
            raise ValueError("CSV must contain a spectral value column.")
        y_col = candidates[0]
    x = df["wavelength_um"].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    order = np.argsort(x)
    return np.interp(wavelengths, x[order], y[order])
