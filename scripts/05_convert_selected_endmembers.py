#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ExperimentConfig
from src.endmembers import gaussian_absorption, toy_endmember_library
from src.relab_loader import read_relab_tab
from src.spectral_preprocess import interpolate_to_grid


REQUIRED = {
    "ice",
    "ocean_salt",
    "simple_organic",
    "tholin_pah",
    "sulfuric_acid_hydrate",
    "sulfur_so2",
    "h2o2",
    "rad_salt_proxy",
}

PROXY_DATABASES = {"parametric_proxy", "derived_proxy", "hybrid_proxy", "literature_proxy"}


def full_range_continuum(wavelengths: np.ndarray, intercept: float = 0.92, slope: float = 0.0) -> np.ndarray:
    return intercept + slope * (wavelengths - wavelengths.mean())


def simple_organic_proxy(wavelengths: np.ndarray) -> np.ndarray:
    values = full_range_continuum(wavelengths, intercept=0.92, slope=0.004)
    for center, width, depth in [
        (3.35, 0.055, 0.055),
        (3.45, 0.070, 0.045),
        (4.42, 0.120, 0.035),
    ]:
        values -= gaussian_absorption(wavelengths, center, width, depth)
    return np.clip(values, 0.05, 1.2)


def ocean_salt_hybrid_proxy(row: pd.Series, wavelengths: np.ndarray) -> np.ndarray:
    raw_xml = str(row.get("raw_xml", "") or "").strip()
    raw_tab = str(row.get("raw_tab", "") or "").strip()
    if raw_xml and raw_tab:
        source = read_relab_tab(ROOT / raw_tab, ROOT / raw_xml)
        short_range = interpolate_to_grid(source, wavelengths)
    else:
        short_range = full_range_continuum(wavelengths, intercept=0.82, slope=-0.006)

    proxy = full_range_continuum(wavelengths, intercept=0.82, slope=-0.006)
    for center, width, depth in [
        (1.45, 0.10, 0.07),
        (1.95, 0.12, 0.10),
        (3.05, 0.28, 0.16),
        (4.05, 0.18, 0.035),
    ]:
        proxy -= gaussian_absorption(wavelengths, center, width, depth)

    # Preserve real RELAB structure where coverage is present, use smooth proxy elsewhere.
    real_mask = (wavelengths >= source["wavelength_um"].min()) & (wavelengths <= source["wavelength_um"].max()) if raw_xml and raw_tab else np.zeros_like(wavelengths, dtype=bool)
    values = proxy.copy()
    values[real_mask] = 0.65 * short_range[real_mask] + 0.35 * proxy[real_mask]
    return np.clip(values, 0.05, 1.2)


def read_optional_raw_csv(row: pd.Series) -> pd.DataFrame | None:
    raw_csv = str(row.get("raw_csv", "") or "").strip()
    if not raw_csv:
        return None
    return pd.read_csv(ROOT / raw_csv)


def ice_hybrid_proxy(row: pd.Series, wavelengths: np.ndarray) -> np.ndarray:
    toy = toy_endmember_library(wavelengths)["ice"]
    source = read_optional_raw_csv(row)
    if source is None:
        return toy
    real = interpolate_to_grid(source, wavelengths)
    real_mask = (wavelengths >= source["wavelength_um"].min()) & (wavelengths <= source["wavelength_um"].max())
    values = toy.copy()
    values[real_mask] = 0.70 * real[real_mask] + 0.30 * toy[real_mask]
    return np.clip(values, 0.05, 1.2)


def tholin_pah_hybrid_proxy(row: pd.Series, wavelengths: np.ndarray) -> np.ndarray:
    toy = toy_endmember_library(wavelengths)["tholin_pah"]
    source = read_optional_raw_csv(row)
    if source is None:
        return toy
    real = interpolate_to_grid(source, wavelengths)
    real_mask = (wavelengths >= source["wavelength_um"].min()) & (wavelengths <= source["wavelength_um"].max())
    values = toy.copy()
    values[real_mask] = 0.75 * real[real_mask] + 0.25 * toy[real_mask]
    return np.clip(values, 0.05, 1.2)


def sulfuric_acid_hydrate_literature_shape_proxy(wavelengths: np.ndarray) -> np.ndarray:
    """Shape-constrained hydrated sulfuric-acid proxy for Europa radiolysis tests.

    This is not a tabulated Carlson/Loeffler spectrum. It is a transparent literature
    shape proxy with broad hydrate/sulfuric-acid-like absorption structure until a
    digitized figure or public table is supplied.
    """
    values = full_range_continuum(wavelengths, intercept=0.93, slope=-0.018)
    for center, width, depth in [
        (1.50, 0.15, 0.060),
        (1.95, 0.18, 0.075),
        (2.75, 0.22, 0.055),
        (3.05, 0.38, 0.260),
        (3.90, 0.30, 0.055),
        (4.45, 0.32, 0.040),
    ]:
        values -= gaussian_absorption(wavelengths, center, width, depth)
    return np.clip(values, 0.05, 1.2)


def h2o2_literature_marker_proxy(wavelengths: np.ndarray) -> np.ndarray:
    values = full_range_continuum(wavelengths, intercept=0.92, slope=-0.004)
    for center, width, depth in [
        (3.50, 0.075, 0.040),
        (2.85, 0.120, 0.018),
    ]:
        values -= gaussian_absorption(wavelengths, center, width, depth)
    return np.clip(values, 0.05, 1.2)


def sulfur_so2_proxy(wavelengths: np.ndarray) -> np.ndarray:
    values = full_range_continuum(wavelengths, intercept=0.86, slope=-0.004)
    for center, width, depth in [
        (2.05, 0.070, 0.030),
        (3.95, 0.090, 0.055),
        (4.08, 0.090, 0.050),
    ]:
        values -= gaussian_absorption(wavelengths, center, width, depth)
    return np.clip(values, 0.05, 1.2)


def sulfur_so2_hybrid_proxy(row: pd.Series, wavelengths: np.ndarray) -> np.ndarray:
    """Use sulfur-bearing RELAB structure where available, with weak full-range proxy support."""
    proxy = sulfur_so2_proxy(wavelengths)
    csv_source = read_optional_raw_csv(row)
    if csv_source is not None:
        source = csv_source
        relab = interpolate_to_grid(source, wavelengths)
        real_mask = (wavelengths >= source["wavelength_um"].min()) & (wavelengths <= source["wavelength_um"].max())
        values = proxy.copy()
        values[real_mask] = 0.45 * relab[real_mask] + 0.55 * proxy[real_mask]
        return np.clip(values, 0.05, 1.2)

    raw_xml = str(row.get("raw_xml", "") or "").strip()
    raw_tab = str(row.get("raw_tab", "") or "").strip()
    if not raw_xml or not raw_tab:
        return proxy

    source = read_relab_tab(ROOT / raw_tab, ROOT / raw_xml)
    relab = interpolate_to_grid(source, wavelengths)
    real_mask = (wavelengths >= source["wavelength_um"].min()) & (wavelengths <= source["wavelength_um"].max())
    values = proxy.copy()
    # Keep this as a weak auxiliary radiation template rather than letting the lab spectrum dominate.
    values[real_mask] = 0.40 * relab[real_mask] + 0.60 * proxy[real_mask]
    return np.clip(values, 0.05, 1.2)


def normalize_columns(df: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    out = df[["wavelength_um", "value"]].copy()
    out["kind"] = row["kind"]
    out["source_id"] = row["spectrum_id"]
    out["source_name"] = row["source_name"]
    out["notes"] = row["notes"]
    return out


def generated_proxy(row: pd.Series) -> pd.DataFrame:
    wavelengths = ExperimentConfig().wavelengths
    toy = toy_endmember_library(wavelengths)
    output_name = row["output_name"]

    if output_name == "ice":
        values = ice_hybrid_proxy(row, wavelengths)
    elif output_name == "simple_organic":
        values = simple_organic_proxy(wavelengths)
    elif output_name == "ocean_salt":
        values = ocean_salt_hybrid_proxy(row, wavelengths)
    elif output_name == "tholin_pah":
        values = tholin_pah_hybrid_proxy(row, wavelengths)
    elif output_name == "sulfur_so2":
        values = sulfur_so2_hybrid_proxy(row, wavelengths)
    elif output_name == "h2o2":
        values = h2o2_literature_marker_proxy(wavelengths)
    elif output_name == "sulfuric_acid_hydrate":
        values = sulfuric_acid_hydrate_literature_shape_proxy(wavelengths)
    elif output_name == "rad_salt_proxy":
        endmember_dir = ROOT / "data" / "processed" / "endmembers"
        salt_path = endmember_dir / "ocean_salt.csv"
        acid_path = endmember_dir / "sulfuric_acid_hydrate.csv"
        if salt_path.exists() and acid_path.exists():
            salt = interpolate_to_grid(pd.read_csv(salt_path), wavelengths)
            acid = interpolate_to_grid(pd.read_csv(acid_path), wavelengths)
            values = 0.5 * salt + 0.5 * acid
        else:
            values = toy["rad_salt"]
    else:
        raise ValueError(f"No generated proxy rule for {output_name}")

    return pd.DataFrame({"wavelength_um": wavelengths, "value": values})


def validate_source_coverage(source: pd.DataFrame, row: pd.Series) -> None:
    source_database = str(row.get("source_database", "") or "").strip()
    output_name = row["output_name"]
    wavelengths = source["wavelength_um"].to_numpy(dtype=float)
    values = source["value"].to_numpy(dtype=float)

    if source_database not in PROXY_DATABASES:
        if wavelengths.min() > 0.75 or wavelengths.max() < 5.0:
            raise ValueError(
                f"{output_name} coverage insufficient for non-proxy source: "
                f"{wavelengths.min():.3f}-{wavelengths.max():.3f} um"
            )
    if np.nanmin(values) < 0.02:
        print(f"WARNING: {output_name} near-zero reflectance/proxy detected: min={np.nanmin(values):.4f}")
    if np.nanmax(values) > 1.5:
        print(f"WARNING: {output_name} high reflectance/proxy detected: max={np.nanmax(values):.4f}")


def read_source(row: pd.Series) -> pd.DataFrame:
    raw_format = str(row.get("raw_format", "") or "").strip().lower()
    raw_csv = str(row.get("raw_csv", "") or "").strip()
    raw_xml = str(row.get("raw_xml", "") or "").strip()
    raw_tab = str(row.get("raw_tab", "") or "").strip()

    if raw_format in {"generated", "derived", "derived_proxy", "parametric_proxy"}:
        return generated_proxy(row)
    if raw_csv:
        return pd.read_csv(ROOT / raw_csv)
    if raw_xml and raw_tab:
        return read_relab_tab(ROOT / raw_tab, ROOT / raw_xml)
    raise ValueError(f"No usable raw source for {row['output_name']}. Fill raw_csv or raw_xml/raw_tab.")


def main() -> None:
    selection_path = ROOT / "data" / "manifest" / "endmember_selection.csv"
    selection = pd.read_csv(selection_path).fillna("")

    seen = set(selection["output_name"])
    missing_rows = REQUIRED - seen
    if missing_rows:
        raise ValueError(f"Selection file missing rows for: {sorted(missing_rows)}")

    out_dir = ROOT / "data" / "processed" / "endmembers"
    out_dir.mkdir(parents=True, exist_ok=True)

    converted = []
    for _, row in selection.iterrows():
        output_name = row["output_name"]
        if output_name not in REQUIRED:
            continue
        source = read_source(row)
        validate_source_coverage(source, row)
        out = normalize_columns(source, row)
        filename = "rad_salt_proxy.csv" if output_name == "rad_salt_proxy" else f"{output_name}.csv"
        out.to_csv(out_dir / filename, index=False)
        converted.append(filename)

    print("Converted processed endmembers:")
    for filename in converted:
        print(f"  {filename}")


if __name__ == "__main__":
    main()
