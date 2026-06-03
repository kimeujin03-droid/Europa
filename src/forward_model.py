from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ExperimentConfig, HIDDEN_CLASSES
from .endmembers import toy_endmember_library, clip_reflectance
from .endmember_loader import load_processed_endmembers, processed_endmembers_available


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(v, 0.0) for v in weights.values())
    if total <= 0:
        raise ValueError("Weights must sum to positive value.")
    return {k: max(v, 0.0) / total for k, v in weights.items()}


def mix(endmembers: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    weights = normalize_weights(weights)
    r = np.zeros_like(next(iter(endmembers.values())))
    for name, weight in weights.items():
        r += weight * endmembers[name]
    return clip_reflectance(r)


def degrade_resolution(r: np.ndarray, window: int) -> np.ndarray:
    """Approximate coarse NIMS-like bandpass mixing with a boxcar smoother."""
    if window <= 1:
        return r
    if window % 2 == 0:
        window += 1
    pad = window // 2
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(r, pad_width=pad, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def sample_hidden_class(rng: np.random.Generator) -> str:
    # Rare positive class by design; tweak if class imbalance is too severe.
    probs = np.array([0.18, 0.22, 0.28, 0.20, 0.12])
    return rng.choice(HIDDEN_CLASSES, p=probs)


def beta_near(rng: np.random.Generator, high: bool, strength: float) -> float:
    """Generate proximity/activity value in [0,1]. strength controls separation."""
    if strength <= 0:
        return rng.beta(2, 2)
    if high:
        a = 2 + 6 * strength
        b = 2
    else:
        a = 2
        b = 2 + 6 * strength
    return rng.beta(a, b)


def generate_spatial_features(z: str, rng: np.random.Generator, rho_geo: float, rho_rad: float) -> dict[str, float]:
    ocean_related = z in {"ocean_organic", "ocean_nonorganic"}
    rad_related = z == "radiation_mimic"

    geo = {
        "chaos_proximity": beta_near(rng, ocean_related, rho_geo),
        "lineament_proximity": beta_near(rng, ocean_related, rho_geo),
        "ridge_proximity": beta_near(rng, ocean_related, rho_geo * 0.7),
        "young_terrain": float(rng.random() < (0.2 + 0.55 * rho_geo if ocean_related else 0.2)),
        "activity_proxy": beta_near(rng, ocean_related, rho_geo * 0.8),
    }

    trailing_p = 0.5
    if rad_related:
        trailing_p = 0.5 + 0.45 * rho_rad
    elif ocean_related:
        trailing_p = 0.5 - 0.20 * rho_rad
    trailing = float(rng.random() < np.clip(trailing_p, 0.05, 0.95))

    rad = {
        "trailing_hemisphere": trailing,
        "radiation_exposure": beta_near(rng, rad_related or trailing > 0, rho_rad),
        "sulfur_proxy": beta_near(rng, rad_related or trailing > 0, rho_rad * 0.9),
        "rad_mimic_proxy": beta_near(rng, rad_related, rho_rad),
    }
    return {**geo, **rad}


def generate_spectrum(
    z: str,
    rng: np.random.Generator,
    cfg: ExperimentConfig,
    lib: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, float]]:
    # Draw mixture coefficients by hidden source class.
    if z == "ocean_organic":
        weights = {
            "ice": rng.uniform(0.35, 0.65),
            "ocean_salt": rng.uniform(0.18, 0.40),
            "simple_organic": rng.uniform(0.020, 0.090),
        }
    elif z == "ocean_nonorganic":
        weights = {
            "ice": rng.uniform(0.45, 0.72),
            "ocean_salt": rng.uniform(0.20, 0.45),
            "simple_organic": rng.uniform(0.0, 0.030),
        }
    elif z == "radiation_mimic":
        rad_so_hi = cfg.rad_simple_organic_hi
        weights = {
            "ice": rng.uniform(0.35, 0.65),
            "ocean_salt": rng.uniform(0.16, 0.38),
            "simple_organic": rng.uniform(0.020, rad_so_hi),
            "sulfuric_acid_hydrate": rng.uniform(0.12, 0.28),
            "sulfur_so2": rng.uniform(0.03, 0.13),
            "h2o2": rng.uniform(0.00, 0.08),
            "rad_salt": rng.uniform(0.08, 0.22),
        }
    elif z == "exogenic_complex_organic":
        weights = {
            "ice": rng.uniform(0.38, 0.68),
            "tholin_pah": rng.uniform(0.14, 0.34),
            "ocean_salt": rng.uniform(0.04, 0.18),
            "simple_organic": rng.uniform(0.015, 0.070),
        }
    elif z == "noise_artifact":
        weights = {
            "ice": rng.uniform(0.55, 0.90),
            "ocean_salt": rng.uniform(0.00, 0.08),
            "simple_organic": rng.uniform(0.00, 0.035),
        }
    else:
        raise ValueError(f"Unknown hidden class: {z}")

    normalized_weights = normalize_weights(weights)
    r = mix(lib, normalized_weights)

    # Instrument-like effects: low-frequency slope variation, weak sinusoidal artifact, Gaussian noise.
    w = cfg.wavelengths
    slope = rng.normal(0, 0.015)
    r = r + slope * (w - w.mean())
    if z == "noise_artifact":
        r += rng.normal(0, 0.03) * np.sin(2 * np.pi * w / rng.uniform(0.25, 0.6))
    r += rng.normal(0, cfg.effective_noise_sigma, size=w.size)
    r = degrade_resolution(r, cfg.effective_smoothing_window)
    return clip_reflectance(r), normalized_weights


def generate_dataset(n: int, rho_geo: float, rho_rad: float, cfg: ExperimentConfig | None = None) -> pd.DataFrame:
    cfg = cfg or ExperimentConfig()
    rng = np.random.default_rng(cfg.seed)
    if processed_endmembers_available():
        lib = load_processed_endmembers(cfg.wavelengths)
        endmember_source = "processed"
    else:
        lib = toy_endmember_library(cfg.wavelengths)
        endmember_source = "toy"

    rows = []
    for i in range(n):
        z = sample_hidden_class(rng)
        spectrum, weights = generate_spectrum(z, rng, cfg, lib)
        spatial = generate_spatial_features(z, rng, rho_geo=rho_geo, rho_rad=rho_rad)
        row = {
            "sample_id": i,
            "z": z,
            "y": int(z == "ocean_organic"),
            "rho_geo": rho_geo,
            "rho_rad": rho_rad,
            "endmember_source": endmember_source,
        }
        for name in lib:
            row[f"w_{name}"] = weights.get(name, 0.0)
        row.update(spatial)
        row.update({f"spec_{j:03d}": value for j, value in enumerate(spectrum)})
        rows.append(row)
    return pd.DataFrame(rows)
