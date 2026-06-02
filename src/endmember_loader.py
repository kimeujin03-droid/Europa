from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .spectral_preprocess import interpolate_to_grid


ROOT = Path(__file__).resolve().parents[1]
ENDMEMBER_DIR = ROOT / "data" / "processed" / "endmembers"

ENDMEMBER_FILES = {
    "ice": "ice.csv",
    "ocean_salt": "ocean_salt.csv",
    "simple_organic": "simple_organic.csv",
    "tholin_pah": "tholin_pah.csv",
    "sulfuric_acid_hydrate": "sulfuric_acid_hydrate.csv",
    "sulfur_so2": "sulfur_so2.csv",
    "h2o2": "h2o2.csv",
    "rad_salt": "rad_salt_proxy.csv",
}


def processed_endmember_paths(base_dir: Path = ENDMEMBER_DIR) -> dict[str, Path]:
    return {name: base_dir / filename for name, filename in ENDMEMBER_FILES.items()}


def processed_endmembers_available(base_dir: Path = ENDMEMBER_DIR) -> bool:
    return all(path.exists() for path in processed_endmember_paths(base_dir).values())


def missing_processed_endmembers(base_dir: Path = ENDMEMBER_DIR) -> list[str]:
    return [name for name, path in processed_endmember_paths(base_dir).items() if not path.exists()]


def load_processed_endmembers(
    target_wavelengths: np.ndarray,
    base_dir: Path = ENDMEMBER_DIR,
    intensity_depth: float = 0.08,
) -> dict[str, np.ndarray]:
    missing = missing_processed_endmembers(base_dir)
    if missing:
        raise FileNotFoundError(
            "Missing processed endmember CSVs: "
            + ", ".join(missing)
            + f". Expected files under {base_dir}."
        )

    spectra = {}
    for name, path in processed_endmember_paths(base_dir).items():
        df = pd.read_csv(path)
        spectra[name] = interpolate_to_grid(df, target_wavelengths, intensity_depth=intensity_depth)
    return spectra
