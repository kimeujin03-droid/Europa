#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


END_DIR = Path("data/processed/endmembers")
OUT_DIR = Path("results/qc")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ENDMEMBERS = [
    "ice",
    "ocean_salt",
    "simple_organic",
    "tholin_pah",
    "sulfuric_acid_hydrate",
    "sulfur_so2",
    "h2o2",
    "rad_salt_proxy",
]

META_COLS = {"wavelength_um", "kind", "source_id", "source_name", "notes"}


def read_endmember(name: str) -> pd.DataFrame:
    path = END_DIR / f"{name}.csv"
    df = pd.read_csv(path)
    if "wavelength_um" not in df.columns:
        raise ValueError(f"{path} must contain wavelength_um")
    if "value" in df.columns:
        value_col = "value"
    else:
        value_cols = [c for c in df.columns if c not in META_COLS]
        if not value_cols:
            raise ValueError(f"{path} must contain a spectral value column")
        value_col = value_cols[0]
    return df[["wavelength_um", value_col]].rename(columns={value_col: name})


def spectral_angle_deg(x: np.ndarray, y: np.ndarray, centered: bool = False) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if centered:
        x = x - np.mean(x)
        y = y - np.mean(y)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return np.nan
    cosv = np.clip(np.dot(x, y) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))


def main() -> None:
    merged = None
    for name in ENDMEMBERS:
        df = read_endmember(name)
        merged = df if merged is None else pd.merge(merged, df, on="wavelength_um", how="inner")

    if merged is None or merged.empty:
        raise ValueError("No common endmember wavelength grid found")

    merged = merged.sort_values("wavelength_um")
    merged.to_csv(OUT_DIR / "endmember_matrix_common_grid.csv", index=False)

    x = merged[ENDMEMBERS]
    corr = x.corr(method="pearson")
    corr.to_csv(OUT_DIR / "endmember_pearson_corr.csv", encoding="utf-8-sig")

    angle = pd.DataFrame(index=ENDMEMBERS, columns=ENDMEMBERS, dtype=float)
    centered_angle = pd.DataFrame(index=ENDMEMBERS, columns=ENDMEMBERS, dtype=float)
    for a in ENDMEMBERS:
        for b in ENDMEMBERS:
            angle.loc[a, b] = spectral_angle_deg(x[a].to_numpy(), x[b].to_numpy())
            centered_angle.loc[a, b] = spectral_angle_deg(x[a].to_numpy(), x[b].to_numpy(), centered=True)
    angle.to_csv(OUT_DIR / "endmember_spectral_angle_deg.csv", encoding="utf-8-sig")
    centered_angle.to_csv(OUT_DIR / "endmember_centered_spectral_angle_deg.csv", encoding="utf-8-sig")

    rows = []
    for i, a in enumerate(ENDMEMBERS):
        for b in ENDMEMBERS[i + 1 :]:
            r = float(corr.loc[a, b])
            theta = float(angle.loc[a, b])
            centered_theta = float(centered_angle.loc[a, b])
            rows.append(
                {
                    "a": a,
                    "b": b,
                    "pearson_r": r,
                    "abs_pearson_r": abs(r),
                    "spectral_angle_deg": theta,
                    "centered_spectral_angle_deg": centered_theta,
                    "flag_high_corr": abs(r) > 0.95,
                    "flag_low_angle": theta < 5.0,
                    "flag_low_centered_angle": centered_theta < 5.0,
                }
            )

    pair_df = pd.DataFrame(rows).sort_values(
        ["flag_high_corr", "flag_low_angle", "abs_pearson_r"], ascending=False
    )
    pair_df.to_csv(OUT_DIR / "endmember_similarity_pairs.csv", index=False, encoding="utf-8-sig")

    print("\nTop similar endmember pairs:")
    print(pair_df.head(15).to_string(index=False))
    print("\nSaved QC files to:", OUT_DIR)


if __name__ == "__main__":
    main()
