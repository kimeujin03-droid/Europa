#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
USGS = ROOT / "data" / "raw" / "usgs_splib07" / "ASCIIdata_splib07a"
OUT = ROOT / "data" / "raw" / "usgs_splib07" / "processed_candidates"


def read_usgs_ascii(path: Path) -> pd.Series:
    values = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        next(handle)
        for line in handle:
            text = line.strip()
            if not text:
                continue
            value = float(text)
            if value < -1e20:
                values.append(float("nan"))
            else:
                values.append(value)
    return pd.Series(values, dtype=float)


def make_candidate(spectrum_rel: str, wavelength_rel: str, out_name: str, source_name: str, kind: str) -> None:
    spectrum = read_usgs_ascii(USGS / spectrum_rel)
    wavelengths = read_usgs_ascii(USGS / wavelength_rel)
    n = min(len(spectrum), len(wavelengths))
    df = pd.DataFrame(
        {
            "wavelength_um": wavelengths.iloc[:n].to_numpy(),
            "value": spectrum.iloc[:n].to_numpy(),
            "kind": kind,
            "source_id": out_name.replace(".csv", ""),
            "source_name": source_name,
            "notes": "USGS Spectral Library v7 ASCII extraction",
        }
    )
    df = df.dropna(subset=["wavelength_um", "value"])
    df = df[(df["wavelength_um"] > 0) & (df["value"] > -1e20)]
    df = df.sort_values("wavelength_um")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / out_name, index=False)
    print(out_name, len(df), f"{df.wavelength_um.min():.3f}-{df.wavelength_um.max():.3f} um", f"{df.value.min():.4f}-{df.value.max():.4f}")


def main() -> None:
    make_candidate(
        "ChapterL_Liquids/splib07a_H2O-Ice_GDS136_77K_BECKa_AREF.txt",
        "splib07a_Wavelengths_BECK_Beckman_0.2-3.0_microns.txt",
        "usgs_h2o_ice_77k_beck.csv",
        "USGS H2O-Ice GDS136 77K BECKa",
        "reflectance",
    )
    make_candidate(
        "ChapterO_OrganicCompounds/splib07a_2-3-Benzanthracene_SA-B2403_NIC4aa_RREF.txt",
        "splib07a_Wavelengths_NIC4_Nicolet_1.12-216microns.txt",
        "usgs_benzanthracene_nic4.csv",
        "USGS 2-3-Benzanthracene SA-B2403 NIC4aa",
        "reflectance",
    )
    make_candidate(
        "ChapterO_OrganicCompounds/splib07a_2-Methylnapthalen_SA-442359_NIC4aa_RREF.txt",
        "splib07a_Wavelengths_NIC4_Nicolet_1.12-216microns.txt",
        "usgs_2_methylnaphthalene_nic4.csv",
        "USGS 2-Methylnaphthalene SA-442359 NIC4aa",
        "reflectance",
    )
    make_candidate(
        "ChapterM_Minerals/splib07a_Sulfur_GDS94_Reagent_NIC4aa_RREF.txt",
        "splib07a_Wavelengths_NIC4_Nicolet_1.12-216microns.txt",
        "usgs_sulfur_nic4.csv",
        "USGS Sulfur GDS94 Reagent NIC4aa",
        "reflectance",
    )


if __name__ == "__main__":
    main()
