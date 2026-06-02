"""Optional real-data loaders to fill in after downloading external data.

This file intentionally contains thin placeholders. It documents where existing GitHub/PyPI tools
should be plugged in when moving from toy synthetic spectra to real NIMS/PDS/lab spectra.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_generic_csv_spectrum(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "wavelength_um" not in df.columns:
        raise ValueError("Expected a wavelength_um column.")
    return df


def load_pds4_label_with_pds4_tools(label_xml: str | Path):
    """Read a PDS4 product using pds4_tools.

    Install first:
        pip install pds4-tools

    Example:
        import pds4_tools
        structures = pds4_tools.read(str(label_xml))
        arr = structures[0].data
    """
    import pds4_tools  # type: ignore
    return pds4_tools.read(str(label_xml))


def open_envi_or_hyperspectral(path: str | Path):
    """Open an ENVI-style hyperspectral image using spectralpython.

    Install first:
        pip install spectral
    """
    import spectral  # type: ignore
    return spectral.open_image(str(path))
