#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ENDMEMBERS = [
    "ice.csv",
    "ocean_salt.csv",
    "simple_organic.csv",
    "tholin_pah.csv",
    "sulfuric_acid_hydrate.csv",
    "sulfur_so2.csv",
    "h2o2.csv",
    "rad_salt_proxy.csv",
]


def main() -> None:
    endmember_dir = ROOT / "data" / "processed" / "endmembers"
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    rows = []
    plt.figure(figsize=(8, 5))
    for filename in ENDMEMBERS:
        path = endmember_dir / filename
        if not path.exists():
            rows.append({"file": filename, "exists": False})
            continue
        df = pd.read_csv(path)
        stats = {
            "file": filename,
            "exists": True,
            "n": len(df),
            "wavelength_min_um": df["wavelength_um"].min(),
            "wavelength_max_um": df["wavelength_um"].max(),
            "value_min": df["value"].min(),
            "value_max": df["value"].max(),
            "nan_count": int(df[["wavelength_um", "value"]].isna().sum().sum()),
        }
        rows.append(stats)
        plt.plot(df["wavelength_um"], df["value"], label=filename.replace(".csv", ""))

    summary = pd.DataFrame(rows)
    summary.to_csv(results_dir / "processed_endmember_validation.csv", index=False)

    plt.xlim(0.7, 5.2)
    plt.xlabel("Wavelength (um)")
    plt.ylabel("Reflectance / proxy reflectance")
    plt.title("Processed endmembers")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(results_dir / "processed_endmembers.png", dpi=200)
    plt.close()

    print(summary.to_string(index=False))
    print(f"Saved validation outputs to {results_dir}")


if __name__ == "__main__":
    main()
