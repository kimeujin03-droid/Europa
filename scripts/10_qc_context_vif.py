#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


OUT_DIR = Path("results/qc")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONTEXT_ALIASES = {
    "d_chaos": "chaos_proximity",
    "d_lineament": "lineament_proximity",
    "d_ridge": "ridge_proximity",
    "terrain_activity": "activity_proxy",
    "trailing_flag": "trailing_hemisphere",
    "radiation_exposure": "radiation_exposure",
    "sulfur_proxy": "sulfur_proxy",
    "mimic_proxy": "rad_mimic_proxy",
}


def compute_vif(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    xall = df[cols].dropna().astype(float)
    for col in cols:
        y = xall[col].to_numpy()
        x = xall[[c for c in cols if c != col]].to_numpy()
        model = LinearRegression()
        model.fit(x, y)
        r2 = float(model.score(x, y))
        vif = np.inf if r2 >= 0.999999 else 1.0 / (1.0 - r2)
        rows.append(
            {
                "feature": col,
                "r2_explained_by_others": r2,
                "vif": vif,
                "flag_vif_gt_5": vif > 5,
                "flag_vif_gt_10": vif > 10,
            }
        )
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/synthetic_dataset_lab_endmember_v1.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    cols = []
    missing = []
    for _paper_name, actual_name in CONTEXT_ALIASES.items():
        if actual_name in df.columns:
            cols.append(actual_name)
        else:
            missing.append(actual_name)

    if missing:
        print("Missing context columns:", missing)
    if len(cols) < 2:
        raise ValueError("Need at least two context columns for VIF")

    corr = df[cols].corr()
    corr.to_csv(OUT_DIR / "context_feature_corr.csv", encoding="utf-8-sig")

    vif = compute_vif(df, cols)
    vif.to_csv(OUT_DIR / "context_feature_vif.csv", index=False, encoding="utf-8-sig")

    print("\nContext feature correlation max abs off-diagonal:")
    corr_abs = corr.abs().where(~np.eye(len(corr), dtype=bool))
    print(float(corr_abs.max().max()))
    print("\nContext VIF:")
    print(vif.to_string(index=False))
    print("\nSaved QC files to:", OUT_DIR)


if __name__ == "__main__":
    main()
