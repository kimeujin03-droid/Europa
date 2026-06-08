#!/usr/bin/env python
"""Create OOD-gated reporting tables for staged NIMS model outputs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "results" / "nims_stage_model"
OUT_DIR = ROOT / "results" / "paper"
STAGE = "02_ocean_salt_usgs_epsomite"


def assign_gate(ood_ratio: float) -> tuple[str, str]:
    """Return reporting gate and permitted interpretation.

    Thresholds are intentionally conservative:
    <=10x: exploratory score reporting is allowed.
    10-25x: report context signs only.
    >25x: report OOD diagnostic only.
    """
    if ood_ratio <= 10.0:
        return "score_reportable", "exploratory score + context sign"
    if ood_ratio <= 25.0:
        return "context_only", "context sign only; calibrated score withheld"
    return "ood_only", "OOD diagnostic only; calibrated score withheld"


def load_summary(tag: str, band_label: str) -> pd.DataFrame:
    path = IN_DIR / f"nims_summary_{STAGE}_{tag}.csv"
    df = pd.read_csv(path)
    df["band_label"] = band_label
    df["summary_file"] = str(path.relative_to(ROOT))
    return df


def main() -> None:
    summaries = pd.concat(
        [
            load_summary("0p7_5p2", "0.7-5.2"),
            load_summary("0p7_2p0", "0.7-2.0"),
        ],
        ignore_index=True,
    )
    gates = summaries["mean_ood_ratio"].apply(assign_gate)
    summaries["ood_gate"] = [gate for gate, _ in gates]
    summaries["permitted_interpretation"] = [interp for _, interp in gates]
    summaries["report_p_full"] = np.where(
        summaries["ood_gate"] == "score_reportable",
        summaries["mean_p_full"].round(6),
        np.nan,
    )
    summaries["report_context_geo_sign"] = np.where(
        summaries["ood_gate"].isin(["score_reportable", "context_only"]),
        np.sign(summaries["delta_geo"]).astype(int),
        np.nan,
    )
    summaries["report_context_rad_sign"] = np.where(
        summaries["ood_gate"].isin(["score_reportable", "context_only"]),
        np.sign(summaries["delta_rad"]).astype(int),
        np.nan,
    )

    keep = [
        "band_label",
        "encounter",
        "hemisphere",
        "terrain",
        "geo_unit",
        "mean_ood_ratio",
        "ood_gate",
        "permitted_interpretation",
        "report_p_full",
        "delta_geo",
        "delta_rad",
        "delta_full",
        "report_context_geo_sign",
        "report_context_rad_sign",
        "val_pr_auc_full",
        "val_roc_auc_full",
    ]
    out = summaries[keep].copy()
    for col in ["mean_ood_ratio", "delta_geo", "delta_rad", "delta_full", "val_pr_auc_full", "val_roc_auc_full"]:
        out[col] = out[col].round(3)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "nims_ood_gated_reporting.csv"
    out.to_csv(out_path, index=False)

    markdown_path = OUT_DIR / "nims_ood_gated_reporting.md"
    markdown_path.write_text(to_markdown(out), encoding="utf-8")
    print(out.to_string(index=False))
    print(f"\nSaved {out_path}")
    print(f"Saved {markdown_path}")


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if pd.isna(value):
                values.append("")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
