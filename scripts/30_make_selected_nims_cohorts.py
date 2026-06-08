#!/usr/bin/env python
"""Create fixed core/expanded NIMS observation cohorts from pre-deployment QC."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "manifest"
PDS_BASE = "https://planetarydata.jpl.nasa.gov/img/data/go-e-nims-4-mosaic-v1.0"


EXPANDED = [
    # leading chaos: one existing CI pilot + two OOD-pass TI observations
    ("leading_chaos", "go_1108", "e6e002ci.qub", "E6", "ci", "existing_four_nims", 111, None, 11.881),
    ("leading_chaos", "go_1116", "15e015ti.qub", "15e015", "ti", "candidate_screen", 2640, 0.0075, 7.462),
    ("leading_chaos", "go_1117", "17e009ti.qub", "17e009", "ti", "candidate_screen", 2340, 0.0000, 8.643),
    # leading plains: choose CI products; exclude duplicate 25e006cr and high-missing 25e002ci
    ("leading_plains", "go_1118", "25e006ci.qub", "25e006", "ci", "candidate_screen", 1078, 0.0626, 7.444),
    ("leading_plains", "go_1118", "25e001ci.qub", "25e001", "ci", "candidate_screen", 944, 0.0465, 10.781),
    ("leading_plains", "go_1118", "25e004ci.qub", "25e004", "ci", "candidate_screen", 3262, 0.3708, 15.861),
    # trailing chaos: existing CI pilot + two CI candidates
    ("trailing_chaos", "go_1105", "g2e002ci.qub", "G2", "ci", "existing_four_nims", 53, None, 5.837),
    ("trailing_chaos", "go_1115", "12e003ci.qub", "12e003", "ci", "candidate_screen", 2993, 0.1836, 20.680),
    ("trailing_chaos", "go_1115", "12e001ci.qub", "12e001", "ci", "candidate_screen", 1768, 0.2300, 22.278),
    # trailing plains: existing CI pilot + two independent candidates
    ("trailing_plains", "go_1114", "11e001ci.qub", "E11", "ci", "existing_four_nims", 3431, None, 11.345),
    ("trailing_plains", "go_1117", "17e008ci.qub", "17e008", "ci", "candidate_screen", 4930, 0.4892, 7.154),
    ("trailing_plains", "go_1115", "14e001ci.qub", "14e001", "ci", "candidate_screen", 616, 0.3516, 21.852),
]

CORE_OBS_IDS = {
    "E6",
    "15e015",
    "25e006",
    "25e001",
    "G2",
    "12e003",
    "E11",
    "17e008",
}


def build(rows: list[tuple]) -> pd.DataFrame:
    cols = [
        "group",
        "volume",
        "file",
        "obs_id",
        "product",
        "selection_source",
        "screen_n_valid_pixels",
        "screen_missing_frac",
        "screen_ood_ratio_0p7_2p0",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df["hemisphere"] = df["group"].str.split("_").str[0]
    df["terrain"] = df["group"].str.split("_").str[1]
    df["url"] = PDS_BASE + "/" + df["volume"] + "/europa/" + df["file"]
    df["selection_rule"] = (
        "unique observation; CI cube preferred when available; 0.7-2.0 coverage; "
        "OOD<=25x; then valid pixels high / OOD low; model scores not used"
    )
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    expanded = build(EXPANDED)
    core = expanded[expanded["obs_id"].isin(CORE_OBS_IDS)].copy()
    expanded.to_csv(OUT / "selected_nims_observations_12.csv", index=False)
    core.to_csv(OUT / "selected_nims_observations_8.csv", index=False)
    print("Core cohort:")
    print(core[["group", "obs_id", "volume", "file", "screen_ood_ratio_0p7_2p0"]].to_string(index=False))
    print("\nExpanded cohort:")
    print(expanded[["group", "obs_id", "volume", "file", "screen_ood_ratio_0p7_2p0"]].to_string(index=False))


if __name__ == "__main__":
    main()
