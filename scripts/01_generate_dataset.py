#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ExperimentConfig
from src.config import NOISE_PRESETS
from src.forward_model import generate_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8000)
    parser.add_argument("--rho-geo", type=float, default=0.75)
    parser.add_argument("--rho-rad", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=3421)
    parser.add_argument("--noise-condition", choices=NOISE_PRESETS.keys(), default="moderate")
    parser.add_argument("--smoothing-window", type=int, default=None)
    parser.add_argument("--out", type=str, default="data/processed/synthetic_dataset.csv")
    args = parser.parse_args()

    cfg = ExperimentConfig(
        seed=args.seed,
        noise_condition=args.noise_condition,
        smoothing_window=args.smoothing_window,
    )
    df = generate_dataset(n=args.n, rho_geo=args.rho_geo, rho_rad=args.rho_rad, cfg=cfg)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")
    print(df["z"].value_counts(normalize=True).round(3))


if __name__ == "__main__":
    main()
