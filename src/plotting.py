from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def save_pr_plot(curves: dict[str, pd.DataFrame], out_path: str):
    plt.figure(figsize=(6, 4))
    for label, df in curves.items():
        plt.plot(df["recall"], df["precision"], label=label)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_prior_heatmap(df: pd.DataFrame, value_col: str, out_path: str):
    pivot = df.pivot_table(index="rho_geo", columns="rho_rad", values=value_col, aggfunc="mean")
    plt.figure(figsize=(6, 5))
    im = plt.imshow(pivot.values, origin="lower", aspect="auto")
    plt.colorbar(im, label=value_col)
    plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
    plt.yticks(range(len(pivot.index)), [str(i) for i in pivot.index])
    plt.xlabel("rho_rad")
    plt.ylabel("rho_geo")
    plt.title(value_col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
