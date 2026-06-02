from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

GEO_COLS = ["chaos_proximity", "lineament_proximity", "ridge_proximity", "young_terrain", "activity_proxy"]
RAD_COLS = ["trailing_hemisphere", "radiation_exposure", "sulfur_proxy", "rad_mimic_proxy"]


def spectral_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("spec_")]


def feature_cols(df: pd.DataFrame, setting: str) -> list[str]:
    if setting == "context_only":
        return GEO_COLS + RAD_COLS

    cols = spectral_cols(df)
    if setting in {"spectral_geology", "full"}:
        cols += GEO_COLS
    if setting in {"spectral_radiation", "full"}:
        cols += RAD_COLS
    return cols


def build_model(name: str, calibrate: bool = True, seed: int = 3421):
    if name == "logreg":
        base = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        ])
    elif name == "rf":
        base = RandomForestClassifier(
            n_estimators=150,
            max_depth=18,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=seed,
        )
    elif name == "hgb":
        base = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            l2_regularization=0.01,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model: {name}")

    if calibrate and name in {"rf", "hgb"}:
        return CalibratedClassifierCV(base, method="isotonic", cv=3)
    return base


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k_frac: float = 0.1) -> float:
    k = max(1, int(len(y_true) * k_frac))
    idx = np.argsort(y_score)[::-1][:k]
    return float(np.mean(y_true[idx]))


def fpr_on_class(df_test: pd.DataFrame, y_score: np.ndarray, class_name: str, threshold: float) -> float:
    mask = df_test["z"].to_numpy() == class_name
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(y_score[mask] >= threshold))


def threshold_for_top_k(y_score: np.ndarray, k_frac: float = 0.1) -> float:
    k = max(1, int(len(y_score) * k_frac))
    return float(np.sort(y_score)[::-1][k - 1])


def top_k_class_counts(df_test: pd.DataFrame, y_score: np.ndarray, k_frac: float = 0.1) -> dict[str, int]:
    k = max(1, int(len(y_score) * k_frac))
    top = df_test.iloc[np.argsort(y_score)[::-1][:k]]
    counts = top["z"].value_counts()
    return {
        "top10_positive_count": int(counts.get("ocean_organic", 0)),
        "top10_radiation_mimic_count": int(counts.get("radiation_mimic", 0)),
        "top10_exogenic_count": int(counts.get("exogenic_complex_organic", 0)),
        "top10_noise_count": int(counts.get("noise_artifact", 0)),
        "top10_ocean_nonorganic_count": int(counts.get("ocean_nonorganic", 0)),
    }


def evaluate_setting(
    df: pd.DataFrame,
    setting: str,
    model_name: str = "rf",
    seed: int = 3421,
    test_size: float = 0.30,
) -> tuple[dict, pd.DataFrame, object]:
    cols = feature_cols(df, setting)
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["y"],
        random_state=seed,
    )
    X_train = train_df[cols].to_numpy()
    y_train = train_df["y"].to_numpy()
    X_test = test_df[cols].to_numpy()
    y_test = test_df["y"].to_numpy()

    model = build_model(model_name, calibrate=True, seed=seed)
    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)[:, 1]
    threshold = threshold_for_top_k(y_score, k_frac=0.1)
    y_pred_top = (y_score >= threshold).astype(int)

    metrics = {
        "setting": setting,
        "model": model_name,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "positive_rate_test": float(y_test.mean()),
        "pr_auc": float(average_precision_score(y_test, y_score)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "precision_at_10pct": precision_at_k(y_test, y_score, 0.1),
        "recall_at_top10pct": float(recall_score(y_test, y_pred_top, zero_division=0)),
        "brier": float(brier_score_loss(y_test, y_score)),
        "fpr_radiation_mimic_top10pct": fpr_on_class(test_df, y_score, "radiation_mimic", threshold),
        "fpr_exogenic_complex_organic_top10pct": fpr_on_class(test_df, y_score, "exogenic_complex_organic", threshold),
    }
    metrics.update(top_k_class_counts(test_df, y_score, 0.1))
    preds = test_df[["sample_id", "z", "y"]].copy()
    preds["score"] = y_score
    preds["setting"] = setting
    preds["model"] = model_name
    return metrics, preds, model


def pr_curve_data(y_true: np.ndarray, y_score: np.ndarray) -> pd.DataFrame:
    p, r, t = precision_recall_curve(y_true, y_score)
    # Threshold array is one shorter than p/r.
    out = pd.DataFrame({"precision": p, "recall": r})
    out["threshold"] = np.r_[t, np.nan]
    return out
