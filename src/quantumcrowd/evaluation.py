"""Forecast evaluation and event-clustered uncertainty."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .models import clip_probability


def probability_metrics(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = clip_probability(probability)
    bin_index = np.minimum((p * bins).astype(int), bins - 1)
    ece = 0.0
    for index in range(bins):
        mask = bin_index == index
        if mask.any():
            ece += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    auc = float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float("nan")
    return {
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece": float(ece),
        "auc": auc,
    }


def calibration_table(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=int)
    p = clip_probability(probability)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.minimum(np.digitize(p, edges[1:-1], right=False), bins - 1)
    rows: list[dict[str, float | int]] = []
    for i in range(bins):
        mask = index == i
        rows.append(
            {
                "bin": i,
                "lower": float(edges[i]),
                "upper": float(edges[i + 1]),
                "count": int(mask.sum()),
                "mean_probability": float(p[mask].mean()) if mask.any() else float("nan"),
                "event_rate": float(y[mask].mean()) if mask.any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def clustered_brier_delta_interval(
    df: pd.DataFrame,
    model_probability: np.ndarray,
    baseline_probability: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 7,
) -> dict[str, float]:
    """Bootstrap Brier(model)-Brier(baseline) by event, not by row."""

    work = df[["event_id", "resolved"]].copy()
    work["model_sq"] = (np.asarray(model_probability) - work["resolved"].to_numpy()) ** 2
    work["baseline_sq"] = (np.asarray(baseline_probability) - work["resolved"].to_numpy()) ** 2
    event_losses = work.groupby("event_id")[["model_sq", "baseline_sq"]].mean()
    event_delta = (event_losses["model_sq"] - event_losses["baseline_sq"]).to_numpy()
    rng = np.random.default_rng(seed)
    sampled = rng.choice(event_delta, size=(n_bootstrap, len(event_delta)), replace=True).mean(axis=1)
    return {
        "delta_brier": float(event_delta.mean()),
        "ci_low": float(np.quantile(sampled, 0.025)),
        "ci_high": float(np.quantile(sampled, 0.975)),
    }

