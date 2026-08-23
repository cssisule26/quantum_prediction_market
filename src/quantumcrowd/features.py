"""Leakage-conscious market-microstructure feature construction."""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "event_id",
    "market_id",
    "timestamp",
    "close_time",
    "market_prob",
    "yes_bid",
    "yes_ask",
    "volume",
    "open_interest",
    "resolved",
}

FEATURE_COLUMNS = [
    "venue_polymarket",
    "quote_available",
    "volume_available",
    "open_interest_available",
    "spread",
    "price_return_1",
    "price_return_3",
    "volatility_3",
    "log_volume",
    "volume_change",
    "log_open_interest",
    "signed_flow_proxy",
    "kyle_lambda_proxy",
    "tail_distance",
    "log_hours_to_close",
]


def _check_columns(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create only backward-looking features within each market."""

    _check_columns(df)
    out = df.copy()
    if "venue" not in out:
        out["venue"] = "unknown"
    if "has_executable_quotes" not in out:
        out["has_executable_quotes"] = True
    if "volume_available" not in out:
        out["volume_available"] = True
    if "open_interest_available" not in out:
        out["open_interest_available"] = True
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    out["close_time"] = pd.to_datetime(out["close_time"], utc=True)
    out = out.sort_values(["market_id", "timestamp"]).reset_index(drop=True)

    for column in ["market_prob", "yes_bid", "yes_ask", "volume", "open_interest"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out["market_prob"] = out["market_prob"].clip(0.001, 0.999)
    out["venue_polymarket"] = (out["venue"].astype(str).str.lower() == "polymarket").astype(float)
    out["quote_available"] = out["has_executable_quotes"].fillna(False).astype(float)
    out["volume_available"] = out["volume_available"].fillna(False).astype(float)
    out["open_interest_available"] = out["open_interest_available"].fillna(False).astype(float)
    out["spread"] = (out["yes_ask"] - out["yes_bid"]).clip(lower=0.0)
    out["hours_to_close"] = (
        (out["close_time"] - out["timestamp"]).dt.total_seconds() / 3600.0
    ).clip(lower=0.0)

    grouped = out.groupby("market_id", sort=False, group_keys=False)
    out["price_return_1"] = grouped["market_prob"].diff()
    out["price_return_3"] = out["market_prob"] - grouped["market_prob"].shift(3)
    out["volume_change"] = grouped["volume"].diff()
    out["volatility_3"] = grouped["price_return_1"].transform(
        lambda series: series.rolling(3, min_periods=2).std()
    )
    out["log_volume"] = np.log1p(out["volume"].clip(lower=0.0))
    out["log_open_interest"] = np.log1p(out["open_interest"].clip(lower=0.0))
    out["signed_flow_proxy"] = np.sign(out["price_return_1"]) * out["log_volume"]
    out["kyle_lambda_proxy"] = out["price_return_1"].abs() / np.sqrt(
        out["volume"].clip(lower=0.0) + 1.0
    )
    out["tail_distance"] = (out["market_prob"] - 0.5).abs()
    out["log_hours_to_close"] = np.log1p(out["hours_to_close"])

    out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["resolved"] = pd.to_numeric(out["resolved"], errors="raise").astype(int)
    if not out["resolved"].isin([0, 1]).all():
        raise ValueError("resolved must contain only 0 and 1")
    return out


def select_forecast_horizon(df: pd.DataFrame, horizon_hours: float = 24.0) -> pd.DataFrame:
    """Select one snapshot per market, closest to a pre-specified horizon."""

    if horizon_hours < 0:
        raise ValueError("horizon_hours must be non-negative")
    if "hours_to_close" not in df.columns:
        raise ValueError("Run build_features before select_forecast_horizon")

    candidates = df.loc[df["hours_to_close"] >= 0].copy()
    candidates["_horizon_distance"] = (candidates["hours_to_close"] - horizon_hours).abs()
    selected = (
        candidates.sort_values(["market_id", "_horizon_distance", "timestamp"])
        .groupby("market_id", as_index=False, sort=False)
        .head(1)
        .drop(columns="_horizon_distance")
        .sort_values("close_time")
        .reset_index(drop=True)
    )
    return selected
