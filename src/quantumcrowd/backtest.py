"""Conservative one-contract paper backtest using executable asks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import clip_probability


def one_contract_backtest(
    df: pd.DataFrame,
    probability: np.ndarray,
    min_edge: float = 0.03,
    fee_per_contract: float = 0.01,
    max_spread: float = 0.15,
) -> pd.DataFrame:
    """Choose at most one YES or NO contract per market snapshot."""

    columns = ["event_id", "market_id", "close_time", "resolved", "yes_bid", "yes_ask"]
    for optional in ("venue", "has_executable_quotes"):
        if optional in df.columns:
            columns.append(optional)
    work = df[columns].copy()
    if "venue" not in work:
        work["venue"] = "unknown"
    if "has_executable_quotes" not in work:
        work["has_executable_quotes"] = True
    work["has_executable_quotes"] = work["has_executable_quotes"].fillna(False).astype(bool)
    work["probability"] = clip_probability(probability)
    work["spread"] = (work["yes_ask"] - work["yes_bid"]).clip(lower=0.0)
    work["no_ask"] = (1.0 - work["yes_bid"]).clip(0.001, 0.999)
    work["yes_edge"] = work["probability"] - work["yes_ask"] - fee_per_contract
    work["no_edge"] = (1.0 - work["probability"]) - work["no_ask"] - fee_per_contract

    best_is_yes = work["yes_edge"] >= work["no_edge"]
    work["side"] = np.where(best_is_yes, "YES", "NO")
    work["estimated_edge"] = np.where(best_is_yes, work["yes_edge"], work["no_edge"])
    trade_mask = (
        (work["estimated_edge"] >= min_edge)
        & (work["spread"] <= max_spread)
        & work["has_executable_quotes"]
    )
    work["pass_reason"] = ""
    work.loc[~work["has_executable_quotes"], "pass_reason"] = "non_executable_history"
    work.loc[
        work["has_executable_quotes"] & (work["spread"] > max_spread), "pass_reason"
    ] = "spread"
    work.loc[
        work["has_executable_quotes"]
        & (work["spread"] <= max_spread)
        & (work["estimated_edge"] < min_edge),
        "pass_reason",
    ] = "edge"
    work.loc[~trade_mask, "side"] = "PASS"

    yes_pnl = work["resolved"] - work["yes_ask"] - fee_per_contract
    no_pnl = (1 - work["resolved"]) - work["no_ask"] - fee_per_contract
    work["pnl"] = np.where(
        work["side"] == "YES", yes_pnl, np.where(work["side"] == "NO", no_pnl, 0.0)
    )
    work = work.sort_values("close_time").reset_index(drop=True)
    work["cumulative_pnl"] = work["pnl"].cumsum()
    return work


def backtest_summary(trades: pd.DataFrame) -> dict[str, float | int]:
    selected = trades[trades["side"] != "PASS"]
    cumulative = trades["cumulative_pnl"].to_numpy(dtype=float)
    running_max = np.maximum.accumulate(np.concatenate([[0.0], cumulative]))[1:]
    drawdown = cumulative - running_max
    return {
        "markets": int(len(trades)),
        "trades": int(len(selected)),
        "coverage": float(len(selected) / len(trades)) if len(trades) else 0.0,
        "net_pnl": float(selected["pnl"].sum()),
        "mean_pnl_per_trade": float(selected["pnl"].mean()) if len(selected) else 0.0,
        "win_rate": float((selected["pnl"] > 0).mean()) if len(selected) else 0.0,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
    }
