"""Synthetic prediction markets for a reproducible, network-free smoke test."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(x)))


def generate_synthetic_markets(
    n_markets: int = 300,
    seed: int = 7,
    horizons: tuple[int, ...] = (336, 168, 72, 24, 6),
) -> pd.DataFrame:
    """Generate market snapshots with informed-flow and favorite-longshot effects.

    The generator is not evidence for performance. It only provides a stable dataset
    for testing the end-to-end research workflow before real data are collected.
    """

    if n_markets < 20:
        raise ValueError("n_markets must be at least 20 for chronological splitting")

    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2022-01-01", tz="UTC")
    rows: list[dict[str, object]] = []

    for i in range(n_markets):
        category = i % 4
        event_id = f"event-{i // 2:05d}" if i % 10 in (0, 1) else f"event-{i:05d}"
        market_id = f"market-{i:05d}"
        close_time = start + pd.Timedelta(days=2 * i + 30)

        public_signal = rng.normal()
        private_signal = rng.normal(scale=0.8)
        category_effect = (-0.35, 0.05, 0.25, -0.1)[category]
        true_logit = -0.15 + 0.9 * public_signal + 0.55 * private_signal + category_effect
        true_prob = float(_sigmoid(true_logit))
        resolved = int(rng.random() < true_prob)

        previous_mid = float(np.clip(_sigmoid(0.45 * public_signal), 0.03, 0.97))
        cumulative_volume = 0.0
        open_interest = float(rng.uniform(20, 100))

        for step, hours_to_close in enumerate(horizons):
            progress = (step + 1) / len(horizons)
            informed_share = 0.15 + 0.65 * progress
            noise = rng.normal(scale=0.8 * (1.0 - 0.65 * progress))
            belief_logit = (
                (1.0 - informed_share) * (0.45 * public_signal)
                + informed_share * true_logit
                + noise
            )

            # Mild longshot overpricing and favorite underpricing.
            raw_prob = float(_sigmoid(belief_logit))
            tail_pull = 0.06 * np.sign(0.5 - raw_prob) * abs(raw_prob - 0.5)
            midpoint = float(np.clip(raw_prob + tail_pull, 0.015, 0.985))

            interval_volume = float(rng.gamma(2.0, 15.0) * (0.5 + 2.0 * progress))
            cumulative_volume += interval_volume
            open_interest += float(rng.uniform(0.1, 0.5) * interval_volume)
            spread = float(np.clip(0.09 / np.sqrt(1.0 + interval_volume / 10.0), 0.01, 0.09))
            yes_bid = float(np.clip(midpoint - spread / 2.0, 0.001, 0.999))
            yes_ask = float(np.clip(midpoint + spread / 2.0, 0.001, 0.999))

            rows.append(
                {
                    "venue": "synthetic",
                    "source_event_id": event_id,
                    "source_market_id": market_id,
                    "event_id": event_id,
                    "market_id": market_id,
                    "category": f"category-{category}",
                    "timestamp": close_time - pd.Timedelta(hours=hours_to_close),
                    "close_time": close_time,
                    "hours_to_close": float(hours_to_close),
                    "market_prob": midpoint,
                    "yes_bid": yes_bid,
                    "yes_ask": yes_ask,
                    "quote_source": "synthetic_book",
                    "has_executable_quotes": True,
                    "volume": interval_volume,
                    "cumulative_volume": cumulative_volume,
                    "open_interest": open_interest,
                    "volume_available": True,
                    "open_interest_available": True,
                    "resolved": resolved,
                    "synthetic_true_prob": true_prob,
                    "synthetic_price_change": midpoint - previous_mid,
                }
            )
            previous_mid = midpoint

    return pd.DataFrame(rows).sort_values(["timestamp", "market_id"]).reset_index(drop=True)
