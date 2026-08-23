"""Chronological event-grouped data splitting."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class EventSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def chronological_event_split(
    df: pd.DataFrame,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> EventSplit:
    """Keep every market from an event in one chronological split."""

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be in (0, 1)")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be less than 1")

    required = {"event_id", "close_time"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing split columns: {sorted(required - set(df.columns))}")

    event_order = (
        df.assign(close_time=pd.to_datetime(df["close_time"], utc=True))
        .groupby("event_id", as_index=False)["close_time"]
        .max()
        .sort_values(["close_time", "event_id"])
    )
    n_events = len(event_order)
    if n_events < 10:
        raise ValueError("At least 10 distinct events are required")

    train_end = max(1, int(n_events * train_fraction))
    val_end = max(train_end + 1, int(n_events * (train_fraction + validation_fraction)))
    val_end = min(val_end, n_events - 1)

    train_events = set(event_order.iloc[:train_end]["event_id"])
    val_events = set(event_order.iloc[train_end:val_end]["event_id"])
    test_events = set(event_order.iloc[val_end:]["event_id"])

    return EventSplit(
        train=df[df["event_id"].isin(train_events)].copy(),
        validation=df[df["event_id"].isin(val_events)].copy(),
        test=df[df["event_id"].isin(test_events)].copy(),
    )

