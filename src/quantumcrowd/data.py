"""Shared snapshot schema and safe multi-venue concatenation."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


PROVENANCE_COLUMNS = [
    "venue",
    "source_event_id",
    "source_market_id",
    "quote_source",
    "has_executable_quotes",
    "volume_available",
    "open_interest_available",
]


def combine_snapshot_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Combine normalized venue frames without allowing identifier collisions."""

    nonempty = [frame.copy() for frame in frames if frame is not None and not frame.empty]
    if not nonempty:
        return pd.DataFrame()

    for frame in nonempty:
        missing = sorted(set(PROVENANCE_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"Snapshot frame is missing provenance columns: {missing}")

    combined = pd.concat(nonempty, ignore_index=True, sort=False)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
    combined["close_time"] = pd.to_datetime(combined["close_time"], utc=True)
    combined = combined.drop_duplicates(
        subset=["venue", "source_market_id", "timestamp"], keep="last"
    )
    return combined.sort_values(["timestamp", "venue", "market_id"]).reset_index(drop=True)
