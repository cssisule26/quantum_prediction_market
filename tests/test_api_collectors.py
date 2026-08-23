from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quantumcrowd.backtest import one_contract_backtest
from quantumcrowd.data import combine_snapshot_frames
from quantumcrowd.kalshi import KalshiHistoricalClient
from quantumcrowd.polymarket import PolymarketHistoricalClient


class _Response:
    def __init__(self, payload: Any):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self.payload


class _PolymarketSession:
    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, params: dict[str, Any], timeout: float) -> _Response:
        self.calls.append((url, params))
        if url.endswith("/markets/keyset"):
            return _Response(
                {
                    "markets": [
                        {
                            "id": "42",
                            "conditionId": "0xcondition",
                            "question": "Will the test pass?",
                            "category": "science",
                            "startDate": "2024-01-01T00:00:00Z",
                            "closedTime": "2024-01-03T00:00:00Z",
                            "closed": True,
                            "outcomes": '["Yes", "No"]',
                            "clobTokenIds": '["yes-token", "no-token"]',
                            "outcomePrices": '["1", "0"]',
                            "events": [{"id": "7"}],
                        }
                    ]
                }
            )
        if url.endswith("/prices-history"):
            return _Response(
                {
                    "history": [
                        {"t": 1704153600, "p": 0.41},
                        {"t": 1704240000, "p": 0.62},
                    ]
                }
            )
        raise AssertionError(f"Unexpected URL: {url}")


class _KalshiSession:
    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, params: dict[str, Any], timeout: float) -> _Response:
        self.calls.append((url, params))
        if url.endswith("/historical/markets"):
            return _Response(
                {
                    "markets": [
                        {
                            "ticker": "KX-TEST-YES",
                            "event_ticker": "KX-TEST",
                            "market_type": "binary",
                            "open_time": "2024-01-01T00:00:00Z",
                            "settlement_ts": "2024-01-03T00:00:00Z",
                            "result": "yes",
                            "title": "Will the test pass?",
                        }
                    ],
                    "cursor": "",
                }
            )
        if url.endswith("/historical/markets/KX-TEST-YES/candlesticks"):
            return _Response(
                {
                    "candlesticks": [
                        {
                            "end_period_ts": 1704153600,
                            "price": {"close": "0.4100"},
                            "yes_bid": {"close": "0.4000"},
                            "yes_ask": {"close": "0.4200"},
                            "volume": "12.00",
                            "open_interest": "31.00",
                        }
                    ]
                }
            )
        raise AssertionError(f"Unexpected URL: {url}")


def test_polymarket_collector_uses_gamma_token_and_clob_contract() -> None:
    session = _PolymarketSession()
    frame = PolymarketHistoricalClient(session=session, request_pause=0).collect_snapshots(
        max_markets=1,
        fidelity_minutes=60,
    )

    assert len(frame) == 2
    assert frame["event_id"].unique().tolist() == ["polymarket:7"]
    assert frame["market_id"].unique().tolist() == ["polymarket:42"]
    assert frame["source_yes_token_id"].unique().tolist() == ["yes-token"]
    assert frame["resolved"].unique().tolist() == [1]
    assert not frame["has_executable_quotes"].any()
    assert session.calls[0][1]["closed"] == "true"
    assert session.calls[0][1]["order"] == "id"
    assert session.calls[0][1]["ascending"] == "false"
    assert "offset" not in session.calls[0][1]
    assert session.calls[1][1] == {
        "market": "yes-token",
        "startTs": 1704067200,
        "endTs": 1704240000,
        "fidelity": 60,
    }


def test_kalshi_collector_preserves_historical_bid_ask_provenance() -> None:
    session = _KalshiSession()
    frame = KalshiHistoricalClient(session=session, request_pause=0).collect_snapshots(max_markets=1)

    assert len(frame) == 1
    assert frame.loc[0, "event_id"] == "kalshi:KX-TEST"
    assert frame.loc[0, "market_id"] == "kalshi:KX-TEST-YES"
    assert bool(frame.loc[0, "has_executable_quotes"])
    assert frame.loc[0, "quote_source"] == "candlestick_close_bid_ask"
    assert session.calls[1][1]["period_interval"] == 60


def test_kalshi_series_filter_is_not_combined_with_mutually_exclusive_mve_filter() -> None:
    session = _KalshiSession()
    client = KalshiHistoricalClient(session=session, request_pause=0)
    list(client.iter_archived_markets(max_markets=1, series_ticker="KX-TEST-SERIES"))

    assert session.calls[0][1]["series_ticker"] == "KX-TEST-SERIES"
    assert "mve_filter" not in session.calls[0][1]


def test_price_only_history_cannot_create_paper_trades() -> None:
    session = _PolymarketSession()
    frame = PolymarketHistoricalClient(session=session, request_pause=0).collect_snapshots(max_markets=1)
    probability = np.full(len(frame), 0.99)
    trades = one_contract_backtest(frame, probability, min_edge=0.01, fee_per_contract=0.0)

    assert (trades["side"] == "PASS").all()
    assert (trades["pass_reason"] == "non_executable_history").all()


def test_combined_schema_keeps_same_source_id_separate_by_venue() -> None:
    base = {
        "source_event_id": "7",
        "source_market_id": "42",
        "timestamp": pd.Timestamp("2024-01-02", tz="UTC"),
        "close_time": pd.Timestamp("2024-01-03", tz="UTC"),
        "quote_source": "test",
        "has_executable_quotes": True,
        "volume_available": True,
        "open_interest_available": True,
    }
    kalshi = pd.DataFrame([{**base, "venue": "kalshi", "market_id": "kalshi:42"}])
    polymarket = pd.DataFrame(
        [{**base, "venue": "polymarket", "market_id": "polymarket:42"}]
    )
    combined = combine_snapshot_frames([kalshi, polymarket])

    assert len(combined) == 2
    assert set(combined["market_id"]) == {"kalshi:42", "polymarket:42"}
