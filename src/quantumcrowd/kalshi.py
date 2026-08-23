"""Read-only Kalshi historical-data collector."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd


DEFAULT_BASE_URL = "https://external-api.kalshi.com/trade-api/v2"


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _close(value: Any) -> float:
    if isinstance(value, dict):
        for key in ("close", "mean", "previous", "open"):
            if value.get(key) is not None:
                return _float(value[key])
    return _float(value)


@dataclass
class KalshiHistoricalClient:
    base_url: str = DEFAULT_BASE_URL
    timeout: float = 30.0
    request_pause: float = 0.05
    session: Any = None

    def __post_init__(self) -> None:
        if self.session is None:
            import requests

            self.session = requests.Session()
        if hasattr(self.session, "headers"):
            self.session.headers.update({"User-Agent": "quantumcrowd-research/0.2"})

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        assert self.session is not None
        response = self.session.get(
            f"{self.base_url.rstrip('/')}/{path.lstrip('/')}",
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        time.sleep(self.request_pause)
        return response.json()

    def iter_archived_markets(
        self,
        max_markets: int = 100,
        series_ticker: str | None = None,
    ) -> Iterable[dict[str, Any]]:
        cursor: str | None = None
        yielded = 0
        while yielded < max_markets:
            params: dict[str, Any] = {
                "limit": min(1000, max_markets - yielded),
            }
            if cursor:
                params["cursor"] = cursor
            if series_ticker:
                params["series_ticker"] = series_ticker
            else:
                # Kalshi documents the historical-market filters as mutually exclusive.
                params["mve_filter"] = "exclude"
            payload = self._get("historical/markets", params=params)
            markets = payload.get("markets", [])
            if not markets:
                break
            for market in markets:
                if market.get("market_type") == "binary" and market.get("result") in {"yes", "no"}:
                    yield market
                    yielded += 1
                    if yielded >= max_markets:
                        return
            cursor = payload.get("cursor")
            if not cursor:
                break

    def market_candles(
        self,
        ticker: str,
        start_ts: int,
        end_ts: int,
        period_minutes: int = 60,
    ) -> list[dict[str, Any]]:
        if period_minutes not in {1, 60, 1440}:
            raise ValueError("period_minutes must be one of 1, 60, or 1440")
        payload = self._get(
            f"historical/markets/{ticker}/candlesticks",
            params={"start_ts": start_ts, "end_ts": end_ts, "period_interval": period_minutes},
        )
        return list(payload.get("candlesticks", []))

    def collect_snapshots(
        self,
        max_markets: int = 100,
        series_ticker: str | None = None,
        period_minutes: int = 60,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for market in self.iter_archived_markets(max_markets=max_markets, series_ticker=series_ticker):
            ticker = str(market["ticker"])
            open_time = pd.to_datetime(market.get("open_time"), utc=True, errors="coerce")
            close_time = pd.to_datetime(
                market.get("settlement_ts") or market.get("close_time"), utc=True, errors="coerce"
            )
            if pd.isna(open_time) or pd.isna(close_time):
                continue
            candles = self.market_candles(
                ticker=ticker,
                start_ts=int(open_time.timestamp()),
                end_ts=int(close_time.timestamp()),
                period_minutes=period_minutes,
            )
            for candle in candles:
                price = _close(candle.get("price"))
                yes_bid = _close(candle.get("yes_bid"))
                yes_ask = _close(candle.get("yes_ask"))
                has_quotes = yes_bid > 0 and yes_ask > 0 and yes_ask >= yes_bid
                if price <= 0 and yes_bid > 0 and yes_ask > 0:
                    price = (yes_bid + yes_ask) / 2.0
                if not 0 < price < 1:
                    continue
                timestamp = pd.to_datetime(candle.get("end_period_ts"), unit="s", utc=True)
                rows.append(
                    {
                        "venue": "kalshi",
                        "source_event_id": market.get("event_ticker") or ticker,
                        "source_market_id": ticker,
                        "event_id": f"kalshi:{market.get('event_ticker') or ticker}",
                        "market_id": f"kalshi:{ticker}",
                        "category": market.get("series_ticker") or "unknown",
                        "title": market.get("title") or "",
                        "timestamp": timestamp,
                        "close_time": close_time,
                        "market_prob": price,
                        "yes_bid": yes_bid if yes_bid > 0 else price,
                        "yes_ask": yes_ask if yes_ask > 0 else price,
                        "quote_source": "candlestick_close_bid_ask",
                        "has_executable_quotes": has_quotes,
                        "volume": _float(candle.get("volume")),
                        "open_interest": _float(candle.get("open_interest")),
                        "volume_available": candle.get("volume") is not None,
                        "open_interest_available": candle.get("open_interest") is not None,
                        "resolved": int(market["result"] == "yes"),
                        "resolution_source": "kalshi_result",
                    }
                )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values(["timestamp", "market_id"]).reset_index(drop=True)
