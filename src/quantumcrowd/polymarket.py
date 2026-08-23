"""Read-only Polymarket Gamma and CLOB historical-data collector."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd


DEFAULT_GAMMA_URL = "https://gamma-api.polymarket.com"
DEFAULT_CLOB_URL = "https://clob.polymarket.com"


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _binary_market_parts(market: dict[str, Any]) -> tuple[str, int] | None:
    """Return the YES outcome token and settled YES label for a binary market."""

    outcomes = [str(value).strip().lower() for value in _json_list(market.get("outcomes"))]
    tokens = [str(value) for value in _json_list(market.get("clobTokenIds"))]
    prices = [_float(value, default=float("nan")) for value in _json_list(market.get("outcomePrices"))]
    if len(outcomes) != 2 or len(tokens) != 2 or len(prices) != 2:
        return None
    if set(outcomes) != {"yes", "no"}:
        return None

    yes_index = outcomes.index("yes")
    no_index = outcomes.index("no")
    yes_price = prices[yes_index]
    no_price = prices[no_index]
    threshold = 0.99
    if yes_price >= threshold and no_price <= 1.0 - threshold:
        resolved = 1
    elif no_price >= threshold and yes_price <= 1.0 - threshold:
        resolved = 0
    else:
        return None
    return tokens[yes_index], resolved


def _event_source_id(market: dict[str, Any]) -> str:
    events = market.get("events")
    if isinstance(events, list) and events and isinstance(events[0], dict):
        event = events[0]
        for key in ("id", "slug", "ticker"):
            if event.get(key):
                return str(event[key])
    return str(market.get("conditionId") or market.get("id"))


@dataclass
class PolymarketHistoricalClient:
    gamma_url: str = DEFAULT_GAMMA_URL
    clob_url: str = DEFAULT_CLOB_URL
    timeout: float = 30.0
    request_pause: float = 0.05
    session: Any = None

    def __post_init__(self) -> None:
        if self.session is None:
            import requests

            self.session = requests.Session()
        if hasattr(self.session, "headers"):
            self.session.headers.update({"User-Agent": "quantumcrowd-research/0.2"})

    def _get(
        self,
        base_url: str,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> Any:
        response = self.session.get(
            f"{base_url.rstrip('/')}/{path.lstrip('/')}",
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        time.sleep(self.request_pause)
        return response.json()

    def iter_resolved_binary_markets(
        self,
        max_markets: int = 100,
        tag_id: int | None = None,
        max_scan_pages: int = 50,
    ) -> Iterable[dict[str, Any]]:
        """Page through closed Gamma markets and keep confidently settled YES/NO markets."""

        if max_markets <= 0:
            return
        cursor: str | None = None
        yielded = 0
        for _ in range(max_scan_pages):
            params: dict[str, Any] = {
                "closed": "true",
                "limit": min(100, max(20, max_markets - yielded)),
            }
            if cursor:
                params["after_cursor"] = cursor
            if tag_id is not None:
                params["tag_id"] = tag_id
            payload = self._get(self.gamma_url, "markets/keyset", params=params)
            markets = payload.get("markets", []) if isinstance(payload, dict) else []
            for market in markets:
                if not isinstance(market, dict) or _binary_market_parts(market) is None:
                    continue
                yield market
                yielded += 1
                if yielded >= max_markets:
                    return
            next_cursor = payload.get("next_cursor") if isinstance(payload, dict) else None
            if not next_cursor or next_cursor == cursor:
                break
            cursor = str(next_cursor)

    def price_history(
        self,
        token_id: str,
        start_ts: int,
        end_ts: int,
        fidelity_minutes: int = 60,
    ) -> list[dict[str, Any]]:
        if fidelity_minutes <= 0:
            raise ValueError("fidelity_minutes must be positive")
        payload = self._get(
            self.clob_url,
            "prices-history",
            params={
                "market": token_id,
                "startTs": start_ts,
                "endTs": end_ts,
                "fidelity": fidelity_minutes,
            },
        )
        return list(payload.get("history", [])) if isinstance(payload, dict) else []

    def collect_snapshots(
        self,
        max_markets: int = 100,
        tag_id: int | None = None,
        fidelity_minutes: int = 60,
        max_scan_pages: int = 50,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        markets = self.iter_resolved_binary_markets(
            max_markets=max_markets,
            tag_id=tag_id,
            max_scan_pages=max_scan_pages,
        )
        for market in markets:
            parts = _binary_market_parts(market)
            if parts is None:
                continue
            yes_token_id, resolved = parts
            source_market_id = str(market["id"])
            source_event_id = _event_source_id(market)
            open_time = pd.to_datetime(
                market.get("startDate") or market.get("startDateIso") or market.get("createdAt"),
                utc=True,
                errors="coerce",
            )
            close_time = pd.to_datetime(
                market.get("closedTime")
                or market.get("endDate")
                or market.get("endDateIso")
                or market.get("umaEndDate"),
                utc=True,
                errors="coerce",
            )
            if pd.isna(open_time) or pd.isna(close_time) or close_time <= open_time:
                continue
            history = self.price_history(
                token_id=yes_token_id,
                start_ts=int(open_time.timestamp()),
                end_ts=int(close_time.timestamp()),
                fidelity_minutes=fidelity_minutes,
            )
            for point in history:
                price = _float(point.get("p"), default=float("nan"))
                timestamp = pd.to_datetime(point.get("t"), unit="s", utc=True, errors="coerce")
                if pd.isna(timestamp) or not 0.0 <= price <= 1.0 or timestamp > close_time:
                    continue
                rows.append(
                    {
                        "venue": "polymarket",
                        "source_event_id": source_event_id,
                        "source_market_id": source_market_id,
                        "source_yes_token_id": yes_token_id,
                        "event_id": f"polymarket:{source_event_id}",
                        "market_id": f"polymarket:{source_market_id}",
                        "category": market.get("category") or "unknown",
                        "title": market.get("question") or "",
                        "timestamp": timestamp,
                        "close_time": close_time,
                        "market_prob": price,
                        # The historical endpoint returns observed prices, not book quotes.
                        "yes_bid": price,
                        "yes_ask": price,
                        "quote_source": "clob_observed_price",
                        "has_executable_quotes": False,
                        # Do not repeat terminal market totals across old snapshots: that leaks.
                        "volume": 0.0,
                        "open_interest": 0.0,
                        "volume_available": False,
                        "open_interest_available": False,
                        "resolved": resolved,
                        "resolution_source": "terminal_outcome_prices",
                    }
                )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values(["timestamp", "market_id"]).reset_index(drop=True)
