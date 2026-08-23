# API integration contract

Checked against the official documentation on 2026-08-23. The collectors are
public, read-only research clients. They do not place orders, access portfolios,
or require API keys.

## Kalshi

Base URL: `https://external-api.kalshi.com/trade-api/v2`

| Purpose | Request | Contract used by QuantumCrowd |
|---|---|---|
| Archived market discovery | `GET /historical/markets` | Cursor pagination; `limit` up to 1000; optional `series_ticker`; otherwise `mve_filter=exclude` |
| Archived snapshots | `GET /historical/markets/{ticker}/candlesticks` | Required `start_ts`, `end_ts`, and `period_interval` in `{1, 60, 1440}` |

Candles expose `end_period_ts`, YES bid and ask OHLC objects, price OHLC/mean,
volume, and open interest. QuantumCrowd uses each object's close field and marks
a row quote-eligible only when both bid and ask are present and ordered.

Kalshi moves older settled data behind a time-varying historical cutoff. This
version intentionally collects the archived partition only. A complete future
collector should read `GET /historical/cutoff`, query the corresponding live
partition, and de-duplicate at the boundary.

Official references:

- [Historical markets](https://docs.kalshi.com/api-reference/historical/get-historical-markets)
- [Historical market candlesticks](https://docs.kalshi.com/api-reference/historical/get-historical-market-candlesticks)
- [Historical data partitioning](https://docs.kalshi.com/getting_started/historical_data)

## Polymarket

Metadata base URL: `https://gamma-api.polymarket.com`

Price-history base URL: `https://clob.polymarket.com`

| Purpose | Request | Contract used by QuantumCrowd |
|---|---|---|
| Closed market discovery | `GET /markets/keyset` | `closed=true`, recent IDs first, `limit` up to 100, optional `tag_id`, and opaque `after_cursor`; never `offset` |
| Observed price history | `GET /prices-history` | YES outcome token in `market`; Unix seconds in `startTs` and `endTs`; sampling minutes in `fidelity` |

Gamma currently returns `outcomes`, `outcomePrices`, and `clobTokenIds` as
JSON-encoded string arrays. QuantumCrowd aligns their indexes, accepts only
binary YES/NO markets, and requires terminal prices to identify the settled
outcome. It requests history with the YES token ID—not the Gamma market ID or
condition ID.

Some pre-CLOB closed records still have Gamma metadata but no queryable CLOB
history. The collector requests recent IDs first and records per-market history
errors instead of aborting the complete collection.

The CLOB history response contains points shaped like `{t, p}`. It is not a
historical order book and supplies neither bid nor ask. QuantumCrowd therefore
sets `has_executable_quotes=false`, uses the rows only for probability scoring,
and forces the paper backtest to pass.

Official references:

- [Market discovery](https://docs.polymarket.com/market-data/discover-markets)
- [Keyset market listing](https://docs.polymarket.com/api-reference/markets/list-markets-keyset-pagination)
- [Price history](https://docs.polymarket.com/market-data/prices-order-books#price-history)

## Normalized identifiers and missingness

Modeling IDs are prefixed (`kalshi:...` and `polymarket:...`) to prevent a
cross-venue collision. Native IDs remain in `source_event_id` and
`source_market_id`. Data-source differences are explicit through `venue`,
`quote_source`, `has_executable_quotes`, `volume_available`, and
`open_interest_available`.

Combining rows does not make the venues statistically exchangeable. Prefer one
venue for model development and the other as an external-validation sample;
only pool them after defining common contract families and venue fixed effects.
