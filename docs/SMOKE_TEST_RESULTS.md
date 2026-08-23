# Real-data smoke-test results

Run on 2026-08-23. These checks establish that the public API responses can be
parsed and passed through the research pipeline. They are **not** evidence of
predictive or quantum advantage.

## Polymarket

- 26 historical price observations.
- 12 markets from 12 distinct events.
- One-hour forecast horizon.
- Chronological event split: 7 train, 2 validation, and 3 test events.
- No collection errors.
- No paper trades, because `/prices-history` does not contain historical
  executable bid/ask quotes.

| Model | Test Brier | Test log loss | ECE | AUC | Trades |
|---|---:|---:|---:|---:|---:|
| Raw market | 0.2500 | 0.6931 | 0.1667 | 0.5000 | 0 |
| Anchored logit | 0.3333 | 3.8376 | 0.3333 | 0.5000 | 0 |
| Classical MLP | 0.6666 | 6.3894 | 0.6666 | 1.0000 | 0 |

The raw market price won this smoke test. The learned corrections were unstable
because the training split had only seven rows. The sample also had 11 YES
outcomes and one NO outcome, while the locked test contained only three rows.
The MLP's apparent AUC of 1.0 is therefore meaningless; its Brier score and log
loss show that its probabilities were poor.

Files:

- `data/raw/polymarket_smoke_diverse.csv`
- `outputs/polymarket_smoke/summary.json`
- `outputs/polymarket_smoke/metrics.json`
- `outputs/polymarket_smoke/predictions.csv`

## Kalshi

The documented `external-api.kalshi.com` host returned 403 from this execution
environment. Kalshi's public `api.elections.kalshi.com` host returned the same
historical schema and was used for a single-market round trip.

- Market: New York City temperature above 68.99°F on 2026-06-22.
- Settled outcome: NO.
- Two hourly normalized candles.
- Recorded market probability: 0.02 at both observations.
- The rows were correctly marked non-executable because the raw candle lacked a
  complete positive bid/ask pair at the relevant close.

File: `data/raw/kalshi_smoke_single_market.csv`

One market is enough to verify parsing, but not to fit or score a model.

## What the next experiment requires

1. Choose one contract family instead of mixing sports, crypto, weather, and
   politics.
2. Collect at least 200 distinct events, preferably more.
3. Pre-specify a horizon such as 24 hours and inspect class balance before
   fitting.
4. Compare the raw market, anchored logit, and classical nonlinear model first.
5. Run the Qiskit simulator only after the classical pipeline is stable.
6. Use IBM hardware only for a frozen inference subset; do not tune on hardware
   or on the locked test set.

A frontend should read cached CSV and JSON outputs from this workflow. It should
not query exchange APIs or train models during a page request.
