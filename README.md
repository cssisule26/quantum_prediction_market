# QuantumCrowd

QuantumCrowd is a research scaffold for testing whether a small hybrid
quantum-classical model can improve prediction-market probabilities **after**
conditioning on the market's own price.

The central model is deliberately anchored to the crowd forecast:

```text
logit(p_hat) = logit(p_market) + classical_residual(x) + alpha * quantum_feature(x)
```

This makes the research question harder and more meaningful than predicting an
outcome from scratch. A useful model must add information beyond the market
price, spread, fees, and other strong classical baselines.

## What makes this project different

- It does not reproduce the EWL Prisoner's Dilemma from the Rutgers
  `Quantum_Trading` repository. There is no hypothetical quantum referee.
- It borrows the *hybrid* principle from QuantumLeap, but it does not reproduce
  QuantumLeap's density-matrix encoder or maximum-stock-price target.
- It treats a binary contract price as a market-aggregated prior and learns a
  residual probability correction.
- It adds Kyle-inspired, market-level information-flow proxies: signed flow,
  price impact per unit volume, recent reversal, volatility, and liquidity.
- It evaluates probability quality first and trading P&L second. No simulated
  payoff matrix is labeled "alpha."

The complete scientific rationale, hypotheses, leakage controls, and literature
map are in [`docs/RESEARCH_DESIGN.md`](docs/RESEARCH_DESIGN.md).
The endpoint and schema checks are recorded in
[`docs/API_INTEGRATION.md`](docs/API_INTEGRATION.md).

## Install

Classical baseline and synthetic demonstration:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
quantumcrowd demo --output-dir outputs/demo
```

Quantum model:

```bash
pip install -e ".[quantum,dev]"
quantumcrowd demo --include-quantum --quantum-maxiter 60
```

The quantum demonstration uses an exact state-vector estimator. IBM hardware
should be used only on a frozen, small test subset after simulator experiments
are complete; otherwise queueing, shot noise, and repeated tuning on the test
set will invalidate the comparison.

## Fetch public Kalshi and Polymarket research data

Both collectors are read-only and request no account keys. Kalshi supplies
archived binary-market candlesticks with bid/ask OHLC fields:

```bash
quantumcrowd fetch-kalshi \
  --output data/raw/kalshi_hourly.csv \
  --max-markets 100 \
  --period-minutes 60
```

To limit the sample to a recurring family, add `--series-ticker SERIES`. The
collector intentionally performs no trading and requests no account keys.

Polymarket uses Gamma to discover confidently resolved YES/NO markets, extracts
the YES outcome token, and then requests that token's observed CLOB price
history:

```bash
quantumcrowd fetch-polymarket \
  --output data/raw/polymarket_hourly.csv \
  --max-markets 100 \
  --fidelity-minutes 60
```

To produce one provenance-preserving CSV:

```bash
quantumcrowd fetch-both \
  --output data/raw/both_hourly.csv \
  --kalshi-max-markets 100 \
  --polymarket-max-markets 100
```

Do not interpret this combined file as a license to pool venues blindly. Start
with a venue-specific model and use the other venue for external validation.
Contract wording, participant populations, fees, resolution processes, and
historical data granularity differ.

### Important API asymmetry

Kalshi historical candlesticks expose bid/ask closes, so rows with valid quotes
can enter the conservative paper backtest. Polymarket's public
`/prices-history` response contains observed outcome-token prices, but not a
historical order book. Those rows therefore set `has_executable_quotes=false`.
They remain valid for forecast scoring and calibration, while the backtest
automatically passes on them. The code never treats the observed price as a
historically executable fill.

## Expected input schema

One row is one market snapshot:

| Column | Meaning |
|---|---|
| `venue` | `kalshi`, `polymarket`, or `synthetic` |
| `source_event_id`, `source_market_id` | Venue-native identifiers retained for audits |
| `event_id` | Group identifier used to prevent related-contract leakage |
| `market_id` | Unique binary contract identifier |
| `timestamp` | Snapshot time in UTC |
| `close_time` | Resolution/close proxy in UTC |
| `market_prob` | Last or mean YES price in `[0, 1]` |
| `yes_bid`, `yes_ask` | Executable YES quotes when available |
| `quote_source` | Feed field used to construct the price/quote |
| `has_executable_quotes` | Whether the row is eligible for the paper backtest |
| `volume` | Interval volume |
| `open_interest` | Open interest |
| `volume_available`, `open_interest_available` | Missingness/provenance indicators |
| `resolved` | Settled binary outcome, `0` or `1` |

Use `build_features()` before fitting. For a fixed-horizon experiment, use
`select_forecast_horizon()` so each market contributes one forecast.

## Models and evaluation

The demo evaluates:

1. Raw market probability.
2. Anchored logistic residual model.
3. Small classical MLP.
4. Optional Qiskit `EstimatorQNN` residual model.

Primary metrics are Brier score, log loss, expected calibration error, and AUC.
The included paper-trading layer buys at the ask only when estimated edge clears
the configured fee and threshold. It is a research backtest, not a live trader.

## Repository layout

```text
src/quantumcrowd/
  synthetic.py       reproducible market simulator
  features.py        microstructure feature construction
  splits.py          chronological event-grouped split
  models.py          market, anchored-logit, and MLP baselines
  quantum_model.py   Qiskit EstimatorQNN residual model
  evaluation.py      forecast metrics and clustered bootstrap
  backtest.py        one-contract, executable-price paper backtest
  data.py            normalized multi-venue schema and combination
  kalshi.py          public historical-data collector
  polymarket.py      Gamma discovery + CLOB price-history collector
  cli.py             demo and data-collection commands
```
