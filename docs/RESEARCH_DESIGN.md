# Research design: hybrid quantum calibration of prediction markets

## Proposed title

**QuantumCrowd: Hybrid Quantum-Classical Residual Calibration of Prediction
Market Probabilities**

## Research question

At fixed forecast horizons and under event-grouped chronological evaluation,
does a shallow variational quantum feature improve the calibration or net
paper-trading performance of prediction-market probabilities relative to the
raw market price and matched classical models?

This is a comparative empirical question. A null or negative result is valid.

## Intellectual lineage without replication

### QuantumLeap

Paquet and Soleymani's QuantumLeap system encodes partitioned financial time
series into density matrices, passes them through a deep quantum network, and
uses a classical measurement network to predict a later maximum security price.
QuantumCrowd retains only the broad hybrid principle. It instead uses shallow
variational circuits, predicts a binary-event probability, and anchors the
model to a market-generated prior.

Source: [QuantumLeap: Hybrid quantum neural network for financial predictions](https://doi.org/10.1016/j.eswa.2022.116583)

### Prediction-market theory

Wolfers and Zitzewitz describe binary contracts paying one dollar when an event
occurs and explain why their prices can be interpreted as market-aggregated
probability forecasts. They also document rapid information incorporation,
generally strong predictive performance, and potential pathologies such as the
favorite-longshot bias.

Sources:

- [Prediction Markets in Theory and Practice](https://www.nber.org/papers/w12083)
- [Interpreting Prediction Market Prices as Probabilities](https://www.nber.org/papers/w12200)

### Price formation and informed flow

Bossaerts and coauthors apply a Kyle-derived method to field prediction markets.
They find that trades from price-sensitive participants add information while
other trades look more like noise. Public exchange data generally do not expose
stable trader identities, so QuantumCrowd does **not** reproduce their trader
classification. It uses weaker market-level proxies and labels them accordingly:
signed price-volume flow, price impact per unit volume, reversals, volatility,
and liquidity.

Source: [Price Formation in Field Prediction Markets: The Wisdom in the Crowd](https://arxiv.org/abs/2209.08778)

### Additional methodological anchors

- Favorite-longshot bias motivates explicit tail features and calibration by
  probability band: [Snowberg and Wolfers](https://www.nber.org/papers/w15923).
- Quantum feature maps motivate the variational feature block:
  [Havlíček et al., Nature 2019](https://doi.org/10.1038/s41586-019-0980-2).
- Barren plateaus motivate shallow circuits, few qubits, several seeds, and a
  strict optimizer budget:
  [McClean et al., Nature Communications 2018](https://doi.org/10.1038/s41467-018-07090-4).
- Qiskit's `EstimatorQNN` supplies a current, hardware-compatible expectation
  value interface:
  [Qiskit Machine Learning documentation](https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.neural_networks.EstimatorQNN.html).

## Model

For market snapshot `i`, let `m_i` be the clipped market midpoint or last price
and `x_i` be the microstructure vector. The forecast is

```text
z_i     = quantum_feature(PCA(scale(x_i)); theta)
delta_i = beta^T scale(x_i) + alpha z_i + b
p_i     = sigmoid(logit(m_i) + delta_i)
```

The quantum block uses approximately four qubits, one ZZ feature-map layer, one
shallow real-amplitudes ansatz, and a Z-tensor observable. The classical term is
kept in the hybrid model so the test is whether `alpha * z_i` adds information
beyond an ordinary residual correction.

Train with weighted binary cross-entropy. If multiple snapshots per market are
used during training, give each market equal total weight.

## Hypotheses

- **H1:** Anchored classical residual calibration improves Brier score relative
  to raw market price in at least one pre-specified market family.
- **H2:** Kyle-inspired market-level features add incremental information over
  price, spread, and time to close alone.
- **H3:** The quantum residual improves locked-test Brier score relative to both
  anchored linear and parameter-matched nonlinear classical residual models.
- **H4:** Hardware noise reduces or eliminates any simulator improvement.

H3 is the quantum claim. H1 or H2 succeeding does not imply quantum advantage.

## Dataset plan

Start with one recurring, structured contract family. Avoid pooling unrelated
politics, weather, sports, and macro markets until category-specific results are
established.

For every settled market, retain:

- Contract and event identifiers.
- Exact resolution text and final outcome.
- Timestamped last price or midpoint.
- Executable bid and ask when available.
- Volume, open interest, and trade direction where available.
- Close/settlement time.

Candidate sources:

- [Kalshi historical markets](https://docs.kalshi.com/api-reference/historical/get-historical-markets)
- [Kalshi historical candlesticks](https://docs.kalshi.com/api-reference/historical/get-historical-market-candlesticks)
- [Kalshi historical trades](https://docs.kalshi.com/api-reference/historical/get-historical-trades)
- [Polymarket historical prices](https://docs.polymarket.com/api-reference/markets/get-prices-history)

The collectors preserve venue-native IDs and prefix modeling IDs with the venue
name. Polymarket Gamma market records map human-readable YES/NO outcomes to CLOB
token IDs; the historical request must use the YES token ID, not the Gamma
market ID or condition ID. Its public price history contains observations but
no historical bid/ask book, so it is excluded from executable-price P&L unless
a separate, timestamped order-book archive is supplied. Kalshi historical
candlesticks do include bid/ask OHLC fields, but their candle-close quotes are
still a backtest proxy rather than proof of a fill.

Kalshi partitions old settled markets at a moving historical cutoff. A complete
longitudinal collector should query the cutoff, route old data to the historical
endpoints, route newer settled data to live endpoints, and de-duplicate the two.
Version 0.2 intentionally starts with the archived endpoint to keep the sample
definition auditable.

Use one venue for training and, if feasible, the other only as external
validation after harmonizing contract definitions.

## Leakage controls

1. Split chronologically by **event**, not by row.
2. Keep every contract and snapshot from a related event in one split.
3. Fit scalers, PCA, calibrators, and feature selection on training data only.
4. Freeze one forecast horizon, such as 24 hours before close, before testing.
5. Exclude metadata created after the snapshot and markets with ambiguous or
   administratively canceled resolutions.
6. Do not tune circuits or trading thresholds on the locked test set.

Suggested split: earliest 60% of events for training, next 20% for validation,
latest 20% for testing. A rolling-origin analysis should be the robustness check.

## Baselines

All are required:

1. Raw market midpoint/last price.
2. Tail-only calibration curve.
3. Anchored linear logit residual.
4. Gradient boosting or a small MLP.
5. Parameter-matched classical bottleneck.
6. Quantum circuit with entangling gates removed.
7. Full quantum residual on an exact simulator.
8. Shot-based/noisy simulator.
9. Frozen IBM hardware subset.

## Metrics

Primary:

- Brier score.
- Log loss.
- Reliability curve and expected calibration error.
- Cluster-bootstrap confidence interval for `Brier(model) - Brier(market)`.

Secondary:

- ROC AUC.
- Net one-contract P&L at executable quotes.
- Coverage: fraction of markets with tradeable estimated edge.
- Maximum drawdown and profit per selected contract.

Never optimize the headline model on test P&L. The forecast metric is primary.

## Paper-backtest rule

For a YES ask `a_yes`, NO ask `a_no`, probability `p`, and per-contract cost
`c`:

```text
edge_yes = p - a_yes - c
edge_no  = (1 - p) - a_no - c
```

Select at most one side per market and only if the larger edge exceeds a
threshold fixed on validation data. This avoids treating midpoint prices as
executable trades.

## Minimum publishable experiment

1. One structured market family.
2. One pre-specified forecast horizon.
3. At least three classical baselines.
4. Four-qubit shallow QNN on exact and shot-based simulators.
5. Five or more independent initialization seeds.
6. Event-clustered confidence intervals.
7. A small frozen hardware inference run.
8. Full reporting of failures, runtime, circuit depth, and parameter counts.

## Claims to avoid

- “Quantum alpha” based only on a simulator.
- “Nash equilibrium” when maximizing joint payoff.
- “Quantum speedup” without wall-clock and complexity evidence.
- “Profitable strategy” when using midpoint rather than executable quotes.
- “Wisdom of the crowd” as proof that every participant contributes equally.
