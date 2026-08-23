from __future__ import annotations

import numpy as np

from quantumcrowd.backtest import backtest_summary, one_contract_backtest
from quantumcrowd.features import FEATURE_COLUMNS, build_features, select_forecast_horizon
from quantumcrowd.models import AnchoredLogitModel, MarketBaseline
from quantumcrowd.splits import chronological_event_split
from quantumcrowd.synthetic import generate_synthetic_markets


def _dataset():
    raw = generate_synthetic_markets(n_markets=80, seed=11)
    return select_forecast_horizon(build_features(raw), horizon_hours=24)


def test_features_and_split_do_not_leak_events():
    data = _dataset()
    assert set(FEATURE_COLUMNS).issubset(data.columns)
    split = chronological_event_split(data)
    train_events = set(split.train["event_id"])
    validation_events = set(split.validation["event_id"])
    test_events = set(split.test["event_id"])
    assert train_events.isdisjoint(validation_events)
    assert train_events.isdisjoint(test_events)
    assert validation_events.isdisjoint(test_events)


def test_anchored_model_returns_valid_probabilities():
    split = chronological_event_split(_dataset())
    model = AnchoredLogitModel(FEATURE_COLUMNS, l2=0.2).fit(split.train)
    probability = model.predict_proba(split.test)
    assert probability.shape == (len(split.test),)
    assert np.all((probability > 0) & (probability < 1))


def test_backtest_passes_without_sufficient_edge():
    data = _dataset().iloc[:10].copy()
    probability = MarketBaseline().predict_proba(data)
    trades = one_contract_backtest(data, probability, min_edge=0.5, fee_per_contract=0.01)
    summary = backtest_summary(trades)
    assert summary["trades"] == 0
    assert summary["net_pnl"] == 0.0

