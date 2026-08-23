"""Command-line research workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .backtest import backtest_summary, one_contract_backtest
from .data import combine_snapshot_frames
from .evaluation import calibration_table, clustered_brier_delta_interval, probability_metrics
from .features import FEATURE_COLUMNS, build_features, select_forecast_horizon
from .models import AnchoredLogitModel, MLPProbabilityModel, MarketBaseline
from .splits import chronological_event_split
from .synthetic import generate_synthetic_markets


def _run_demo(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = generate_synthetic_markets(n_markets=args.n_markets, seed=args.seed)
    featured = build_features(raw)
    horizon = select_forecast_horizon(featured, horizon_hours=args.horizon_hours)
    split = chronological_event_split(horizon)

    models: list[object] = [
        MarketBaseline(),
        AnchoredLogitModel(FEATURE_COLUMNS, l2=args.l2),
        MLPProbabilityModel(FEATURE_COLUMNS, seed=args.seed),
    ]
    if args.include_quantum:
        from .quantum_model import QuantumResidualModel

        models.append(
            QuantumResidualModel(
                FEATURE_COLUMNS,
                n_qubits=args.qubits,
                maxiter=args.quantum_maxiter,
                seed=args.seed,
            )
        )

    predictions = split.test[
        ["event_id", "market_id", "timestamp", "close_time", "resolved", "market_prob", "yes_bid", "yes_ask"]
    ].copy()
    metrics: dict[str, dict[str, object]] = {}
    market_probability = MarketBaseline().predict_proba(split.test)

    for model in models:
        model.fit(split.train)
        probability = model.predict_proba(split.test)
        name = str(model.name)
        predictions[name] = probability
        forecast = probability_metrics(split.test["resolved"].to_numpy(), probability)
        interval = clustered_brier_delta_interval(split.test, probability, market_probability, seed=args.seed)
        trades = one_contract_backtest(
            split.test,
            probability,
            min_edge=args.min_edge,
            fee_per_contract=args.fee,
        )
        metrics[name] = {
            "forecast": forecast,
            "delta_vs_market": interval,
            "paper_backtest": backtest_summary(trades),
        }
        trades.to_csv(output_dir / f"backtest_{name}.csv", index=False)
        calibration_table(split.test["resolved"].to_numpy(), probability).to_csv(
            output_dir / f"calibration_{name}.csv", index=False
        )

    predictions.to_csv(output_dir / "predictions.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, allow_nan=True)
    with (output_dir / "split_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "train_rows": len(split.train),
                "validation_rows": len(split.validation),
                "test_rows": len(split.test),
                "horizon_hours": args.horizon_hours,
                "synthetic_data": True,
            },
            handle,
            indent=2,
        )

    display = pd.DataFrame(
        {
            name: {
                "brier": values["forecast"]["brier"],
                "log_loss": values["forecast"]["log_loss"],
                "ece": values["forecast"]["ece"],
                "net_pnl": values["paper_backtest"]["net_pnl"],
            }
            for name, values in metrics.items()
        }
    ).T
    print(display.to_string(float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved synthetic demonstration outputs to {output_dir.resolve()}")


def _fetch_kalshi(args: argparse.Namespace) -> None:
    from .kalshi import KalshiHistoricalClient

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    client = KalshiHistoricalClient(request_pause=args.request_pause)
    frame = client.collect_snapshots(
        max_markets=args.max_markets,
        series_ticker=args.series_ticker,
        period_minutes=args.period_minutes,
    )
    if frame.empty:
        raise RuntimeError("No eligible archived binary-market candles were returned")
    frame.to_csv(output, index=False)
    print(f"Saved {len(frame):,} snapshots from {frame['market_id'].nunique():,} markets to {output}")


def _fetch_polymarket(args: argparse.Namespace) -> None:
    from .polymarket import PolymarketHistoricalClient

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    client = PolymarketHistoricalClient(request_pause=args.request_pause)
    frame = client.collect_snapshots(
        max_markets=args.max_markets,
        tag_id=args.tag_id,
        fidelity_minutes=args.fidelity_minutes,
        max_scan_pages=args.max_scan_pages,
    )
    if frame.empty:
        raise RuntimeError("No confidently resolved binary Polymarket price histories were returned")
    frame.to_csv(output, index=False)
    print(f"Saved {len(frame):,} snapshots from {frame['market_id'].nunique():,} markets to {output}")


def _fetch_both(args: argparse.Namespace) -> None:
    from .kalshi import KalshiHistoricalClient
    from .polymarket import PolymarketHistoricalClient

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    kalshi = KalshiHistoricalClient(request_pause=args.request_pause).collect_snapshots(
        max_markets=args.kalshi_max_markets,
        series_ticker=args.kalshi_series_ticker,
        period_minutes=args.kalshi_period_minutes,
    )
    polymarket = PolymarketHistoricalClient(request_pause=args.request_pause).collect_snapshots(
        max_markets=args.polymarket_max_markets,
        tag_id=args.polymarket_tag_id,
        fidelity_minutes=args.polymarket_fidelity_minutes,
        max_scan_pages=args.polymarket_max_scan_pages,
    )
    missing = [name for name, frame in (("Kalshi", kalshi), ("Polymarket", polymarket)) if frame.empty]
    if missing:
        raise RuntimeError(f"No eligible snapshots returned for: {', '.join(missing)}")
    frame = combine_snapshot_frames([kalshi, polymarket])
    frame.to_csv(output, index=False)
    counts = frame.groupby("venue")["market_id"].nunique().to_dict()
    print(f"Saved {len(frame):,} snapshots to {output}; markets by venue: {counts}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quantumcrowd")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Run the reproducible synthetic experiment")
    demo.add_argument("--output-dir", default="outputs/demo")
    demo.add_argument("--n-markets", type=int, default=300)
    demo.add_argument("--horizon-hours", type=float, default=24.0)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--l2", type=float, default=0.1)
    demo.add_argument("--min-edge", type=float, default=0.03)
    demo.add_argument("--fee", type=float, default=0.01)
    demo.add_argument("--include-quantum", action="store_true")
    demo.add_argument("--qubits", type=int, default=4)
    demo.add_argument("--quantum-maxiter", type=int, default=60)
    demo.set_defaults(func=_run_demo)

    fetch = subparsers.add_parser("fetch-kalshi", help="Collect public archived Kalshi data")
    fetch.add_argument("--output", required=True)
    fetch.add_argument("--series-ticker")
    fetch.add_argument("--max-markets", type=int, default=100)
    fetch.add_argument("--period-minutes", type=int, choices=[1, 60, 1440], default=60)
    fetch.add_argument("--request-pause", type=float, default=0.05)
    fetch.set_defaults(func=_fetch_kalshi)

    polymarket = subparsers.add_parser(
        "fetch-polymarket", help="Collect public resolved Polymarket price history"
    )
    polymarket.add_argument("--output", required=True)
    polymarket.add_argument("--tag-id", type=int)
    polymarket.add_argument("--max-markets", type=int, default=100)
    polymarket.add_argument("--fidelity-minutes", type=int, default=60)
    polymarket.add_argument("--max-scan-pages", type=int, default=50)
    polymarket.add_argument("--request-pause", type=float, default=0.05)
    polymarket.set_defaults(func=_fetch_polymarket)

    both = subparsers.add_parser("fetch-both", help="Collect and normalize both public venues")
    both.add_argument("--output", required=True)
    both.add_argument("--kalshi-series-ticker")
    both.add_argument("--kalshi-max-markets", type=int, default=100)
    both.add_argument("--kalshi-period-minutes", type=int, choices=[1, 60, 1440], default=60)
    both.add_argument("--polymarket-tag-id", type=int)
    both.add_argument("--polymarket-max-markets", type=int, default=100)
    both.add_argument("--polymarket-fidelity-minutes", type=int, default=60)
    both.add_argument("--polymarket-max-scan-pages", type=int, default=50)
    both.add_argument("--request-pause", type=float, default=0.05)
    both.set_defaults(func=_fetch_both)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)
