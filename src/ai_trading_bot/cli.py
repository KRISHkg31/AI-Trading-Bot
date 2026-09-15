"""Command-line entrypoints for the trading bot (console script: ``ai-trading-bot``).

Examples
--------
  ai-trading-bot fetch BTC/USDT --timeframe 1h --days 2
  ai-trading-bot run --seconds 60
  ai-trading-bot backtest BTC/USDT --days 60 --walk-forward --stress
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging

from ai_trading_bot.adapters.data import build_adapters
from ai_trading_bot.config import get_credentials, load_config, validate_config
from ai_trading_bot.domain import Timeframe, instrument
from ai_trading_bot.engine import TradingEngine
from ai_trading_bot.monitoring import setup_logging


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ai-trading-bot", description="AI multi-asset trading bot")
    p.add_argument("--profile", default=None, choices=["dev", "backtest", "paper", "live"])
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fetch", help="fetch historical bars and print a summary")
    f.add_argument("symbol", type=str)
    f.add_argument("--timeframe", type=str, default="1h")
    f.add_argument("--days", type=int, default=2)
    f.add_argument("--persist", action="store_true", help="write bars to the Parquet store")

    r = sub.add_parser("run", help="run the core engine loop")
    r.add_argument("--seconds", type=int, default=60, help="run duration in seconds")
    r.add_argument("--poll", type=float, default=30.0, help="poll interval in seconds")
    r.add_argument("--symbol", type=str, default="BTC/USDT")
    r.add_argument("--timeframe", type=str, default="5m")

    b = sub.add_parser("backtest", help="run the cost-aware backtester on a symbol")
    b.add_argument("symbol", type=str)
    b.add_argument("--timeframe", type=str, default="5m")
    b.add_argument("--days", type=int, default=30)
    b.add_argument("--walk-forward", action="store_true", help="REQ-BT-03 out-of-sample run")
    b.add_argument("--stress", action="store_true", help="REQ-BT-05 run adverse-regime scenarios")

    m = sub.add_parser("models", help="inspect ML model registry and manage active versions (REQ-STR-04)")
    m.add_argument("--activate", metavar="MODEL_ID@VERSION", default=None,
                   help="pin the active version, e.g. ml_momentum@2 (the rollback lever)")

    r = sub.add_parser("retrain", help="retrain ML models from stored bars; promote only if improved")
    r.add_argument("--symbol", type=str, default="BTC/USDT")
    r.add_argument("--timeframe", type=str, default="5m")
    r.add_argument("--horizon-bars", type=int, default=None,
                   help="override models.label_horizon_bars (default: config)")

    sub.add_parser("show-config", help="print the resolved config (no secrets)")

    sub.add_parser("monitor", help="out-of-process dead-man's switch: exit 0 if the engine is fresh, non-zero if it stopped beating (REQ-RSK-34)")

    v = sub.add_parser("validate", help="REQ-BT-03/07 go-live gate: walk-forward + stress on STORED bars, exit non-zero on acceptance failure")
    v.add_argument("symbol", type=str)
    v.add_argument("--timeframe", type=str, default="5m")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_config(args.profile)
    setup_logging(config)

    problems = validate_config(config)
    for problem in problems:
        logging.getLogger("ai_trading_bot.cli").error("config problem: %s", problem)
    if problems:
        logging.getLogger("ai_trading_bot.cli").error("refusing to run with config problems")
        return 2

    if args.command == "show-config":
        print(config.to_dict())
        return 0

    adapters = build_adapters(get_credentials())

    if args.command == "fetch":
        inst = instrument(args.symbol)
        tf = Timeframe(args.timeframe)
        adapter = adapters[inst.asset_class]
        end = dt.datetime.now(dt.UTC)
        start = end - dt.timedelta(days=args.days)
        bars = adapter.fetch_bars(inst, tf, start, end)
        if args.persist:
            from ai_trading_bot.persistence import BarStore

            store = BarStore(config.data.store_dir)
            store.append_bars(bars)
            print(f"persisted {len(bars)} bars to {config.data.store_dir}")
        last = bars[-1]
        print(f"{len(bars)} {tf.value} bars for {inst.id}")
        print(f"last close={last.close:.2f} high={last.high:.2f} low={last.low:.2f} vol={last.volume:.2f}")
        print(f"feed: {adapter.health()}")
        return 0

    if args.command == "run":
        engine = TradingEngine(config, market_data=adapters)
        engine.run_forever(
            [instrument(args.symbol)],
            timeframe=Timeframe(args.timeframe),
            poll_interval_s=args.poll,
            max_iterations=max(1, round(args.seconds / args.poll)),
        )
        return 0

    if args.command == "models":
        from ai_trading_bot.models import MODEL_IDS, ModelRepo

        repo = ModelRepo(config.models.model_dir)
        if args.activate:
            try:
                model_id, version = args.activate.rsplit("@", 1)
                version = int(version)
            except ValueError:
                logging.getLogger("ai_trading_bot.cli").error(
                    "--activate expects MODEL_ID@VERSION, got %r", args.activate
                )
                return 2
            if model_id not in MODEL_IDS:
                logging.getLogger("ai_trading_bot.cli").error(
                    "unknown model %r (known: %s)", model_id, ", ".join(MODEL_IDS)
                )
                return 2
            try:
                repo.activate(model_id, version)
                print(f"active {model_id} pinned -> v{version}")
            except ValueError as exc:
                logging.getLogger("ai_trading_bot.cli").error("%s", exc)
                return 2
        for row in repo.list():
            mark = "*" if row["active"] else " "
            acc = f"{row['accuracy']:.3f}" if row["accuracy"] is not None else "-"
            ll = f"{row['log_loss']:.3f}" if row["log_loss"] is not None else "-"
            print(
                f"{mark} {row['model_id']}@v{row['version']} kind={row['kind']:14} "
                f"feat={row['feature_version']} acc={acc:>6} ll={ll:>7} "
                f"n={row['n_samples']} skl={row['sklearn_version']}"
            )
        if not repo.list():
            print("(no trained models yet - run `ai-trading-bot retrain`)")
        return 0

    if args.command == "retrain":
        from ai_trading_bot.models import ModelRepo
        from ai_trading_bot.persistence import BarStore
        from ai_trading_bot.training import retrain_all

        repo = ModelRepo(config.models.model_dir)
        horizon = args.horizon_bars or config.models.label_horizon_bars
        inst = instrument(args.symbol)
        bars = BarStore(config.data.store_dir).load_bars(inst.id, args.timeframe)
        if not bars:
            logging.getLogger("ai_trading_bot.cli").error(
                "no stored %s/%s bars in %s (run `fetch --persist` first)",
                inst.id, args.timeframe, config.data.store_dir,
            )
            return 1
        print(
            f"retraining on {len(bars)} {args.timeframe} bars of {inst.id} "
            f"(horizon={horizon} bars)"
        )
        for outcome in retrain_all(repo, {inst.id: bars}, label_horizon_bars=horizon):
            if outcome.get("skipped"):
                print(f"  {outcome['model_id']}: SKIPPED {outcome['skipped']}")
                continue
            cand = outcome["candidate_metrics"]
            ll = f"{cand.get('log_loss'):.3f}" if cand.get("log_loss") else "-"
            print(
                f"  {outcome['model_id']} v{outcome['version']}: promoted={outcome['promoted']} "
                f"val_acc={cand['accuracy']:.3f} val_ll={ll} "
                f"(incumbent v{outcome['incumbent_version']})"
            )
        return 0

    if args.command == "monitor":
        from ai_trading_bot.monitoring import latest_metrics, metrics_age_seconds
        from ai_trading_bot.persistence import EventLog

        # Same default EventLog the engine writes to; a relocated storedir must
        # be re-pointed here (the engine currently always uses the default).
        log = EventLog()
        row = latest_metrics(log)
        if row is None:
            print("ERROR: no metric rows in the EventLog yet - engine has never run")
            return 3
        age = metrics_age_seconds(log)
        stale = age is not None and age > config.monitoring.heartbeat_stale_seconds
        marker = "STALE" if stale else "OK"
        print(f"HEARTBEAT   {marker}  age={age:.1f}s (threshold {config.monitoring.heartbeat_stale_seconds}s)")
        print(f"RISK_STATE  {row.get('risk_state', '?')}")
        print(f"EQUITY      {row.get('equity'):,.2f}  peak {row.get('peak_equity'):,.2f}  "
              f"dd {row.get('drawdown_from_peak_pct')}%  day {row.get('day_loss_pct')}%")
        print(f"POSITIONS   n={row.get('n_positions')}  exposure={row.get('exposure_usd'):,.2f}")
        versions = row.get("model_versions") or {}
        if versions:
            print("MODELS      " + "  ".join(f"{k}={v or '-'}" for k, v in versions.items()))
        print(f"ENVIRONMENT {row.get('environment', '?')}  iteration={row.get('iteration', '?')}")
        return 1 if stale else 0

    if args.command == "validate":
        from ai_trading_bot.backtest import BacktestEngine, scenarios
        from ai_trading_bot.persistence import BarStore

        inst = instrument(args.symbol)
        bars = BarStore(config.data.store_dir).load_bars(inst.id, args.timeframe)
        if not bars:
            logging.getLogger("ai_trading_bot.cli").error(
                "no stored %s/%s bars in %s (run `fetch --persist` first)",
                inst.id, args.timeframe, config.data.store_dir,
            )
            return 1
        engine = BacktestEngine(config)
        print(f"validating {len(bars)} stored {args.timeframe} bars of {inst.id}")
        base = engine.run(inst, bars)
        print(_format_report("backtest", base))
        wf = engine.walk_forward(inst, bars)
        print(_format_report("walk_forward", wf))
        _print_delta(base, wf, "walk-forward")
        stress = {}
        for name in ("flash_crash", "bear_market", "volatility_spike", "gap_down"):
            transform = getattr(scenarios, name)
            result = engine.run_scenario(inst, bars, name, transform)
            stress[name] = result
            print(_format_report(f"scenario:{name}", result))
        return _gate_validate(config, base, wf, stress)

    if args.command == "backtest":
        from ai_trading_bot.backtest import BacktestEngine, scenarios

        inst = instrument(args.symbol)
        tf = Timeframe(args.timeframe)
        adapter = adapters[inst.asset_class]
        end = dt.datetime.now(dt.UTC)
        start = end - dt.timedelta(days=args.days)
        bars = adapter.fetch_bars(inst, tf, start, end)
        if not bars:
            logging.getLogger("ai_trading_bot.cli").error("no data for %s", inst.id)
            return 1
        engine = BacktestEngine(config)
        base = engine.run(inst, bars)
        print(_format_report("backtest", base))
        if args.walk_forward:
            wf = engine.walk_forward(inst, bars)
            print(_format_report("walk_forward", wf))
            _print_delta(base, wf, "walk-forward")
        if args.stress:
            for name in ("flash_crash", "bear_market", "volatility_spike", "gap_down"):
                transform = getattr(scenarios, name)
                stress = engine.run_scenario(inst, bars, name, transform)
                print(_format_report(f"scenario:{name}", stress))
                _print_delta(base, stress, name)
        return 0

    return 0


def _format_report(label: str, result) -> str:
    m = result.metrics.summary()
    return (
        f"\n=== {label} - {result.instrument_id} ===\n"
        f"  return          {m['total_return_pct']:+.2f}%   ann {m['ann_return_pct']:+.2f}%\n"
        f"  max drawdown    {m['max_drawdown_pct']:.2f}%   sharpe {m['sharpe']}  sortino {m['sortino']}\n"
        f"  calmar          {m['calmar']}   profit factor {m['profit_factor']}\n"
        f"  win rate        {m['win_rate']:.1%}  trades {m['n_trades']}  exposure {m['exposure_pct']:.1f}%\n"
        f"  fees            ${m['fees_usd']:.2f}  final equity ${m['final_equity']:,.2f}"
    )


def _gate_validate(config, base, wf, stress: dict) -> int:
    """Phase 6 acceptance gate (REQ-BT-03/07): out-of-sample results decide.

    PASS/FAIL per criterion against the config'd thresholds; also reports each
    stress scenario's win/loss so adverse regimes are visible, not just summed.
    Returns 0 when the walk-forward gate passes, 1 otherwise.
    """
    g = config.backtest
    m = wf.metrics.summary()
    checks = [
        ("walk-forward return >= 0%", m["total_return_pct"], g.accept_min_return_pct,
         f"return {m['total_return_pct']:+.2f}%", m["total_return_pct"] >= g.accept_min_return_pct),
        ("walk-forward max-dd <= 20%", g.accept_max_drawdown_pct, m["max_drawdown_pct"],
         f"dd {m['max_drawdown_pct']:.2f}%", m["max_drawdown_pct"] <= g.accept_max_drawdown_pct),
        ("walk-forward profit factor >= 1", m["profit_factor"], g.accept_min_profit_factor,
         f"pf {m['profit_factor']:.2f}", m["profit_factor"] >= g.accept_min_profit_factor),
    ]
    all_pass = True
    print("\n=== ACCEPTANCE GATE (walk-forward / out-of-sample) ===")
    for label, lo, hi, desc, ok in checks:
        mark = "PASS" if ok else "FAIL"
        all_pass = all_pass and ok
        print(f"  [{mark}] {label:32} {desc}")
    print("\n=== STRESS REGIMES (loss visibility) ===")
    for name, result in stress.items():
        mm = result.metrics.summary()
        print(f"  {name:16} return {mm['total_return_pct']:+6.2f}%  max-dd {mm['max_drawdown_pct']:6.2f}%  "
              f"trades {mm['n_trades']:3}  pf {mm['profit_factor']:.2f}")
    verdict = "GO-LIVE GATE PASSED (paper-parallel per GO_LIVE.md)" if all_pass \
        else "GO-LIVE GATE FAILED (do not promote to paper/live; review strategy/config)"
    print(f"\n{verdict}")
    return 0 if all_pass else 1


def _print_delta(base, other, label: str) -> None:
    b, o = base.metrics.summary(), other.metrics.summary()
    print(
        f"  vs backtest: return {o['total_return_pct'] - b['total_return_pct']:+.2f}pp | "
        f"max_dd {o['max_drawdown_pct'] - b['max_drawdown_pct']:+.2f}pp | "
        f"sharpe {o['sharpe'] - b['sharpe']:+.2f}"
    )


if __name__ == "__main__":
    raise SystemExit(main())