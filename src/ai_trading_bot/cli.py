"""Command-line entrypoints for the trading bot (console script: ``ai-trading-bot``).

Examples
--------
  ai-trading-bot fetch BTC/USDT --timeframe 1h --days 2
  ai-trading-bot run --seconds 60
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

    sub.add_parser("show-config", help="print the resolved config (no secrets)")
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())