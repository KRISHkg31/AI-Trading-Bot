"""The 24/7 core engine loop (BRD Section 11, REQ-MON-06).

Wires together: data -> strategy -> signal filter -> risk gate -> execution,
with heartbeat + data-quality staleness handling. Active strategies come from
config (REQ-STR-10); the best filtered signal per instrument is passed to the
risk gate, and execution is paper-logged until Phase 3.
"""

from __future__ import annotations

import datetime as _dt
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import pandas as _pd

from ai_trading_bot.adapters.data import (
    DataQualityError,
    DataStalenessError,
    MarketDataAdapter,
    assert_fresh,
)
from ai_trading_bot.backtest.costs import CostModel
from ai_trading_bot.config import AppConfig
from ai_trading_bot.domain import (
    AccountState,
    Instrument,
    Signal,
    SignalDirection,
    Timeframe,
    utc_now_ns,
)
from ai_trading_bot.execution import ExecutionEngine, ExecutionRouter
from ai_trading_bot.features import indicators
from ai_trading_bot.monitoring import get_logger
from ai_trading_bot.persistence import BarStore, EventLog
from ai_trading_bot.risk import RiskContext, RiskGate
from ai_trading_bot.signal import ExpectedEdgeFilter, SignalFilter
from ai_trading_bot.strategy import BUILTIN_STRATEGIES, Strategy, make_hold

logger = get_logger("engine")


@dataclass
class Heartbeat:
    """Consumed by the dead-man's switch (REQ-RSK-34, REQ-MON-06)."""

    last_ok_ns: int = field(default_factory=utc_now_ns)

    def beat(self) -> None:
        self.last_ok_ns = utc_now_ns()

    def age_seconds(self) -> float:
        return (utc_now_ns() - self.last_ok_ns) / 1e9


class TradingEngine:
    def __init__(
        self,
        config: AppConfig,
        market_data: dict[object, MarketDataAdapter],  # keyed by AssetClass
        store: BarStore | None = None,
        eventlog: EventLog | None = None,
        risk: RiskGate | None = None,
        heartbeat: Heartbeat | None = None,
        strategies: dict[str, Strategy] | None = None,
        filter_: SignalFilter | None = None,
        edge: ExpectedEdgeFilter | None = None,
        executor: ExecutionEngine | None = None,
        router: ExecutionRouter | None = None,
    ) -> None:
        self.cfg = config
        self._data = market_data
        self._store = store or BarStore(config.data.store_dir)
        self._events = eventlog or EventLog()
        self._risk = risk or RiskGate(config.risk)
        self._heartbeat = heartbeat or Heartbeat()
        # Paper capital from config; fills sync real ledger state (REQ-EXE-03).
        initial = config.risk.initial_equity
        self._account = AccountState(
            equity=initial, start_of_day_equity=initial, peak_equity=initial,
        )
        self._last_close: dict[str, float] = {}
        self._router = router or ExecutionRouter(
            config, reference_price=self._resolve_reference_price
        )
        self._executor = executor or ExecutionEngine(
            config, self._router, self._account, eventlog=self._events,
        )
        # Config-driven strategy set (REQ-STR-10); parameter overrides apply here.
        base = dict(BUILTIN_STRATEGIES)
        if strategies:
            base.update(strategies)
        self._strategies = self._resolve_strategies(base)
        self._filter = filter_ or SignalFilter(
            min_confidence=config.risk.min_confidence,
            cooldown_seconds=config.risk.cooldown_after_exit_seconds,
        )
        self._edge = edge or ExpectedEdgeFilter(
            costs=CostModel.from_dict(config.costs.model_dump()),
            min_multiple=config.costs.min_edge_multiple,
        )

    def _resolve_reference_price(self, instrument_id: str) -> float | None:
        """Feed the paper broker the latest close for a given instrument."""
        return self._last_close.get(instrument_id)

    def _resolve_strategies(self, available: dict[str, Strategy]) -> dict[str, Strategy]:
        active: dict[str, Strategy] = {}
        for sid in self.cfg.strategy.active:
            base = available.get(sid)
            if base is None:
                logger.warning("strategy %r not in registry; skipping", sid)
                continue
            params = self.cfg.strategy.params.get(sid) or {}
            active[sid] = base.__class__(**params) if params else base
        return active

    # -- data -------------------------------------------------------------
    def _adapter(self, instrument: Instrument) -> MarketDataAdapter:
        return self._data[instrument.asset_class]

    def _fetch(self, instrument: Instrument, timeframe: Timeframe) -> Sequence:
        adapter = self._adapter(instrument)
        end = _dt.datetime.now(_dt.UTC)
        start = end - _dt.timedelta(seconds=60 * self.cfg.data.buffer_bars)
        bars = adapter.fetch_bars(instrument, timeframe, start, end)
        self._store.append_bars(bars)
        # Pause, don't spike orders, on stale/corrupt data (BR-08, REQ-RSK-32).
        assert_fresh(adapter.last_update_ns, self.cfg.data.stale_after_seconds, instrument.id)
        return bars

    # -- strategy + risk + execution -------------------------------------
    def _decide(self, instrument: Instrument, timeframe: Timeframe, bars: Sequence) -> Signal:
        """Run active strategies, apply signal filter, return the highest-confidence pick."""
        _ = timeframe  # available for multi-TF strategies later
        if not self._strategies:
            return make_hold(instrument, strategy_id="none", reason="NO_STRATEGY")
        best: Signal | None = None
        edge_blocked: tuple[str, float] | None = None  # (strategy_id, confidence)
        for sid, strategy in self._strategies.items():
            try:
                signal = strategy.generate(instrument, bars)
            except Exception:
                logger.exception("strategy %s failed on %s", sid, instrument.id)
                continue
            signal = self._filter.apply(signal)
            if signal.direction is SignalDirection.HOLD:
                continue
            # REQ-SIG-04: reject when expected reward cannot clear costs/slippage.
            pre_edge = signal
            signal = self._edge.apply(signal, bars)
            if signal.direction is SignalDirection.HOLD:
                if edge_blocked is None or pre_edge.confidence > edge_blocked[1]:
                    edge_blocked = (sid, pre_edge.confidence)
                continue
            if best is None or signal.confidence > best.confidence:
                best = signal
        if best is not None:
            return best
        if edge_blocked is not None:
            # Auditable: a strategy fired but lacked the edge to clear costs.
            return make_hold(instrument, strategy_id=edge_blocked[0], reason="NO_EDGE")
        return make_hold(instrument, strategy_id="none", reason="NO_SIGNAL")

    _ATR_PERIOD = 14
    _MIN_STOP_PCT = 0.001  # stops closer than 0.1% are degenerate

    def _enrich_signal(self, signal: Signal, bars: Sequence) -> Signal:
        """Attach trade-execution attributes the risk gate requires (REQ-RSK-01..06).

        Stop = ATR x ``backtest.stop_atr_mult`` around the last close (matches the
        backtest, so live and sim share stop logic); target = ``risk.min_risk_reward``
        x risk distance (matches REQ-RSK-02); reference price = last close. If ATR is
        not yet available (warm-up) the signal is returned unmodified and the risk
        gate's NO_STOP_LOSS rule rejects it — a hard stop is never invented.
        """
        if signal.direction in (SignalDirection.HOLD, SignalDirection.CLOSE):
            return signal
        highs = _pd.Series([b.high for b in bars], dtype="float64")
        lows = _pd.Series([b.low for b in bars], dtype="float64")
        closes = _pd.Series([b.close for b in bars], dtype="float64")
        atr = indicators.atr(highs, lows, closes, self._ATR_PERIOD)
        ref = float(closes.iloc[-1])
        if len(atr) == 0 or not math.isfinite(atr.iloc[-1]):
            return signal  # risk gate will reject with NO_STOP_LOSS (safe default)
        stop_dist = max(float(atr.iloc[-1]) * self.cfg.backtest.stop_atr_mult,
                        ref * self._MIN_STOP_PCT)
        direction = 1 if signal.direction is SignalDirection.BUY else -1
        return replace(
            signal,
            reference_price=ref,
            stop_loss_px=ref - direction * stop_dist,
            take_profit_px=ref + direction * stop_dist * self.cfg.risk.min_risk_reward,
            risk_per_trade_pct=signal.risk_per_trade_pct or self.cfg.risk.per_trade_risk_pct_default,
        )

    def _positions_notional(self) -> dict[str, float]:
        """Current open-position notionals by instrument (Phase 3 fills these)."""
        return {
            pid: abs(pos.qty) * pos.avg_entry_price
            for pid, pos in self._account.open_positions.items()
        }

    def __process(self, instrument: Instrument, timeframe: Timeframe) -> None:
        bars = self._fetch(instrument, timeframe)
        if not bars:
            logger.warning("no bars for %s; skipping", instrument.id)
            return
        self._last_close[instrument.id] = float(bars[-1].close)

        signal = self._decide(instrument, timeframe, bars)
        self._events.record_signal(signal)
        if signal.direction is SignalDirection.HOLD:
            logger.debug("%s HOLD (%s)", instrument.id, signal.reason_code)
            return

        signal = self._enrich_signal(signal, bars)
        context = RiskContext(reference_price=signal.reference_price,
                              now_ns=signal.created_utc_ns)
        decision = self._risk.check(
            signal, self._account, context=context, positions=self._positions_notional(),
        )
        self._events.record_risk_decision(signal, decision)
        if not decision.approved:
            logger.info("%s signal REJECTED by risk: %s", instrument.id, decision.reason_code)
            return

        ref = signal.reference_price or self._last_close[instrument.id]
        result = self._executor.execute(instrument, signal, decision, reference_price=ref)
        for order in result.orders:
            notional = (order.filled_qty or order.qty) * (order.avg_fill_price or ref)
            self._risk.record_order(notional_usd=notional, instrument_id=instrument.id)
        if result.traded:
            logger.info("%s action=%s -> %d order(s), %d fill(s) | %s",
                        instrument.id, result.action, len(result.orders),
                        len(result.fills), result.note or "risk-approved paper mode")
        else:
            logger.info("%s execution: %s | %s", instrument.id, result.action, result.note)

    def process_instrument(self, instrument: Instrument, timeframe: Timeframe) -> None:
        try:
            self.__process(instrument, timeframe)
        except (DataQualityError, DataStalenessError) as exc:
            # Pause, don't spike: kill new orders on bad data (REQ-RSK-32).
            logger.error("data-quality kill: %s", exc)
            self._risk.kill()
        except Exception:  # pragma: no cover - defensive; never crash the loop
            logger.exception("unhandled error processing %s", instrument.id)

    def run_forever(
        self,
        instruments: Sequence[Instrument],
        timeframe: Timeframe = Timeframe.M5,
        poll_interval_s: float = 60.0,
        max_iterations: int | None = None,
    ) -> None:
        logger.info("engine starting profile=%s instruments=%s timeframe=%s",
                    self.cfg.environment, [i.id for i in instruments], timeframe.value)
        iteration = 0
        while max_iterations is None or iteration < max_iterations:
            iteration += 1
            self._heartbeat.beat()
            for instrument in instruments:
                self.process_instrument(instrument, timeframe)
            age = self._heartbeat.age_seconds()
            logger.info("heartbeat ok age=%.1fs iteration=%s", age, iteration)
            time.sleep(poll_interval_s)