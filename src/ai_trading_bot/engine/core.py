"""The 24/7 core engine loop (BRD Section 11, REQ-MON-06).

Wires together: data -> strategy -> signal filter -> risk gate -> execution,
with heartbeat + data-quality staleness handling. Active strategies come from
config (REQ-STR-10); the best filtered signal per instrument is passed to the
risk gate, and execution is paper-logged until Phase 3.
"""

from __future__ import annotations

import datetime as _dt
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

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
from ai_trading_bot.monitoring import get_logger
from ai_trading_bot.persistence import BarStore, EventLog
from ai_trading_bot.risk import RiskGate
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
    ) -> None:
        self.cfg = config
        self._data = market_data
        self._store = store or BarStore(config.data.store_dir)
        self._events = eventlog or EventLog()
        self._risk = risk or RiskGate(config.risk)
        self._heartbeat = heartbeat or Heartbeat()
        self._account = AccountState(equity=0.0)
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

    def __process(self, instrument: Instrument, timeframe: Timeframe) -> None:
        bars = self._fetch(instrument, timeframe)
        if not bars:
            logger.warning("no bars for %s; skipping", instrument.id)
            return

        signal = self._decide(instrument, timeframe, bars)
        self._events.record_signal(signal)
        if signal.direction is SignalDirection.HOLD:
            logger.debug("%s HOLD (%s)", instrument.id, signal.reason_code)
            return

        decision = self._risk.check(signal, self._account)
        self._events.record_risk_decision(signal, decision)
        if not decision.approved:
            logger.info("%s signal REJECTED by risk: %s", instrument.id, decision.reason_code)
            return
        # Phase 3 replaces this with the execution layer's order submission.
        logger.info("%s %s size=%s (risk-approved; paper mode)", instrument.id,
                    signal.direction.value, decision.suggested_size)

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