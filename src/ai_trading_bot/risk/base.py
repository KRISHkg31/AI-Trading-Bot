"""The independent four-tier risk engine (REQ-RSK-*, BR-01..08).

No order is placed unless :meth:`RiskGate.check` approves. Checks run in
defense-in-depth order so a hardened state is never masked by later rules:

  Tier 4 (system)     kill switch / data-quality kill, circuit breaker
  Tier 3 (portfolio)  daily-loss halt, drawdown-from-peak halt + manual review
  Tier 1 (trade)      stop-loss required, min R:R, slippage, confidence, sanity
  Tier 2 (position)   risk-based sizing capped to symbol/total/notional limits

Every rejection carries a machine-readable reason code (catalogued in
:data:`RiskGate.REASON_CODE_CATALOG`) and is recorded by the surrounding
monitoring, so rejections are auditable (REQ-RSK-26).

Sizing (REQ-RSK-11..14): the risk engine sizes a position as
``risk budget / per-unit risk`` (fractional) and then *scales it down* to fit
the symbol, total-exposure and notional caps. A valid signal is never rejected
for being too large — only for a violation that cannot be fixed (missing stop,
bad risk-reward, low confidence, or a cap already exhausted). ``kelly`` scales
the fractional risk budget by :attr:`~RiskLimits.kelly_fraction` as a
conservative placeholder until Phase 4 supplies model win/loss statistics.
"""

from __future__ import annotations

import math
import time

from ai_trading_bot.config import RiskLimits
from ai_trading_bot.domain import AccountState, RiskDecision, Signal, SignalDirection

#: Float margin on the R:R comparison so a signal built at exactly
#: ``target = stop_dist * min_risk_reward`` is never round-tripped into a reject.
_RR_EPS = 1e-9


class RiskGate:
    """Independent gate between signal generation and execution."""

    #: Machine-readable rejections, catalogued for reporting (REQ-RSK-26).
    REASON_CODE_CATALOG = (
        "OK",
        "SYSTEM_FLAT",
        "SESSION_HALTED",
        "CIRCUIT_BREAKER",
        "MIN_RISK_REWARD",
        "NO_STOP_LOSS",
        "LOW_CONFIDENCE",
        "SLIPPAGE_TOO_LARGE",
        "UNFEASIBLE",
        "SYMBOL_EXPOSURE_LIMIT",
        "TOTAL_EXPOSURE_LIMIT",
        "ORDER_TOO_LARGE",
        "DAILY_LOSS_LIMIT",
        "DRAWDOWN_LIMIT",
    )

    def __init__(self, limits: RiskLimits, now_ns_fn=time.time_ns) -> None:
        self._limits = limits
        self._now_ns = now_ns_fn
        self._flattened = False  # kill switch / data-quality kill (Tier 4)
        self._halted_reason: str | None = None  # Tier 3 session halt
        self._order_ts_ns: list[int] = []  # circuit-breaker window (Tier 4)

    # -- status -----------------------------------------------------------
    @property
    def halted(self) -> str | None:
        """Reason the gate is closed, if any (else ``None``). System kill first."""
        if self._flattened:
            return "SYSTEM_FLAT"
        return self._halted_reason

    @property
    def flattened(self) -> bool:
        return self._flattened

    # -- manual / system controls (Tier 4) --------------------------------
    def kill(self) -> None:
        """Manual/system kill switch: stop all new orders immediately (REQ-RSK-33)."""
        self._flattened = True

    def resume(self) -> None:
        """Re-open the gate after manual review cleared the condition."""
        self._flattened = False
        self._halted_reason = None

    def record_order(self, notional_usd: float | None = None, instrument_id: str | None = None,
                     now_ns: int | None = None) -> None:
        """Feed an allowed order into the circuit-breaker window (REQ-RSK-31)."""
        _ = notional_usd, instrument_id  # breaker is rate-based for now
        self._order_ts_ns.append(now_ns if now_ns is not None else self._now_ns())

    # -- monitoring -------------------------------------------------------
    def assess(self, account: AccountState) -> str | None:
        """Signal-independent Tier-3 assessment (REQ-RSK-21/22).

        ``check`` only discovers a daily-loss/drawdown halt when a *signal*
        passes through it; this lets the monitoring layer evaluate the same
        conditions every poll so a halt alerts even when no signal is live.
        The halt takes hold exactly as in ``check``; recovery is manual via
        :meth:`resume`.
        """
        r = self._limits
        if account.day_loss_pct >= r.max_daily_loss_pct:
            self._halted_reason = "DAILY_LOSS_LIMIT"
            return "DAILY_LOSS_LIMIT"
        if account.drawdown_from_peak_pct >= r.max_drawdown_from_peak_pct:
            self._halted_reason = "DRAWDOWN_LIMIT"
            return "DRAWDOWN_LIMIT"
        return None

    # -- main entry -------------------------------------------------------
    def check(
        self,
        signal: Signal,
        account: AccountState,
        context: RiskContext | None = None,
        positions: dict[str, float] | None = None,
    ) -> RiskDecision:
        """Validate a signal against all applicable risk tiers (BR-01: no order unapproved)."""
        r = self._limits

        # --- Tier 4: system-level ----------------------------------------
        if self._flattened:
            return RiskDecision(False, "SYSTEM_FLAT", details=("kill switch / data-quality kill",))
        if self._halted_reason is not None:
            return RiskDecision(False, "SESSION_HALTED",
                                details=(self._halted_reason, "manual review required"))
        if not self._circuit_breaker_ok():
            return RiskDecision(False, "CIRCUIT_BREAKER",
                                details=("exceeded orders-per-minute burst limit",))

        # --- Tier 3: portfolio-level halts --------------------------------
        if account.day_loss_pct >= r.max_daily_loss_pct:
            self._halted_reason = "DAILY_LOSS_LIMIT"
            return RiskDecision(False, "DAILY_LOSS_LIMIT",
                                details=(f"day loss {account.day_loss_pct:.2f}% >= {r.max_daily_loss_pct}%",))
        if account.drawdown_from_peak_pct >= r.max_drawdown_from_peak_pct:
            self._halted_reason = "DRAWDOWN_LIMIT"
            return RiskDecision(False, "DRAWDOWN_LIMIT",
                                details=(f"drawdown {account.drawdown_from_peak_pct:.2f}% >= {r.max_drawdown_from_peak_pct}%",))

        if signal.direction is SignalDirection.HOLD:
            return RiskDecision(True, "OK")

        # Market reference price (what a market order would hit now) comes
        # from the context when supplied; the signal's own reference_price is
        # the strategy's expected fill. Their divergence is the slippage check.
        market_ref = context.reference_price if context and context.reference_price is not None \
            else signal.reference_price
        proposed = signal.reference_price

        # --- Tier 1: trade-level rules ------------------------------------
        first = self._trade_checks(signal, market_ref, proposed)
        if first is not None:
            return first

        # --- Tier 2: sizing + position-level caps -------------------------
        return self._size_and_cap(signal, account, positions)

    # -- Tier 1 -----------------------------------------------------------
    def _trade_checks(self, signal: Signal, ref: float | None, proposed: float | None) -> RiskDecision | None:
        r = self._limits
        if signal.confidence < r.min_confidence:
            return RiskDecision(False, "LOW_CONFIDENCE",
                                details=(f"confidence {signal.confidence:.2f} < {r.min_confidence}",))
        if ref is None or ref <= 0:
            return RiskDecision(False, "UNFEASIBLE", details=("missing reference price",))
        size = signal.suggested_size
        if size is not None and (size < 0 or not math.isfinite(size)):
            return RiskDecision(False, "UNFEASIBLE", details=("invalid suggested size",))
        if signal.stop_loss_px is None:
            return RiskDecision(False, "NO_STOP_LOSS",
                                details=("every trade must carry a stop-loss (REQ-RSK-01)",))
        stop_dist = abs(signal.stop_loss_px - ref)
        if stop_dist <= 0:
            return RiskDecision(False, "NO_STOP_LOSS", details=("stop equals entry (degenerate)",))
        if signal.take_profit_px is not None:
            reward = abs(signal.take_profit_px - ref)
            # Multiply through rather than divide: exact at the 1.5 boundary.
            if reward < stop_dist * r.min_risk_reward - _RR_EPS:
                rr = reward / stop_dist
                return RiskDecision(False, "MIN_RISK_REWARD",
                                    details=(f"R:R {rr:.2f} < {r.min_risk_reward}",))
        if not self._slippage_ok(ref, proposed):
            pct = abs((proposed or ref) - ref) / ref * 100.0
            details = (f"slippage {pct:.2f}% > {r.slippage_tolerance_pct}% tolerance",)
            return RiskDecision(False, "SLIPPAGE_TOO_LARGE", details=details)
        return None

    def _slippage_ok(self, ref: float, proposed: float | None) -> bool:
        r = self._limits
        if proposed is None:
            return True
        return abs(proposed - ref) / ref <= r.slippage_tolerance_pct / 100.0

    # -- Tier 2 -----------------------------------------------------------
    def _size_and_cap(self, signal: Signal, account: AccountState,
                      positions: dict[str, float] | None) -> RiskDecision:
        r = self._limits
        ref = signal.reference_price or 0.0
        if ref <= 0:
            return RiskDecision(False, "UNFEASIBLE", details=("no reference price for sizing",))

        existing = positions or {}
        now = account.equity
        # Step 1: reject only when a cap is already exhausted, cannot be fixed.
        sym_open = existing.get(signal.instrument_id, 0.0)
        if sym_open >= now * r.max_symbol_exposure_pct / 100.0:
            msg = (f"symbol exposure {sym_open:.2f} already at cap "
                   f"{r.max_symbol_exposure_pct}%")
            return RiskDecision(False, "SYMBOL_EXPOSURE_LIMIT", details=(msg,))
        total_open = sum(existing.values())
        if total_open >= now * r.max_total_exposure_pct / 100.0:
            msg = (f"total exposure {total_open:.2f} already at cap "
                   f"{r.max_total_exposure_pct}%")
            return RiskDecision(False, "TOTAL_EXPOSURE_LIMIT", details=(msg,))

        # Step 2: size-down to fit every cap (a big signal is never rejected,
        # merely scaled). Risk-based size first, then applied caps.
        risk_pct = signal.risk_per_trade_pct or r.per_trade_risk_pct_default
        size = self._size_for_risk(signal, now, ref, risk_pct)
        if size <= 0:
            return RiskDecision(False, "UNFEASIBLE", details=("risk sizing returned <= 0",))
        # Longs and shorts draw on the same exposure budget (single-sided for now).
        sym_budget = now * r.max_symbol_exposure_pct / 100.0 - sym_open
        total_budget = now * r.max_total_exposure_pct / 100.0 - total_open
        size = min(size, sym_budget / ref, total_budget / ref, r.max_order_notional_usd / ref)
        notional = size * ref

        # Step 3: hard safety net (BR-07, REQ-RSK-10). Unreachable under the
        # size-down above; kept so a misconfigured cap cannot pass an order.
        if notional > r.max_order_notional_usd + _RR_EPS:
            return RiskDecision(False, "ORDER_TOO_LARGE",
                                details=(f"notional {notional:.2f} > cap {r.max_order_notional_usd:.2f}",))
        return RiskDecision(True, "OK", suggested_size=size)

    def _size_for_risk(self, signal: Signal, equity: float, ref: float, risk_pct: float) -> float:
        """Risk-based position size: risk budget / per-unit risk (REQ-RSK-11..13)."""
        r = self._limits
        if signal.stop_loss_px is not None:
            stop_dist = abs(signal.stop_loss_px - ref)
        else:
            stop_dist = ref * 0.01  # fallback; _trade_checks already requires a stop
        if stop_dist <= 0:
            return 0.0
        risk_amount = equity * risk_pct / 100.0
        if r.sizing_method == "kelly":
            risk_amount *= r.kelly_fraction  # conservative until Phase 4 stats
        size = risk_amount / stop_dist
        if not math.isfinite(size) or size <= 0:
            return 0.0
        return size

    # -- Tier 4 helper ----------------------------------------------------
    def _circuit_breaker_ok(self) -> bool:
        r = self._limits
        if r.circuit_breaker_max_orders_per_minute <= 0:
            return True
        cutoff = self._now_ns() - r.order_history_window_seconds * 1_000_000_000
        recent = [t for t in self._order_ts_ns if t >= cutoff]
        self._order_ts_ns = recent
        return len(recent) < r.circuit_breaker_max_orders_per_minute


class RiskContext:
    """Extra execution-time data the gate can consult (reference price, ATR, now)."""

    __slots__ = ("atr", "now_ns", "reference_price")

    def __init__(self, reference_price: float | None = None, atr: float | None = None,
                 now_ns: int | None = None) -> None:
        self.reference_price = reference_price
        self.atr = atr
        self.now_ns = now_ns


__all__ = ["RiskContext", "RiskGate"]