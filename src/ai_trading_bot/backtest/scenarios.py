"""Deterministic historical stress scenarios (REQ-BT-05).

Each scenario is a pure function that transforms a sequence of
canonical :class:`Bar` objects without modifying the originals.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy

from ai_trading_bot.domain import Bar


def _scale(bars: list[Bar], idx: int, factor: float) -> None:
    """Multiply OHLCV at *idx* by *factor* in-place on the copied list."""
    old = bars[idx]
    bars[idx] = Bar(
        instrument_id=old.instrument_id,
        timeframe=old.timeframe,
        ts_utc_ns=old.ts_utc_ns,
        open=old.open * factor,
        high=old.high * factor,
        low=old.low * factor,
        close=old.close * factor,
        volume=old.volume,
        source=old.source,
    )


def _slide(bars: list[Bar], idx: int, offset: float) -> None:
    old = bars[idx]
    bars[idx] = Bar(
        instrument_id=old.instrument_id,
        timeframe=old.timeframe,
        ts_utc_ns=old.ts_utc_ns,
        open=old.open + offset,
        high=old.high + offset,
        low=old.low + offset,
        close=old.close + offset,
        volume=old.volume,
        source=old.source,
    )


def flash_crash(
    bars: Sequence[Bar],
    *,
    start_frac: float = 0.50,
    crash_pct: float = 0.30,
    crash_bars: int = 6,
    recovery_bars: int = 24,
    recovery_pct: float = 0.50,
) -> list[Bar]:
    """Abrupt N-bar crash followed by partial recovery (2020 COVID / flash-crash style)."""
    out = deepcopy(list(bars))
    n = len(out)
    start = max(1, int(n * start_frac))
    base = out[start - 1].close
    drop = crash_pct * base

    for i in range(start, min(start + crash_bars, n)):
        frac = (i - start + 1) / crash_bars
        target = base - drop * frac
        _scale(out, i, target / out[i].close if out[i].close else 1.0)

    if start + crash_bars < n:
        bottom = base - drop
        recover_target = bottom + drop * recovery_pct
        for i in range(start + crash_bars, min(start + crash_bars + recovery_bars, n)):
            idx = i - start - crash_bars
            frac = (idx + 1) / recovery_bars
            target = bottom + (recover_target - bottom) * frac
            _scale(out, i, target / out[i].close if out[i].close else 1.0)
    return out


def bear_market(
    bars: Sequence[Bar],
    *,
    start_frac: float = 0.35,
    decline_pct: float = 0.45,
    duration_frac: float = 0.30,
    recovery_frac: float = 0.50,
) -> list[Bar]:
    """Gradual multi-bar grind lower + partial recovery (2022 crypto bear)."""
    out = deepcopy(list(bars))
    n = len(out)
    start = max(1, int(n * start_frac))
    length = max(1, int(n * duration_frac))
    end = min(n, start + length)
    base = out[start - 1].close
    target_low = base * (1.0 - decline_pct)
    for i in range(start, end):
        frac = (i - start) / max(1, end - start - 1)
        target = base + (target_low - base) * frac
        _scale(out, i, target / out[i].close if out[i].close else 1.0)

    if end < n:
        recover_len = min(n - end, int(length * recovery_frac))
        recov_target = target_low + (base - target_low) * recovery_frac
        for i in range(end, min(end + recover_len, n)):
            idx = i - end
            frac = idx / max(1, recover_len - 1)
            target = target_low + (recov_target - target_low) * frac
            _scale(out, i, target / out[i].close if out[i].close else 1.0)
    return out


def volatility_spike(
    bars: Sequence[Bar],
    *,
    start_frac: float = 0.50,
    duration_frac: float = 0.10,
    amplitude_mult: float = 4.0,
) -> list[Bar]:
    """Expand bar ranges (High-Low, Open-Close gaps) by a multiplier for a window."""
    out = deepcopy(list(bars))
    n = len(out)
    start = max(1, int(n * start_frac))
    length = max(2, int(n * duration_frac))
    end = min(n, start + length)
    for i in range(start, end):
        old = out[i]
        mid = (old.high + old.low) / 2.0
        half_range = (old.high - old.low) / 2.0
        new_half = half_range * amplitude_mult
        out[i] = Bar(
            instrument_id=old.instrument_id,
            timeframe=old.timeframe,
            ts_utc_ns=old.ts_utc_ns,
            open=old.open,
            high=mid + new_half,
            low=mid - new_half,
            close=old.close,
            volume=old.volume * amplitude_mult,
            source=old.source,
        )
    return out


def gap_down(
    bars: Sequence[Bar],
    *,
    at_frac: float = 0.40,
    gap_pct: float = 0.15,
) -> list[Bar]:
    """One-bar permanent level shift down, then all subsequent bars offset by same amount."""
    out = deepcopy(list(bars))
    n = len(out)
    idx = max(1, min(n - 1, int(n * at_frac)))
    offset = -out[idx].close * gap_pct
    for i in range(idx, n):
        _slide(out, i, offset)
    return out