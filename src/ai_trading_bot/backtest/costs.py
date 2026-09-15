"""Realistic execution cost model (REQ-BT-02, REQ-SIG-04).

Costs eat a strategy alive if ignored: fees, half-spread, and slippage all
apply every fill. The model is config-driven and identical to what the live
execution layer and expected-edge filter assume, so backtests do not overstate
edge.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class CostModel:
    taker_fee_bps: float = 5.0
    maker_fee_bps: float = 2.0
    spread_bps: float = 2.0  # total spread; crossing it costs spread/2 per side
    slippage_bps: float = 1.0
    fixed_usd: float = 0.0

    def side_cost_pct(self, market: bool = True) -> float:
        """One-way all-in cost as a fraction of notional."""
        fee = self.taker_fee_bps if market else self.maker_fee_bps
        return (fee + self.spread_bps / 2.0 + self.slippage_bps) / 1e4

    def round_trip_cost_pct(self, market: bool = True) -> float:
        return 2.0 * self.side_cost_pct(market)

    def fill_price(self, reference: float, side: int, market: bool = True) -> float:
        """Fill price after adverse move. ``side`` is +1 (buy) or -1 (sell)."""
        return reference * (1.0 + side * self.side_cost_pct(market))

    def fees_usd(self, notional: float, market: bool = True) -> float:
        return notional * self.side_cost_pct(market) + self.fixed_usd

    def edge_clear_cost(self, expected_move_pct: float, min_multiple: float = 2.0) -> bool:
        """REQ-SIG-04: is expected reward >= ``min_multiple`` x round-trip cost?"""
        return expected_move_pct >= self.round_trip_cost_pct() * min_multiple

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, float] | None) -> CostModel:
        if not data:
            return cls()
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def geometric_return(final: float, initial: float, years: float) -> float:
    """Annualised return rate; 0.0 when years is degenerate."""
    if initial <= 0 or years <= 0:
        return 0.0
    if final <= 0:
        return -1.0
    return math.exp(math.log(final / initial) / years) - 1.0