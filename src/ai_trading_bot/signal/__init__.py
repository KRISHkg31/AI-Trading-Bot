"""Signal generation helpers (REQ-SIG-*).

- :class:`SignalFilter` enforces min-confidence and the anti-whipsaw cooldown.
- The :class:`Signal` domain object carries direction, confidence, size, horizon,
  and a machine-readable reason code.

Expected-edge filtering (REQ-SIG-04) is applied alongside trade-level risk checks.
"""

from ai_trading_bot.signal.edge import ExpectedEdgeFilter
from ai_trading_bot.signal.filters import SignalFilter

__all__ = ["ExpectedEdgeFilter", "SignalFilter"]