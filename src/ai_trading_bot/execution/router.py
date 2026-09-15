"""Execution router: pick the broker adapter for an instrument (REQ-EXE-02).

Resolves ``execution.mode``:
- ``dry_run`` -> :class:`DryRunAdapter` for every market.
- ``paper``   -> :class:`PaperBroker` for every market (default).
- ``live``    -> key-guarded vendor adapter; any call raises
  :class:`LiveUnavailableError` until the Phase 6 go-live gate is cleared.
"""

from __future__ import annotations

from ai_trading_bot.adapters.broker import BrokerAdapter
from ai_trading_bot.adapters.broker.dry_run import DryRunAdapter
from ai_trading_bot.adapters.broker.live import build_live_brokers
from ai_trading_bot.adapters.broker.paper import PaperBroker
from ai_trading_bot.backtest.costs import CostModel
from ai_trading_bot.config import AppConfig, Credentials
from ai_trading_bot.domain import AssetClass, Instrument

#: vendor namespace for an AssetClass (matches Credentials + live adapter keys).
_VENDOR_FOR_ASSET: dict[AssetClass, str] = {
    AssetClass.CRYPTO: "binance",
    AssetClass.US_EQUITIES: "alpaca",
    AssetClass.IN_EQUITIES: "zerodha",
}


class ExecutionRouter:
    def __init__(
        self,
        config: AppConfig,
        credentials: Credentials | None = None,
        reference_price: object | None = None,
        live_brokers: dict[str, BrokerAdapter] | None = None,
    ) -> None:
        self.cfg = config
        mode = config.execution.mode
        if mode == "live":
            self._live = live_brokers or build_live_brokers(
                credentials or Credentials(), config.environment
            )
        elif mode == "dry_run":
            self._default = DryRunAdapter()
        else:  # paper
            cost_model = CostModel.from_dict(config.costs.model_dump())
            self._default = PaperBroker(cost_model=cost_model, reference_price=reference_price)

    @property
    def mode(self) -> str:
        return self.cfg.execution.mode

    def get(self, instrument: Instrument) -> BrokerAdapter:
        if self.mode != "live":
            return self._default
        vendor = _VENDOR_FOR_ASSET.get(instrument.asset_class)
        if vendor is None:
            raise ValueError(f"no live broker for asset {instrument.asset_class}")
        return self._live[vendor]


__all__ = ["ExecutionRouter"]