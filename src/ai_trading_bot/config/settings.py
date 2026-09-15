"""Centralised, versioned configuration (REQ-CFG-01/02/03, REQ-NFR-*).

- Defaults live in ``config/default.yaml``; never hard-coded in modules.
- Environment separation: dev / backtest / paper / live (REQ-CFG-05).
- Any value can be overridden via env var ``AIBOT_<DOTTED_PATH>__<LEAF>``
  (e.g. ``AIBOT_EXECUTION__MODE=live``).
- Secrets come from environment / ``.env`` only via ``get_credentials``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

_ENV_OVERRIDE_PREFIX = "AIBOT_"
_DOT_FILE_PATH = Path(__file__).resolve().parent.parent.parent.parent / "config" / "default.yaml"

_VALID_ENVIRONMENTS = ("dev", "backtest", "paper", "live")


class AppMeta(BaseModel):
    name: str = "ai-trading-bot"
    timezone: str = "UTC"
    heartbeat_interval_seconds: int = 10


class DataConfig(BaseModel):
    default_timeframe: str = "5m"
    buffer_bars: int = 200
    stale_after_seconds: int = 30
    retry_backoff_seconds: float = 1.0
    store_dir: str = "data/raw"


class RiskLimits(BaseModel):
    """All risk thresholds are config-driven, never code (BR-07)."""

    initial_equity: float = 10_000.0  # paper capital; real fills sync it (Phase 3)
    per_trade_risk_pct_default: float = 1.0  # REQ-RSK-13, default 0.5-2%
    min_risk_reward: float = 1.5  # REQ-RSK-02, reward >= 1.5x risk
    min_confidence: float = 0.5  # REQ-RSK-05 / REQ-SIG-03
    max_daily_loss_pct: float = 3.0  # REQ-RSK-21, session halt at 3-5%
    max_drawdown_from_peak_pct: float = 15.0  # REQ-RSK-22, halt + manual review at 15-20%
    max_symbol_exposure_pct: float = 10.0  # REQ-RSK-14
    max_total_exposure_pct: float = 60.0  # REQ-RSK-23
    cooldown_after_exit_seconds: int = 300  # REQ-RSK-06, anti-whipsaw
    slippage_tolerance_pct: float = 0.1  # REQ-RSK-03 / REQ-EXE-06
    circuit_breaker_max_orders_per_minute: int = 10  # REQ-RSK-31
    order_history_window_seconds: int = 60
    max_order_notional_usd: float = 250_000.0  # REQ-RSK-10 sane-caps the order size
    sizing_method: str = "fractional"  # fractional | atr | kelly (REQ-RSK-11/12)
    kelly_fraction: float = 0.25  # conservative multiplier for the kelly method


class ExecutionConfig(BaseModel):
    """live | paper | dry_run. Paper/dry-run default until rollout gates pass (REQ-EXE-07/08)."""

    mode: str = "paper"
    default_order_type: str = "market"
    retry_max_attempts: int = 3
    idempotency_key_prefix: str = "tb"

    @field_validator("mode")
    @classmethod
    def _mode_valid(cls, v: str) -> str:
        if v not in ("live", "paper", "dry_run"):
            raise ValueError(f"execution.mode must be live|paper|dry_run, got {v!r}")
        return v


class StrategyConfig(BaseModel):
    """Active strategies and parameter overrides — switch/tune without redeploys (REQ-STR-10)."""

    active: list[str] = Field(default_factory=lambda: ["ema_trend", "macd_crossover"])
    params: dict[str, dict[str, Any]] = Field(default_factory=dict)


class CostConfig(BaseModel):
    """Realistic execution cost model — shared by backtests and the edge filter (REQ-BT-02, REQ-SIG-04)."""

    taker_fee_bps: float = 5.0
    maker_fee_bps: float = 2.0
    spread_bps: float = 2.0
    slippage_bps: float = 1.0
    fixed_usd: float = 0.0
    min_edge_multiple: float = 2.0  # expected reward must clear >=Nx round-trip cost


class BacktestConfig(BaseModel):
    """Defaults for the cost-aware backtester (REQ-BT-01..07)."""

    initial_equity: float = 10_000.0
    risk_per_trade_pct: float = 1.0
    stop_atr_mult: float = 2.0  # stop distance = N x ATR
    take_profit_rr: float = 1.5  # matches risk.min_risk_reward
    max_leverage: float = 1.0  # cap notional <= equity
    bars_per_year: int = 105_120  # calendar-year count of 5m bars
    walk_forward_folds: int = 3
    walk_forward_test_frac: float = 0.25
    seed: int = 42


class MonitoringConfig(BaseModel):
    alert_channels: list[str] = Field(default_factory=list)
    log_dir: str = "logs"
    log_level: str = "INFO"


class ModelsConfig(BaseModel):
    """ML model registry + retraining knobs (Phase 4)."""

    model_dir: str = "data/models"
    label_horizon_bars: int = 12  # forward-return label over 12 x 5m bars = 1h
    min_abs_move: float = 0.001  # |forward return| below this = neutral, dropped
    val_frac: float = 0.2  # held-out walk-forward validation fraction
    min_improvement: float = 0.01  # promote only when validation accuracy beats incumbent by this
    threshold: float = 0.55  # strategy minimum P(class) to emit a direction


class AppConfig(BaseModel):
    environment: str = "dev"
    app: AppMeta = Field(default_factory=AppMeta)
    data: DataConfig = Field(default_factory=DataConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    costs: CostConfig = Field(default_factory=CostConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    risk: RiskLimits = Field(default_factory=RiskLimits)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)

    @field_validator("environment")
    @classmethod
    def _env_valid(cls, v: str) -> str:
        if v not in _VALID_ENVIRONMENTS:
            raise ValueError(f"environment must be one of {_VALID_ENVIRONMENTS}, got {v!r}")
        return v

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


def _load_yaml_defaults() -> dict[str, Any]:
    with open(_DOT_FILE_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _apply_env_overrides(raw: dict[str, Any], environ: dict[str, str] | None = None) -> dict[str, Any]:
    """Merge ``AIBOT_SECTION__KEY=VALUE`` into the nested dict (create sections as needed)."""
    environ = environ if environ is not None else os.environ
    for key, value in environ.items():
        if not key.startswith(_ENV_OVERRIDE_PREFIX):
            continue
        path = [p.lower() for p in key[len(_ENV_OVERRIDE_PREFIX):].split("__")]
        node = raw
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = _coerce(value)
    return raw


def _coerce(value: str) -> Any:
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    return value


_CONFIG_CACHE: dict[str, AppConfig] = {}


def load_config(profile: str | None = None) -> AppConfig:
    """Load config for a given environment profile (default: $AIBOT_ENVIRONMENT or 'dev')."""
    env = profile or os.environ.get("AIBOT_ENVIRONMENT") or "dev"
    if env in _CONFIG_CACHE:
        return _CONFIG_CACHE[env]

    load_dotenv()  # idempotent; pulls `.env` if present
    raw = _load_yaml_defaults()
    raw["environment"] = env
    raw = _apply_env_overrides(raw)
    config = AppConfig.model_validate(raw)
    _CONFIG_CACHE[env] = config
    return config


@dataclass(frozen=True, slots=True)
class Credentials:
    """Optional broker/alert keys. All of them can be absent in dev/backtest."""

    binance_api_key: str | None = None
    binance_api_secret: str | None = None
    binance_testnet: bool = False
    coinbase_api_key: str | None = None
    coinbase_api_secret: str | None = None
    coinbase_api_passphrase: str | None = None
    alpaca_api_key: str | None = None
    alpaca_api_secret: str | None = None
    alpaca_paper: bool = True
    zerodha_api_key: str | None = None
    zerodha_api_secret: str | None = None
    zerodha_access_token: str | None = None
    telegram_bot_token: str | None = None
    slack_webhook_url: str | None = None
    anthropic_api_key: str | None = None


def get_credentials() -> Credentials:
    """Read secrets from the environment / ``.env``. Never logs or persists them."""
    return Credentials(
        binance_api_key=os.environ.get("BINANCE_API_KEY"),
        binance_api_secret=os.environ.get("BINANCE_API_SECRET"),
        binance_testnet=os.environ.get("BINANCE_TESTNET", "").lower() == "true",
        coinbase_api_key=os.environ.get("COINBASE_API_KEY"),
        coinbase_api_secret=os.environ.get("COINBASE_API_SECRET"),
        coinbase_api_passphrase=os.environ.get("COINBASE_API_PASSPHRASE"),
        alpaca_api_key=os.environ.get("ALPACA_API_KEY"),
        alpaca_api_secret=os.environ.get("ALPACA_API_SECRET"),
        alpaca_paper=os.environ.get("ALPACA_PAPER", "true").lower() != "false",
        zerodha_api_key=os.environ.get("ZERODHA_API_KEY"),
        zerodha_api_secret=os.environ.get("ZERODHA_API_SECRET"),
        zerodha_access_token=os.environ.get("ZERODHA_ACCESS_TOKEN"),
        telegram_bot_token=os.environ.get("TELEGRAM_BOT_TOKEN"),
        slack_webhook_url=os.environ.get("SLACK_WEBHOOK_URL"),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )


def validate_config(config: AppConfig) -> list[str]:
    """Sanity checks across sections (catches misconfig before deployment, BR-07)."""
    problems: list[str] = []
    r = config.risk
    if r.max_daily_loss_pct <= 0:
        problems.append("risk.max_daily_loss_pct must be > 0")
    if r.max_drawdown_from_peak_pct <= 0:
        problems.append("risk.max_drawdown_from_peak_pct must be > 0")
    if not (0.0 < r.min_confidence < 1.0):
        problems.append("risk.min_confidence must be in (0, 1)")
    if r.per_trade_risk_pct_default <= 0 or r.per_trade_risk_pct_default > 5:
        problems.append("risk.per_trade_risk_pct_default outside sane range (0, 5]")
    return problems