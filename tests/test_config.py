import pytest

from ai_trading_bot.config import (
    Credentials,
    get_credentials,
    load_config,
    validate_config,
)


def test_load_dev_config_defaults() -> None:
    cfg = load_config("dev")
    assert cfg.environment == "dev"
    assert cfg.data.default_timeframe == "5m"
    assert cfg.risk.max_daily_loss_pct == 3.0
    assert cfg.risk.min_risk_reward == 1.5
    assert cfg.execution.mode == "paper"  # default, until rollout gates pass


def test_invalid_environment_rejected() -> None:
    with pytest.raises(ValueError):
        load_config("prod")  # not one of dev/backtest/paper/live


def test_env_nested_override(monkeypatch) -> None:
    monkeypatch.setenv("AIBOT_RISK__MAX_DAILY_LOSS_PCT", "2.5")
    cfg = load_config("backtest")
    assert cfg.risk.max_daily_loss_pct == 2.5


def test_credentials_all_absent_without_keys(monkeypatch) -> None:
    for k in dir(Credentials()):
        if k.startswith(("binance_", "alpaca_", "zerodha_", "coinbase_")):
            monkeypatch.delenv(k.upper(), raising=False)
    creds = get_credentials()
    assert creds.binance_api_key is None
    assert creds.alpaca_api_key is None
    assert creds.zerodha_api_key is None
    assert creds.binance_testnet is False


def test_validate_config_flags_bad_limits() -> None:
    cfg = load_config("dev")
    # sabotage one limit to prove validation catches it
    bad = cfg.model_copy(deep=True)
    bad.risk.max_daily_loss_pct = 0.0
    problems = validate_config(bad)
    assert any("max_daily_loss_pct" in p for p in problems)
    assert validate_config(cfg) == []