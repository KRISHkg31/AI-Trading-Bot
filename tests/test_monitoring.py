"""Phase 5 observability tests: metrics rows, alert dispatch, transition alerts.

Guarantees that matter here:

- one self-contained ``metric`` snapshot row per engine pass, usable as the
  out-of-process heartbeat (REQ-RSK-34);
- the notifier never raises when channels lack credentials (dev/backtest) and
  posts only to configured channels, with message dedup;
- risk alerts fire on *transitions* (halt/resume), never on every poll, and a
  daily-loss/drawdown breach is caught even with no live signal (assess path);
- the dead-man's switch flattens the risk gate when the engine stops beating.
"""

from __future__ import annotations

from ai_trading_bot.config import Credentials, MonitoringConfig, RiskLimits
from ai_trading_bot.domain import AccountState, Position, utc_now_ns
from ai_trading_bot.monitoring import (
    MetricsRecorder,
    Notifier,
    RiskAlertWatcher,
    latest_metrics,
    metrics_age_seconds,
)
from ai_trading_bot.persistence import EventLog
from ai_trading_bot.risk import RiskGate


def _account(equity: float = 10_000.0) -> AccountState:
    return AccountState(equity=equity, start_of_day_equity=equity, peak_equity=equity)


# -- metrics recorder --------------------------------------------------------


def test_metrics_recorder_writes_self_contained_row(tmp_path) -> None:
    log = EventLog(db_path=tmp_path / "events.sqlite3")
    rec = MetricsRecorder(log, environment="paper")
    now = utc_now_ns()
    rec.record(
        account=_account(),
        last_prices={"binance:BTC/USDT": 65_000.0},
        model_versions={"ml_momentum": "v3"},
        risk_state="OK",
        iteration=7,
        now_ns=now,
    )

    row = latest_metrics(log)
    assert row is not None
    assert row["environment"] == "paper"
    assert row["equity"] == 10_000.0
    assert row["cash"] == 10_000.0
    assert row["risk_state"] == "OK"
    assert row["iteration"] == 7
    assert row["model_versions"] == {"ml_momentum": "v3"}
    assert row["ts_utc_ns"] == now  # ts matches the payload, so age is truthworthy


def test_metrics_age_detects_when_no_engine_ran(tmp_path) -> None:
    log = EventLog(db_path=tmp_path / "events.sqlite3")
    assert latest_metrics(log) is None
    assert metrics_age_seconds(log) is None


# -- notifier ----------------------------------------------------------------


def test_notifier_posts_only_to_configured_channels(tmp_path) -> None:
    posted: list[dict] = []

    def fake_post(url: str, *, json: dict, **_: object) -> object:
        posted.append({"url": url, "json": json})
        class _R:
            status_code = 200
        return _R()

    creds = Credentials(
        telegram_bot_token="tok",
        telegram_chat_id="chat",
        slack_webhook_url="https://hooks.slack.com/xyz",
    )
    notifier = Notifier(creds, channels=["telegram", "slack"], post=fake_post)
    notifier.notify("RISK_HALTED", "daily loss breached")
    assert len(posted) == 2
    tg = next(p for p in posted if "api.telegram.org" in p["url"])
    assert tg["json"] == {"chat_id": "chat", "text": "[RISK_HALTED] daily loss breached"}
    sl = next(p for p in posted if "hooks.slack.com" in p["url"])
    assert sl["json"]["text"].startswith("[RISK_HALTED]")


def test_notifier_dedups_identical_alerts_within_window(tmp_path) -> None:
    posted: list[dict] = []

    def fake_post(url: str, *, json: dict, **_: object) -> object:
        posted.append({})
        class _R:
            status_code = 200
        return _R()

    creds = Credentials(telegram_bot_token="tok", telegram_chat_id="chat")
    notifier = Notifier(creds, channels=["telegram"], post=fake_post)
    notifier.notify("RISK_HALTED", "same text")
    notifier.notify("RISK_HALTED", "same text")  # within dedup window -> suppressed
    assert len(posted) == 1


def test_notifier_logs_instead_of_raising_without_credentials(tmp_path) -> None:
    # No tokens at all, channel configured anyway: log-only, never raise.
    notifier = Notifier(Credentials(), channels=["telegram", "slack"])
    statuses = notifier.notify("RISK_HALTED", "no creds")
    assert statuses == ["telegram:not-configured", "slack:not-configured"]


# -- risk alert watcher ------------------------------------------------------


def test_risk_alert_fires_once_on_halt_transition(tmp_path) -> None:
    log = EventLog(db_path=tmp_path / "events.sqlite3")
    gate = RiskGate(RiskLimits(max_daily_loss_pct=3.0))
    watcher = RiskAlertWatcher(gate, log, Notifier(Credentials()), MonitoringConfig())
    acc = _account()

    # Poll #1: open gate, no alert.
    watcher.poll(acc)
    assert log.query("risk_halted") == []

    # Poll #2: account breaches the daily-loss limit; assess must catch it even
    # with no signal, and fire exactly one alert.
    acc.equity = 9_550.0  # down 4.5% from the 10,000 start -> day_loss_pct 4.5
    watcher.poll(acc)
    halted = log.query("risk_halted")
    assert len(halted) == 1
    assert halted[0]["payload"]["reason"] == "DAILY_LOSS_LIMIT"
    assert halted[0]["payload"]["prior_gate_state"] == "OK"
    assert gate.halted == "DAILY_LOSS_LIMIT"

    # Poll #3: still halted -> no duplicate alert.
    watcher.poll(acc)
    assert len(log.query("risk_halted")) == 1


def test_risk_alert_on_resume_from_flattened(tmp_path) -> None:
    log = EventLog(db_path=tmp_path / "events.sqlite3")
    gate = RiskGate(RiskLimits())
    watcher = RiskAlertWatcher(gate, log, Notifier(Credentials()), MonitoringConfig())

    gate.kill()  # kill switch / data-quality kill (REQ-RSK-32/33)
    watcher.poll(_account())
    assert log.query("risk_halted")[0]["payload"]["reason"] == "SYSTEM_FLAT"

    gate.resume()  # manual review cleared it
    watcher.poll(_account())
    resumed = log.query("risk_resumed")
    assert len(resumed) == 1
    assert resumed[0]["payload"]["prior_gate_state"] in ("SYSTEM_FLAT",)
    # Transition back to OK fires one alert; a further poll stays quiet.
    watcher.poll(_account())
    assert len(log.query("risk_resumed")) == 1


# -- dead-man's switch -------------------------------------------------------


def test_monitor_reports_stale_when_no_metric_rows(tmp_path) -> None:
    log = EventLog(db_path=tmp_path / "events.sqlite3")
    assert metrics_age_seconds(log) is None  # engine never beat -> monitor exits non-zero


def _make_position() -> Position:
    return Position(instrument_id="binance:BTC/USDT", qty=0.5, avg_entry_price=100.0)


def test_metrics_exposure_sums_open_positions_v2(tmp_path) -> None:
    log = EventLog(db_path=tmp_path / "events.sqlite3")
    rec = MetricsRecorder(log, environment="dev")
    acc = _account()
    acc.open_positions["binance:BTC/USDT"] = _make_position()
    rec.record(account=acc, last_prices={"binance:BTC/USDT": 100.0}, iteration=1)
    row = latest_metrics(log)
    assert row is not None
    assert row["n_positions"] == 1
    assert row["exposure_usd"] == 50.0  # 0.5 qty x 100 (latest price wins over entry)