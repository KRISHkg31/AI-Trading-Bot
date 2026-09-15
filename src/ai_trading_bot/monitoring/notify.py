"""Alert dispatch to chat channels (REQ-MON-02).

Only two channels are wired today — Telegram and Slack, both already present as
optional fields on :class:`~ai_trading_bot.config.Credentials`. A notifier is
built *from* credentials and the ``monitoring.alert_channels`` list; channels
with no keys are logged, never raised on, so a missing token in dev/backtest is
harmless. Message-level dedup stops a repeated signal from spamming chat.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import requests

from ai_trading_bot.config import Credentials

logger = logging.getLogger("ai_trading_bot.monitoring.alerts")

#: Fallback when ``requests`` is unavailable inside this module's sandbox.
_HTTP = requests


@dataclass
class Notifier:
    credentials: Credentials
    channels: list[str] = field(default_factory=list)
    post: object | None = None  # injectable: post(url, json=...) -> response

    #: Dedup window seconds per (channel, title); identical alerts within the
    #: window are only logged.
    dedup_window_seconds: float = 60.0
    _last_seen: dict[tuple[str, str], float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _sent: dict[tuple[str, str], str] = field(default_factory=dict)

    def _post(self, url: str, *, json: dict) -> bool:
        post = self.post or _HTTP.post
        try:
            resp = post(url, json=json, timeout=10)
            return resp.status_code < 300
        except Exception:
            logger.exception("alert post failed to %s", url)
            return False

    def _send_telegram(self, body: str) -> str:
        token = self.credentials.telegram_bot_token
        chat_id = self.credentials.telegram_chat_id
        if not token or not chat_id:
            return "not-configured"
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        return "sent" if self._post(url, json={"chat_id": chat_id, "text": body}) else "failed"

    def _send_slack(self, body: str) -> str:
        url = self.credentials.slack_webhook_url
        if not url:
            return "not-configured"
        return "sent" if self._post(url, json={"text": body}) else "failed"

    def _dispatch(self, title: str, body: str) -> list[str]:
        that = f"[{title}] {body}"
        out: list[str] = []
        for channel in self.channels:
            key = (channel, that)
            now = time.monotonic()
            with self._lock:
                last = self._last_seen.get(key)
                if last is not None and now - last < self.dedup_window_seconds:
                    logger.warning("alert suppressed (dedup): %s", that)
                    continue
                self._last_seen[key] = now
            status = self._send_telegram(that) if channel == "telegram" else \
                self._send_slack(that) if channel == "slack" else "unknown-channel"
            logger.warning("alert queued to %s -> %s: %s", channel, status, that)
            out.append(f"{channel}:{status}")
        return out

    def notify(self, title: str, body: str) -> list[str]:
        """Route an alert to every configured channel; log-only when unset."""
        if not self.channels:
            logger.warning("alert (no channels configured): %s", f"[{title}] {body}")
        return self._dispatch(title, body)


__all__ = ["Notifier"]