"""Structured logging foundation (REQ-MON-03, REQ-NFR-14).

Logs go to stdout + a rotating file under the configured ``log_dir``. All records
carry ``env`` and ``utc_ts`` fields so replay and audit stay consistent.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from ai_trading_bot.config import AppConfig


def setup_logging(config: AppConfig) -> None:
    level = getattr(logging, config.monitoring.log_level.upper(), logging.INFO)
    root = logging.getLogger("ai_trading_bot")
    root.setLevel(level)
    root.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] env=%(env)s %(message)s"
    )

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(fmt)
    root.addHandler(stream)

    log_dir = Path(config.monitoring.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    err = RotatingFileHandler(
        log_dir / "trading_bot.log", maxBytes=10_000_000, backupCount=5, encoding="utf-8"
    )
    err.setFormatter(fmt)
    root.addHandler(err)

    # Attach env to every record via a filter
    env = config.environment
    _attach_env(root, env)


class _EnvFilter(logging.Filter):
    def __init__(self, env: str) -> None:
        super().__init__()
        self.env = env

    def filter(self, record: logging.LogRecord) -> bool:
        record.env = self.env
        return True


def _attach_env(logger: logging.Logger, env: str) -> None:
    f = _EnvFilter(env)
    for handler in logger.handlers:
        handler.addFilter(f)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"ai_trading_bot.{name}")