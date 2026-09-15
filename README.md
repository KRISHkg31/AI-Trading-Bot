# AI Trading Bot

Autonomous, AI-powered multi-asset algorithmic trading bot (crypto, US equities, Indian equities),
built from the client BRD v1.0 (see `docs/`). Data → Intelligence → Signals → **Risk Engine** → Execution,
with capital protection as the design priority and paper/backtest gates before any live capital.

## Phase status

- [x] Phase 0 — Foundation & Data Layer *(in progress)*
- [ ] Phase 1 — Intelligence (Strategies & Signals)
- [ ] Phase 2 — Risk Engine (4 tiers)
- [ ] Phase 3 — Execution & Order Management
- [ ] Phase 4 — ML/AI Strategies
- [ ] Phase 5 — Observability & Ops
- [ ] Phase 6 — Validate & Go-Live

## Stack

Python 3.13, pandas/numpy, pydantic + pydantic-settings, PyYAML, ccxt (crypto), pytest/ruff.
Secrets via environment / `.env` only — never in code.

## Layout

```
src/ai_trading_bot/
  domain/      canonical, vendor-neutral data model (Bar, Tick, Instrument, Signal, …)
  adapters/    pluggable market-data & broker adapters (ccxt / Alpaca / Zerodha)
  config/      centralised, versioned config (dev / backtest / paper / live)
  strategy/    rule-based + ML strategy framework
  signal/      signal synthesis, filters, reason codes
  risk/        independent 4-tier risk gate between signals and execution
  execution/   order lifecycle, idempotency, reconciliation
  backtest/    cost-aware simulator, walk-forward, stress tests
  monitoring/  logging, alerts, heartbeat, dashboard
```

## Quick start

```bash
python -m venv .venv && source .venv/Scripts/activate
pip install -e ".[dev]"
cp .env.example .env   # add real keys only when going live; none required for dev
pytest
```