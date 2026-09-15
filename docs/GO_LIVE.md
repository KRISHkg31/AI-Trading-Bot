# GO-LIVE: Validate -> Paper -> Sign-off -> Graduated Live

Phase 6 roll-out gate per BRD. **Nothing below trades real money until every
prior step is green.** A live order is structurally refused everywhere it is not
explicitly enabled (see `ExecutionRouter` / `_LiveBrokerBase._require_live`).

## 1. Roll-out gates (in order)

1. **Backtest validation** — `ai-trading-bot validate SYMBOL`
   Runs walk-forward (out-of-sample) + 4 stress regimes on **stored** bars, then
   an acceptance gate (config `backtest.accept_*`). Exit 0 = PASS, 1 = FAIL.
2. **Paper period** — `execution.mode: paper` (default). The loop trades a
   simulated ledger against live public feeds; `ai-trading-bot monitor` shows
   health. Run at least N sessions; equity must track backtest expectations.
3. **Risk-review sign-off** — confirm the `RiskLimits` below match the account.
4. **Graduated live sizing** — start at a tiny fraction of capital, scale only
   while `monitor` + daily equity stay healthy and alerts stay quiet.

## 2. Market readiness (ground truth as of 2026-09-16)

| Market | Data adapter | Order path | Can it go live tomorrow? |
|---|---|---|---|
| Crypto (Binance spot) | ✅ implemented (public, no key) | ⚠️ structure only — `_place()` not yet wired | **Only candidate.** Needs key + `_place` + tiny size |
| US equities (Alpaca) | ❌ stub (`fetch_bars` raises) | ❌ stub | No — adapter must be implemented first |
| India equities (Zerodha) | ❌ stub | ❌ stub | No — adapter must be implemented first |

**Only Binance spot crypto is reachable for a first live session.**

## 3. Current validation state (real BTC data)

`validate BTC/USDT` on 1,030 real Binance 5m bars **FAILED** the acceptance gate:
walk-forward (out-of-sample) return **-1.03%**, profit factor **0.62**. ML
retrain recorded v2 candidates for both models but **promoted neither** (gate
held). The bot is not validated to trade the default strategies on BTC yet.

## 4. API keys — `.env`

| Key | Market | Purpose | Where to get it |
|---|---|---|---|
| `BINANCE_API_KEY` | Crypto | Live spot orders | Binance → API Management → Create API → enable **Enable Spot** |
| `BINANCE_API_SECRET` | Crypto | Sign orders (with key) | Shown once at creation |
| `BINANCE_TESTNET=TRUE` | Crypto (optional) | Sandbox testnet first | Binance Spot Testnet |
| `ALPACA_API_KEY` / `ALPACA_API_SECRET` | US eq (paper or live) | OAuth-style live broker + feed | app.alpaca.markets. Free paper; real funds needs funding + KYC |
| `ZERODHA_API_KEY` / `ZERODHA_API_SECRET` / `ZERODHA_ACCESS_TOKEN` | India | Kite Connect; access token refreshed **daily** | console.zerodha.com → Kite Connect (KYC required) |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | Alerts | Dead-man's switch + risk alerts reach you | @BotFather creates bot; `/start` the bot to get chat id |
| `SLACK_WEBHOOK_URL` | Alerts | Alternate alert channel | Slack → Incoming Webhooks |
| `ANTHROPIC_API_KEY` | Optional | Future LLM signals | platform.anthropic.com |

`get_credentials()` already reads every key above — no code change needed to use
them. **No key appears in pinning/commits** (`.env` only).

## 5. First-live-day procedure (crypto, when the gate passes)

1. Set `BINANCE_TESTNET=TRUE`, run paper-parallel live with testnet keys; watch
   fills reconcile (`monitor` + EventLog `order` events) for a full session.
2. Review `RiskLimits` against position size: default 1% risk/trade, 3% daily
   stop → a $10k account risks at most ~$300/day. Make sure that is acceptable.
3. Flip `execution.mode: live` OFF testnet, keep `strategy.active` to one
   instrument, day total below daily-loss cap, and a human watching `monitor`.
4. Keep `ExecutionRouter`'s guard: live path only resolves with mode=live AND a
   supplied client — the gate cannot be skipped by accident.