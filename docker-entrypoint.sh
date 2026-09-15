#!/usr/bin/env sh
# Default: run the core engine loop in paper mode. Override CMD/args to run
# other ai-trading-bot subcommands (fetch, show-config, ...).
set -e
exec ai-trading-bot run "$@"