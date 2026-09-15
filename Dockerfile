# 24/7 hosting for the bot core loop (BRD Section 16). Kept small and pin-free
# for portability; pin images/tags before any live deployment.
FROM python:3.13-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AIBOT_ENVIRONMENT=paper

# System deps for pandas/numpy wheels (slim image lacks compilers).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config

RUN pip install --no-cache-dir .

# Data/log volumes are expected to be mounted in production.
VOLUME ["/app/data", "/app/logs"]

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]