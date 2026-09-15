"""Signal generation helpers (REQ-SIG-*).

Full synthesis pipeline (min-confidence, expected-edge filter, anti-whipsaw cooldown,
reason codes) is implemented in Phase 1. The :class:`Signal` domain object already
carries the required elements (direction, confidence, size, horizon, reason).
"""