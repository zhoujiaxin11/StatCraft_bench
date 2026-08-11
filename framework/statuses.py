"""
framework/statuses.py
=====================
Single source of truth for trial status classification.

v4 (吴-3): extracted from runner.py and aggregate.py to eliminate
diverging copies. Both modules MUST import from here.

Status categories
-----------------
- INFRA_STATUSES: trial did NOT measure the model. Dropped from ability
  estimate (usable=False). Examples: Docker not running, disk full, scorer bug.
- MODEL_FAIL_STATUSES: model was measured but failed to produce a valid
  answer. Counts as 0 in the ability estimate (usable=True).
- "ok": model was measured and produced a scoreable answer.

The `bucket()` function maps any status string to one of
{"ok", "infra", "model_fail", "unknown"}. Unknown statuses are treated
conservatively as infra (dropped) with a warning.
"""
from __future__ import annotations

import warnings

# -- Infrastructure failures (not the model's fault) --
INFRA_STATUSES = frozenset({
    "api_error",
    "executor_crash",
    "scorer_error",
    "docker_unavailable",
    "image_missing",
    "bad_io",
    "wall_timeout",
})

# -- Model failures (model was tested, failed to deliver) --
MODEL_FAIL_STATUSES = frozenset({
    "max_turns",
    "no_output",
    "bad_json",
    "no_progress",
    "stuck_loop",
})

# Union of both failure categories (legacy compat).
CRASH_STATUSES = INFRA_STATUSES | MODEL_FAIL_STATUSES

# All statuses the framework can produce.
ALL_KNOWN_STATUSES = CRASH_STATUSES | frozenset({"ok"})


def bucket(status: str | None) -> str:
    """Classify a trial status into a scoring bucket.

    Returns one of: "ok", "infra", "model_fail", "unknown".

    v4 (吴-3): unknown statuses now return "unknown" instead of silently
    falling through to "ok". Callers should treat "unknown" as infra
    (conservative: don't credit the model for an unrecognized state).
    """
    if status is None or status == "ok":
        return "ok"
    if status in INFRA_STATUSES:
        return "infra"
    if status in MODEL_FAIL_STATUSES:
        return "model_fail"
    warnings.warn(
        f"Unrecognized trial status {status!r} — treating as infra (dropped). "
        "Register it in framework/statuses.py if this is a new legitimate status.",
        stacklevel=2,
    )
    return "unknown"
