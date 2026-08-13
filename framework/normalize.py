"""
framework/normalize.py
======================
Value normalization utilities for Factoid + Tolerance evaluation.

Design source
-------------
- DABStep (Adyen/HuggingFace 2025): flexible scoring with type-specific
  normalization and tolerance
- Preserves 159's outcome_grader_v2 numeric comparison philosophy
"""
from __future__ import annotations

import math
import re
import unicodedata
from typing import Any


# ----------------------------------------------------------------------
# Numeric normalization
# ----------------------------------------------------------------------
_NUMBER_STRIP_RE = re.compile(r"[,\s$¥￥%]")


def parse_number(x: Any) -> float | None:
    """Best-effort conversion to float.

    Handles common surface forms: "2,547.83", "$1000", "12%", "24.5 months".
    Returns None if not parseable as a number.
    """
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if not isinstance(x, str):
        return None
    s = x.strip()
    if not s:
        return None
    # Strip trailing unit words (e.g. "24.5 months") — keep the leading number
    m = re.match(r"^([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", _NUMBER_STRIP_RE.sub("", s))
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def within_tolerance(
    agent_val: Any,
    gt_val: float,
    *,
    tol: float | None = None,
    abs_tol: float | None = None,
) -> bool:
    """Adaptive numeric tolerance check.

    - tol: relative tolerance (fraction, e.g. 0.05 = 5%)
    - abs_tol: absolute tolerance
    - At least one must be provided.
    """
    val = parse_number(agent_val)
    if val is None or not math.isfinite(val):
        return False
    diff = abs(val - gt_val)
    if abs_tol is not None and diff <= abs_tol:
        return True
    if tol is not None:
        denom = max(abs(gt_val), 1e-12)
        if (diff / denom) <= tol:
            return True
    return False


# ----------------------------------------------------------------------
# String normalization
# ----------------------------------------------------------------------
_PUNCT_RE = re.compile(r"[^\w\u4e00-\u9fff]+", re.UNICODE)


def normalize_string(s: Any) -> str:
    """Lowercase, strip punctuation and extra whitespace, NFKC-normalize."""
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.lower().strip()
    s = _PUNCT_RE.sub(" ", s)
    return " ".join(s.split())


def string_similarity(a: str, b: str) -> float:
    """Character-level similarity in [0, 1]. Uses SequenceMatcher."""
    from difflib import SequenceMatcher

    return SequenceMatcher(None, a, b).ratio()


def match_string(agent_val: Any, gt_val: str, *, threshold: float = 0.95) -> bool:
    a = normalize_string(agent_val)
    g = normalize_string(gt_val)
    if not a or not g:
        return False
    if a == g:
        return True
    return string_similarity(a, g) >= threshold


def match_string_in_set(agent_val: Any, accept_set: list[str]) -> bool:
    """Return True if agent_val matches any accepted answer.

    Used for questions with multiple valid answers (e.g. "which province is
    the cleanest reform switch point?" — several provinces are acceptable).
    """
    return any(match_string(agent_val, ok) for ok in accept_set)


# ----------------------------------------------------------------------
# List / set normalization
# ----------------------------------------------------------------------
def match_list_ordered(agent_val: Any, gt_list: list) -> bool:
    if not isinstance(agent_val, list) or len(agent_val) != len(gt_list):
        return False
    return all(match_string(a, g) for a, g in zip(agent_val, gt_list))


def match_list_as_set(agent_val: Any, gt_list: list) -> bool:
    if not isinstance(agent_val, list):
        return False
    if len(agent_val) != len(gt_list):
        return False
    a_norm = {normalize_string(x) for x in agent_val}
    g_norm = {normalize_string(x) for x in gt_list}
    return a_norm == g_norm


# ----------------------------------------------------------------------
# Enum matching
# ----------------------------------------------------------------------
def match_enum(agent_val: Any, allowed: list[str]) -> bool:
    a = normalize_string(agent_val)
    return any(a == normalize_string(x) for x in allowed)
