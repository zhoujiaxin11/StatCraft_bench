"""
framework/scorer.py
===================
Unified Factoid + weighted partial-credit scorer for data analysis bench.

Design sources
--------------
- DABStep flexible tolerance protocol (numeric adaptive tol, string fuzzy)
- Weighted partial-credit inherited from 159's outcome_grader_v2.py
- Capability-level attribution (concept adapted from AgenticDataBench's skill tree)
- Graded scoring bands (multi-tier tolerance) instead of hard thresholds

Key features
------------
1. Single scorer serves ALL tasks (加一题不加代码)
2. Weighted partial credit (0.0 – 1.0 per field, weighted sum, normalized)
3. Capability-level breakdown (per-capability scores for radar chart)
4. Multiple scoring rule types: exact / graded / enum / list / accept_set / custom
5. Deterministic — no LLM judge, fully reproducible

Usage
-----
    from framework.scorer import score_task

    result = score_task(
        answer_path="path/to/agent/outcome.json",
        groundtruth_path="path/to/groundtruth/public/159_gaokao_reform.json",
    )
    print(result["total_score"])       # 0.72
    print(result["per_capability"])          # {"did_estimation": 0.85, ...}
    print(result["per_field"])          # {"part2.old_ratio": 1.0, ...}
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from framework.normalize import (
    match_enum,
    match_list_as_set,
    match_list_ordered,
    match_string,
    match_string_in_set,
    parse_number,
    within_tolerance,
)


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------
@dataclass
class FieldResult:
    field: str
    score: float                 # 0.0 – 1.0
    weight: float
    capabilities: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class ScoringResult:
    total_score: float
    total_weight: float
    earned_weight: float
    per_field: dict[str, float]
    per_capability: dict[str, float]
    details: list[FieldResult]

    def to_dict(self) -> dict:
        return {
            "total_score": round(self.total_score, 4),
            "total_weight": round(self.total_weight, 2),
            "earned_weight": round(self.earned_weight, 2),
            "per_field": {k: round(v, 4) for k, v in self.per_field.items()},
            "per_capability": {k: round(v, 4) for k, v in self.per_capability.items()},
            "details": [
                {
                    "field": d.field,
                    "score": round(d.score, 4),
                    "weight": d.weight,
                    "capabilities": d.capabilities,
                    "reason": d.reason,
                }
                for d in self.details
            ],
        }


# ---------------------------------------------------------------------------
# Rule evaluators
# ---------------------------------------------------------------------------
def _get_by_path(obj: Any, dotted: str) -> Any:
    """Fetch a nested value by dotted path, e.g. 'part2.old_ratio'."""
    cur = obj
    for key in dotted.split("."):
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            return None
        if cur is None:
            return None
    return cur


def _score_field(agent_val: Any, gt_val: Any, rule: dict) -> tuple[float, str]:
    """Return (score in [0,1], reason)."""
    rtype = rule.get("type", "exact")

    if rtype == "exact_number":
        val = parse_number(agent_val)
        gt = parse_number(gt_val)
        if val is None or gt is None:
            return 0.0, f"parse fail (agent={agent_val!r}, gt={gt_val!r})"
        return (1.0, "ok") if val == gt else (0.0, f"mismatch: {val} vs {gt}")

    if rtype == "exact_string":
        return (1.0, "ok") if match_string(agent_val, str(gt_val)) else (0.0, "string mismatch")

    if rtype == "graded":
        # rule = {"type": "graded", "levels": [{"tol": 0.02, "score": 1.0}, ...]}
        # Levels are tried in order; first match wins.
        # Supports "tol" (relative) or "abs" (absolute).
        # v3: gt_val must be parseable as a number; otherwise report
        # rather than crash on `float(None)` / `float("abc")`.
        gt_num = parse_number(gt_val)
        if gt_num is None:
            return 0.0, f"graded: gt not numeric (gt={gt_val!r})"
        for level in rule.get("levels", []):
            tol = level.get("tol")
            abs_tol = level.get("abs")
            if within_tolerance(agent_val, gt_num, tol=tol, abs_tol=abs_tol):
                return float(level.get("score", 1.0)), f"graded pass (tol={tol}, abs={abs_tol})"
        return 0.0, "all levels miss"

    if rtype == "enum":
        allowed = rule.get("allowed", [])
        return (1.0, "ok") if match_enum(agent_val, allowed) else (0.0, "not in enum")

    if rtype == "accept_set":
        # For questions with multiple correct answers, e.g. province selection.
        accept = rule.get("accept", [])
        return (1.0, "ok") if match_string_in_set(agent_val, accept) else (0.0, "not in accept set")

    if rtype == "list_ordered":
        return (1.0, "ok") if match_list_ordered(agent_val, gt_val) else (0.0, "ordered list mismatch")

    if rtype == "list_set":
        return (1.0, "ok") if match_list_as_set(agent_val, gt_val) else (0.0, "set mismatch")

    if rtype == "object_keys_match":
        # For dict answers where all keys must have matching values.
        # gt_val is the reference dict; agent_val must match every key.
        #
        # v2 (#25): dispatch by ground-truth value type.
        #   * numeric gt  → within_tolerance (uses rule["levels"] / abs / tol)
        #   * string gt   → match_string as before
        # v3:
        #   * bool gt is NOT numeric — Python's isinstance(True, int) is True,
        #     which pre-v3 sent it into the numeric branch and crashed on
        #     float(None) (parse_number returns None for bool).
        #   * string gt is numeric ONLY when parse_number consumes the whole
        #     string (no unit suffix). "24.5 months" is text, not a number,
        #     even though parse_number would strip the unit.
        if not isinstance(agent_val, dict) or not isinstance(gt_val, dict):
            return 0.0, "not a dict"
        levels = rule.get("levels")
        rule_tol = rule.get("tol")
        rule_abs = rule.get("abs")
        per_key_scores: list[float] = []
        for k, v_gt in gt_val.items():
            v_agent = agent_val.get(k)
            # v3: strict numeric detection.
            is_numeric_gt = (
                isinstance(v_gt, (int, float)) and not isinstance(v_gt, bool)
            )
            gt_num: float | None = float(v_gt) if is_numeric_gt else None
            if not is_numeric_gt and isinstance(v_gt, str):
                # A string is numeric only when it parses cleanly with no
                # residual text — otherwise treat as text.
                s = v_gt.strip()
                if s:
                    try:
                        gt_num = float(s.replace(",", ""))
                        is_numeric_gt = True
                    except ValueError:
                        gt_num = None
                        is_numeric_gt = False
            if is_numeric_gt and gt_num is not None:
                # Numeric ground truth branch.
                if levels:
                    matched_score = 0.0
                    for level in levels:
                        tol = level.get("tol")
                        abs_tol = level.get("abs")
                        if within_tolerance(v_agent, gt_num, tol=tol, abs_tol=abs_tol):
                            matched_score = float(level.get("score", 1.0))
                            break
                    per_key_scores.append(matched_score)
                elif rule_tol is not None or rule_abs is not None:
                    ok = within_tolerance(v_agent, gt_num, tol=rule_tol, abs_tol=rule_abs)
                    per_key_scores.append(1.0 if ok else 0.0)
                else:
                    # Exact numeric equality (no tolerance configured).
                    a_num = parse_number(v_agent)
                    per_key_scores.append(
                        1.0 if a_num is not None and a_num == gt_num else 0.0
                    )
            elif isinstance(v_gt, bool):
                # v3: bool → boolean equality, not string similarity.
                per_key_scores.append(1.0 if v_agent == v_gt else 0.0)
            else:
                # String / mixed — fall through to the pre-v2 SequenceMatcher path.
                per_key_scores.append(1.0 if match_string(v_agent, str(v_gt)) else 0.0)

        if not per_key_scores:
            return 0.0, "empty gt dict"
        avg = sum(per_key_scores) / len(per_key_scores)
        matched_full = sum(1 for s in per_key_scores if s >= 0.999)
        if matched_full == len(per_key_scores):
            return 1.0, "all keys match"
        return round(avg, 4), f"partial: avg {matched_full}/{len(per_key_scores)} full-credit"

    if rtype == "no_leakage_keywords":
        # Check that agent's answer OR its code does not use forbidden features.
        # rule = {"type": "no_leakage_keywords", "forbidden": [...]}
        # agent_val expected to be a concatenated string (feature list or code).
        # An ABSENT field is NOT evidence of cleanliness — it means the model
        # never produced the feature list at all, which must score 0, not 1.
        # Without this guard str(None)=="none" contains no forbidden token and
        # silently earns full credit for a missing answer.
        if agent_val is None or (isinstance(agent_val, str) and not agent_val.strip()) \
                or agent_val in ([], {}):
            return 0.0, "field absent — cannot verify no-leakage"
        forbidden = rule.get("forbidden", [])
        s = str(agent_val).lower()
        # v4: use word-boundary regex for ASCII tokens to avoid false
        # positives (e.g. "china_sales_na" matching forbidden "na"). CJK tokens
        # lack word boundaries, so retain substring matching for them.
        import re as _re
        def _is_cjk(text: str) -> bool:
            return any('一' <= c <= '鿿' for c in text)
        hits = []
        for w in forbidden:
            w_low = w.lower()
            if _is_cjk(w_low):
                if w_low in s:
                    hits.append(w)
            else:
                if _re.search(r'\b' + _re.escape(w_low) + r'\b', s):
                    hits.append(w)
        return (1.0, "no leakage") if not hits else (0.0, f"leakage: {hits}")

    if rtype == "contains":
        # Positive keyword-presence check for free-text fields (e.g. anomaly_notes).
        # Unlike exact_string (which needs SequenceMatcher >= 0.95 and fails on
        # long prose), this passes when the answer MENTIONS the required concepts.
        #   rule = {"type": "contains", "any": ["1989", "异常", "剔除"]}          # any → full credit
        #   rule = {"type": "contains", "all": ["剔除", "缺失"]}                  # all required
        #   rule = {"type": "contains", "all": [...], "any": [...],
        #           "partial": true}                                             # graded
        # Matching is case-insensitive and NFKC-normalized via normalize_string.
        from framework.normalize import normalize_string
        import re as _re
        s = normalize_string(str(agent_val)) if agent_val is not None else ""
        # v2 (#26): normalize_string turns "N/A" into "n a" (slash → space),
        # but the answer text often carries the naked "NA" which normalizes to
        # "na". Build a companion whitespace-stripped view so tokens that lost
        # their separator can still match. Applied on BOTH sides so a keyword
        # written "N / A" and an answer written "na" both round-trip.
        s_squashed = "".join(s.split())
        need_all = [normalize_string(str(w)) for w in rule.get("all", [])]
        need_any = [normalize_string(str(w)) for w in rule.get("any", [])]
        if not need_all and not need_any:
            return 0.0, "contains rule has neither 'all' nor 'any'"

        def _hit(term: str) -> bool:
            if not term:
                return False
            if len(term) < 3 and term.isascii():
                # Short ASCII tokens ("na") must stand ALONE — never substring-
                # match inside longer words ("china", "national" would otherwise
                # earn full credit for unrelated prose). Boundaries are ASCII
                # alphanumerics only, so CJK neighbors still count as boundaries:
                # "将NA替换为0" must keep matching.
                _pat = _re.compile(
                    r"(?<![A-Za-z0-9])" + _re.escape(term) + r"(?![A-Za-z0-9])"
                )
                return bool(_pat.search(s)) or bool(_pat.search(s_squashed))
            if term in s:
                return True
            # v2 (#26): squash-space fallback for tokens like "n a" ↔ "na".
            squashed_term = "".join(term.split())
            return bool(squashed_term) and squashed_term in s_squashed

        all_hits = [w for w in need_all if _hit(w)]
        any_hit = any(_hit(w) for w in need_any) if need_any else True

        all_ok = len(all_hits) == len(need_all)
        if all_ok and any_hit:
            return 1.0, "contains ok"

        if rule.get("partial"):
            # Graded: fraction of 'all' terms present, gated by the 'any' clause.
            denom = len(need_all) if need_all else 1
            frac = (len(all_hits) / denom) if need_all else 0.0
            if need_any and not any_hit:
                return 0.0, f"contains: no 'any' term matched {rule.get('any')}"
            return round(frac, 4), f"contains partial: {len(all_hits)}/{denom} all-terms"

        missing_all = [w for w in need_all if not _hit(w)]
        return 0.0, f"contains miss (missing_all={missing_all}, any_hit={any_hit})"

    if rtype == "min_threshold":
        # For "higher is better" metrics (R², coverage, accuracy).
        # rule = {"type": "min_threshold", "levels": [{"min": 0.85, "score": 1.0}, ...]}
        # Agent value >= level["min"] → that level's score. Levels tried top-down.
        val = parse_number(agent_val)
        if val is None:
            return 0.0, f"parse fail (agent={agent_val!r})"
        for level in rule.get("levels", []):
            if val >= float(level["min"]):
                return float(level.get("score", 1.0)), f"min_threshold pass (val={val} >= {level['min']})"
        return 0.0, f"below all thresholds (val={val})"

    if rtype == "max_threshold":
        # For "lower is better" metrics (RMSE, MAE, error rate).
        # rule = {"type": "max_threshold", "levels": [{"max": 4000, "score": 1.0}, ...]}
        # Agent value <= level["max"] → that level's score. Levels tried top-down.
        val = parse_number(agent_val)
        if val is None:
            return 0.0, f"parse fail (agent={agent_val!r})"
        for level in rule.get("levels", []):
            if val <= float(level["max"]):
                return float(level.get("score", 1.0)), f"max_threshold pass (val={val} <= {level['max']})"
        return 0.0, f"above all thresholds (val={val})"

    if rtype == "bool":
        # Boolean field matching: normalizes both sides to bool.
        # gt can be true/false/"true"/"false"/1/0.
        def _to_bool(x):
            if isinstance(x, bool):
                return x
            if isinstance(x, (int, float)):
                return bool(x)
            if isinstance(x, str):
                return x.strip().lower() in ("true", "yes", "1", "是", "一致")
            return None
        a = _to_bool(agent_val)
        g = _to_bool(gt_val)
        if a is None or g is None:
            return 0.0, f"bool parse fail (agent={agent_val!r}, gt={gt_val!r})"
        return (1.0, "ok") if a == g else (0.0, f"bool mismatch: {a} vs {g}")

    if rtype == "presence":
        # Simply checks the field is present and non-empty.
        if agent_val in (None, "", [], {}):
            return 0.0, "empty"
        return 1.0, "present"

    return 0.0, f"unknown rule type: {rtype}"


# ---------------------------------------------------------------------------
# Known rule types (used by validate_groundtruth_consistency for typo check)
# ---------------------------------------------------------------------------
_KNOWN_RULE_TYPES = frozenset({
    "exact_number", "exact_string", "graded", "enum", "accept_set",
    "list_ordered", "list_set", "object_keys_match", "no_leakage_keywords",
    "contains", "min_threshold", "max_threshold", "bool", "presence",
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def score_task(
    *,
    answer_path: str | Path,
    groundtruth_path: str | Path,
) -> ScoringResult:
    """Score a single task run.

    Reads:
      - answer_path:      the agent's outcome.json
      - groundtruth_path: the task's groundtruth JSON with values/weights/scoring
    """
    answer = _load_json(answer_path)
    gt = _load_json(groundtruth_path)

    weights: dict[str, float] = gt.get("weights", {})
    values: dict[str, Any] = gt.get("values", {})
    scoring: dict[str, dict] = gt.get("scoring", {})
    # Support both new field name (capability_map) and legacy (skill_map)
    capability_map: dict[str, list[str]] = gt.get("capability_map", gt.get("skill_map", {}))

    details: list[FieldResult] = []
    total_weight = 0.0
    earned_weight = 0.0

    for field_key, weight in weights.items():
        weight = float(weight)
        if weight <= 0:
            continue

        # v3.3: an undeclared scoring rule used to default silently to
        # exact_string, masking authoring mistakes as "wrong answers". Now it is
        # a loud CONFIG_ERROR zero so the pre-flight validator stays meaningful.
        rule = scoring.get(field_key)
        if rule is None or not isinstance(rule, dict):
            details.append(
                FieldResult(
                    field=field_key,
                    score=0.0,
                    weight=weight,
                    capabilities=capability_map.get(field_key, []),
                    reason=f"[CONFIG_ERROR] no 'scoring' entry declared for {field_key!r}",
                )
            )
            total_weight += weight
            continue

        gt_val = rule.get("gt", _get_by_path(values, field_key))
        agent_val = _get_by_path(answer, field_key)

        s, reason = _score_field(agent_val, gt_val, rule)

        capabilities = capability_map.get(field_key, [])
        details.append(
            FieldResult(
                field=field_key,
                score=s,
                weight=weight,
                capabilities=capabilities,
                reason=reason,
            )
        )
        total_weight += weight
        earned_weight += s * weight

    per_field = {d.field: d.score for d in details}
    per_capability = _aggregate_by_capability(details)
    total = earned_weight / total_weight if total_weight > 0 else 0.0

    return ScoringResult(
        total_score=total,
        total_weight=total_weight,
        earned_weight=earned_weight,
        per_field=per_field,
        per_capability=per_capability,
        details=details,
    )


def _aggregate_by_capability(details: list[FieldResult]) -> dict[str, float]:
    """Compute per-capability weighted average score."""
    agg: dict[str, list[tuple[float, float]]] = {}
    for d in details:
        for cap in d.capabilities:
            agg.setdefault(cap, []).append((d.score, d.weight))
    result = {}
    for cap, pairs in agg.items():
        w_sum = sum(w for _, w in pairs)
        if w_sum == 0:
            continue
        result[cap] = sum(s * w for s, w in pairs) / w_sum
    return result


def _load_json(path: str | Path) -> dict:
    """v3.3 : load JSON tolerantly. Python's stdlib accepts bare
    ``NaN`` / ``Infinity`` / ``-Infinity`` literals and turns them into floats,
    whereupon ``nan != anything`` makes EVERY numeric comparison fail and zeros
    the whole trial even though its other fields were correct. Mapping these
    non-standard constants to None lets the offending fields be scored as simply
    missing, while genuinely malformed JSON (broken structure) still raises so
    the runner keeps reporting ``bad_json`` correctly."""
    def _nullify_nonstandard_constant(_c: str):  # noqa: ARG001
        return None
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    return json.loads(raw, parse_constant=_nullify_nonstandard_constant)


# ---------------------------------------------------------------------------
# Groundtruth consistency check (v2 #30, )
# ---------------------------------------------------------------------------
def validate_groundtruth_consistency(groundtruth: dict) -> list[str]:
    """Cross-check ``values`` against inlined ``scoring[k].gt``.

    `values` is a legacy fallback: scorer reads ``scoring[k].gt`` first and
    only falls back to ``values`` when the inline gt is missing. Because
    nothing enforces agreement, the two blocks can silently drift apart —
    typo-prone and misleading to anyone auditing the groundtruth.

    v3 : also catches ``scoring[k].type`` typos (e.g. ``exect_number``,
    ``bools``, ``containss``) — these would silently score 0 at runtime and
    look like model failure.

    v4 : adds weights↔scoring key-set cross-validation and mandatory
    field checks (levels for graded/threshold rules, gt for exact_* rules).
    Prevents silent score-zeroing caused by authoring mistakes.

    Returns a list of human-readable mismatch messages; empty list == clean.
    Called from CI / oracle-check so that a bad groundtruth fails loudly
    instead of turning into silent scoring quirks.
    """
    problems: list[str] = []
    values = groundtruth.get("values", {}) or {}
    scoring = groundtruth.get("scoring", {}) or {}
    weights = groundtruth.get("weights", {}) or {}

    # v4: weights ↔ scoring key-set cross-validation.
    scoring_keys = set(scoring.keys())
    weights_keys = set(weights.keys())
    for k in weights_keys - scoring_keys:
        problems.append(
            f"weights[{k!r}] has weight but no scoring rule — "
            "this field will silently score 0 (no rule to evaluate)."
        )

    # v4: per-rule mandatory field checks.
    _NEEDS_LEVELS = {"graded", "min_threshold", "max_threshold"}
    _NEEDS_GT = {"exact_number", "exact_string", "list_set", "list_ordered",
                 "object_keys_match", "bool"}

    for field_key, rule in scoring.items():
        if not isinstance(rule, dict):
            continue
        # v3: rule_type typo check.
        rtype = rule.get("type", "exact_string")
        if rtype not in _KNOWN_RULE_TYPES:
            problems.append(
                f"scoring[{field_key!r}].type={rtype!r} is not a known rule type "
                f"(known: {sorted(_KNOWN_RULE_TYPES)})"
            )

        # v4: levels mandatory for graded/threshold rules.
        if rtype in _NEEDS_LEVELS:
            levels = rule.get("levels")
            if not levels:
                problems.append(
                    f"scoring[{field_key!r}] type={rtype!r} requires 'levels' "
                    "but none found — this field will always score 0."
                )
            elif isinstance(levels, list):
                for i, lv in enumerate(levels):
                    if rtype == "min_threshold" and "min" not in lv:
                        problems.append(
                            f"scoring[{field_key!r}].levels[{i}] (min_threshold) "
                            "is missing 'min' key — will raise KeyError at runtime."
                        )
                    if rtype == "max_threshold" and "max" not in lv:
                        problems.append(
                            f"scoring[{field_key!r}].levels[{i}] (max_threshold) "
                            "is missing 'max' key — will raise KeyError at runtime."
                        )

        # v4: gt mandatory for exact/bool/list rules.
        if rtype in _NEEDS_GT and "gt" not in rule:
            problems.append(
                f"scoring[{field_key!r}] type={rtype!r} requires 'gt' "
                "but none found — this field cannot be evaluated."
            )

        # v3.3: enum / accept_set must declare their candidate lists;
        # an omitted list previously turned every answer into a silent 0.
        if rtype == "enum" and not rule.get("allowed"):
            problems.append(
                f"scoring[{field_key!r}] type='enum' requires a non-empty 'allowed'."
            )
        if rtype == "accept_set" and not rule.get("accept"):
            problems.append(
                f"scoring[{field_key!r}] type='accept_set' requires a non-empty "
                "'accept' list."
            )

        # v3.3: threshold-band monotonicity — catch inverted reward
        # orderings that would silently flip who earns which tier.
        _lvl_seq = rule.get("levels")
        if isinstance(_lvl_seq, list) and len(_lvl_seq) >= 2:
            def _floats(key: str) -> list[float]:
                out: list[float] = []
                for _lv in _lvl_seq:
                    try:
                        out.append(float(_lv.get(key)))
                    except (TypeError, ValueError):
                        return []  # incomplete → skip the assertion, don't crash here.
                return out

            if rtype == "min_threshold":
                ms = _floats("min")
                if ms and ms != sorted(ms, reverse=True):
                    problems.append(
                        f"scoring[{field_key!r}] type=min_threshold levels not descending"
                        f" by 'min' ({ms}) — top tiers must carry the HIGHEST bars first."
                    )
            elif rtype == "max_threshold":
                xs = _floats("max")
                if xs and xs != sorted(xs):
                    problems.append(
                        f"scoring[{field_key!r}] type=max_threshold levels not ascending"
                        f" by 'max' ({xs}) — tightest budget must come FIRST."
                    )
            elif rtype == "graded":
                scs = _floats("score")
                if scs and scs != sorted(scs, reverse=True):
                    problems.append(
                        f"scoring[{field_key!r}] type=graded award ordering inverted ({scs})"
                        " — earlier (tighter-tol) levels should score ≥ later ones."
                    )

        # Original v3 check: values vs scoring.gt consistency.
        if "gt" not in rule:
            continue
        v = _get_by_path(values, field_key)
        if v is None:
            continue
        gt = rule["gt"]
        if v != gt:
            # Tolerant on floats: 0.1 == 0.10 for our purposes.
            try:
                if isinstance(v, (int, float)) and isinstance(gt, (int, float)):
                    if float(v) == float(gt):
                        continue
            except (TypeError, ValueError):
                pass
            problems.append(
                f"values[{field_key!r}]={v!r} but scoring[{field_key!r}].gt={gt!r}"
            )
    return problems


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score a task run")
    parser.add_argument("--answer", required=True, help="Agent outcome.json path")
    parser.add_argument("--groundtruth", required=True, help="Groundtruth JSON path")
    parser.add_argument("--output", default=None, help="Write result JSON to this path")
    args = parser.parse_args()

    result = score_task(answer_path=args.answer, groundtruth_path=args.groundtruth)
    payload = result.to_dict()

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
