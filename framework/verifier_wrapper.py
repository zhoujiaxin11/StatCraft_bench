"""
framework/verifier_wrapper.py
=============================
Bridge between Harbor's pytest-based verifier (all-or-nothing per test)
and our weighted partial-credit scoring.

Design sources
--------------
- 159's outcome_grader_v2.py: parses pytest output + weight_map → weighted sum
- Harbor: `tests/` directory + pytest execution + CTRF-style report

Two evaluation paths
--------------------
Path A (primary): Factoid scorer via scorer.py
  - Agent produces outcome.json
  - scorer reads outcome.json vs groundtruth.json
  - Fine-grained weighted partial credit (0.0 – 1.0 per field)
  - No pytest required for main scoring

Path B (auxiliary): pytest for artifacts
  - Optional per-task tests/test_outputs.py checks file existence,
    plot validity, CSV structure — things not naturally factoid-shaped
  - This wrapper reads pytest results and merges with Path A

Final = alpha * factoid_score + (1 - alpha) * pytest_score
where alpha defaults to 0.9 (main weight on factoid, artifact checks small).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PytestResult:
    passed: int
    failed: int
    skipped: int
    per_test: list[dict]  # [{"name": ..., "status": "PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS"}]
    raw_output: str
    # v2 (#21, 魏-3a+吴-2): count of ERROR / XFAIL / XPASS outcomes.
    errored: int = 0
    # v2 (#22, 魏-3b+吴-3): True when the run produced NEITHER a PASSED nor a
    # FAILED line — i.e. nothing decidable to score against. Treated as
    # "no pytest layer", not as "the model earned 0".
    no_executable_tests: bool = False


def run_pytest(test_dir: Path, workspace: Path, timeout: int = 300) -> PytestResult:
    """Run pytest against tests/ in the task workspace, capture per-test results."""
    if not test_dir.exists():
        return PytestResult(0, 0, 0, [], "", errored=0, no_executable_tests=True)
    # v3.3 hotfix: resolve() to an ABSOLUTE path — we chdir into workspace below,
    # so a relative ``tasks/<id>/tests`` argument silently became unfindable there
    # ("ERROR: file or directory not found"), yielding zero collection every time.
    # That zero-collection tripped the王‑2 pytest_broken_no_collection branch and
    # capped combined_score at α*factoid (=≤0.90), making the >=0.99 self-check gate
    # structurally unreachable regardless of answer quality.
    cmd = [sys.executable, "-m", "pytest", "-v", "--no-header",
           "--rootdir", str(workspace),
           "-p", "no:cacheprovider",
           str((Path.cwd() / test_dir).resolve())]
    # v2 (#24, 魏-4): thread BENCH_WORKSPACE through to the pytest subprocess so
    # test fixtures that read it don't silently fall back to cwd (which happens
    # to equal workspace TODAY but is a fragile coincidence).
    env = {**os.environ, "BENCH_WORKSPACE": str(workspace)}
    proc = subprocess.run(
        cmd,
        cwd=str(workspace),
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    per_test = _parse_verbose(out)
    passed = sum(1 for t in per_test if t["status"] == "PASSED")
    failed = sum(1 for t in per_test if t["status"] == "FAILED")
    skipped = sum(1 for t in per_test if t["status"] == "SKIPPED")
    errored = sum(1 for t in per_test if t["status"] in {"ERROR", "XFAIL", "XPASS"})

    # v2 (#21): cross-check parsed count against pytest's own tally to detect
    # silent parse failures (e.g. after a pytest output-format upgrade).
    m = re.search(r"collected (\d+) item", out)
    if m:
        collected = int(m.group(1))
        if collected != len(per_test):
            warnings.warn(
                f"pytest reported {collected} collected but parser found "
                f"{len(per_test)} lines — output format may have drifted.",
                stacklevel=2,
            )

    no_executable = (passed == 0 and failed == 0)
    return PytestResult(
        passed=passed,
        failed=failed,
        skipped=skipped,
        per_test=per_test,
        raw_output=out,
        errored=errored,
        no_executable_tests=no_executable,
    )


# v2 (#21): accept ERROR / XFAIL / XPASS in addition to PASSED / FAILED / SKIPPED;
# non-greedy on the name so parametrized ids containing spaces
# (e.g. "test_years[2024 2025]") aren't truncated.
_STATUS_TOKEN = r"PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS"
_PARSE_RE = re.compile(rf".*::(.+?)\s+({_STATUS_TOKEN})\b")


def _parse_verbose(stdout: str) -> list[dict]:
    """Parse `pytest -v` output; each status line becomes a record."""
    results = []
    for line in stdout.splitlines():
        m = _PARSE_RE.match(line)
        if m:
            results.append({"name": m.group(1).strip(), "status": m.group(2)})
    return results


def weighted_pytest_score(
    pytest_result: PytestResult,
    weight_map: dict[str, float],
) -> tuple[float, float, float, dict]:
    """Apply weight_map to per-test results; only PASSED earns credit.

    v2 (#23, 吴-4 + 魏-5): unregistered tests default to weight **0** and emit
    a warning. Pre-v2 the default was 1.0, which silently diluted intended
    weights whenever the author added a test without updating the map.

    v3 (吴-1): parametrized tests (nodeid ``test_foo[case1]``) are matched by
    stripping the ``[...]`` suffix and splitting the base weight evenly across
    all parametrized instances. Pre-v3 both instances silently fell to weight
    0 while the base name in the weight_map went un-consumed — a full family
    of tests could disappear from the denominator and the surviving factoid
    layer would return a satisfied-looking 1.0.

    Returns ``(score_ratio, earned, total, details)`` where ``details`` has:
        matched_names   : list[str]  # nodeids that consumed weight
        unmatched_names : list[str]  # nodeids with no weight assigned
        matched_weight_keys : list[str]  # weight_map keys that were used
        unused_weight_keys  : list[str]  # weight_map keys that matched nothing
    """
    # v3 (吴-1): group per_test by function-base name (nodeid without [params]).
    def _base(name: str) -> str:
        return name.split("[", 1)[0] if "[" in name else name

    base_counts: dict[str, int] = {}
    for t in pytest_result.per_test:
        b = _base(t["name"])
        base_counts[b] = base_counts.get(b, 0) + 1

    matched_names: list[str] = []
    unmatched_names: list[str] = []
    matched_weight_keys: set[str] = set()

    total = 0.0
    earned = 0.0
    for t in pytest_result.per_test:
        name = t["name"]
        base = _base(name)
        if name in weight_map:
            w = float(weight_map[name])
            matched_weight_keys.add(name)
            matched_names.append(name)
        elif base in weight_map and base_counts.get(base, 0) > 0:
            # v3 (吴-1): split base weight across parametrized instances.
            w = float(weight_map[base]) / base_counts[base]
            matched_weight_keys.add(base)
            matched_names.append(name)
        else:
            w = 0.0
            unmatched_names.append(name)
            warnings.warn(
                f"pytest test {name!r} is not in pytest_weight_map — "
                "defaulted to weight 0 (v2 policy). Add it to "
                "groundtruth.pytest_weight_map to score it.",
                stacklevel=2,
            )
        if w <= 0:
            continue
        total += w
        if t["status"] == "PASSED":
            earned += w

    unused = [k for k in weight_map.keys() if k not in matched_weight_keys]
    details = {
        "matched_names": matched_names,
        "unmatched_names": unmatched_names,
        "matched_weight_keys": sorted(matched_weight_keys),
        "unused_weight_keys": unused,
    }
    if total == 0:
        return 0.0, 0.0, 0.0, details
    return earned / total, earned, total, details


# ---------------------------------------------------------------------------
# Essential/unacceptable veto exposure
# ---------------------------------------------------------------------------
def evaluate_veto(*, score_json: dict | None = None, groundtruth: dict | None) -> dict:
    """Describe what the task's optional ``veto`` rule WOULD do.

    This NEVER mutates the single-run ``combined_score`` — by design the veto
    zeros whole tasks only at the aggregation layer (see groundtruth _comment).
    Here we surface, alongside every run's result JSON:

      * configured     : does the task declare a veto block?
      * essential / unacceptable : the field keys involved
      * valid          : validate_veto passed (essential/unacceptable refs exist)
                         — surfaced loudly instead of silently zeroing later runs
                         when someone typos a field name in groundtruth.json
      * would_zero_task: would apply_veto() drop THIS run to 0?

    Returns would_zero_task=None when no score_json supplied (config-only mode,
    e.g. authoring-time validation without a concrete trial).
    """
    status: dict[str, Any] = {
        "configured": bool(groundtruth and groundtruth.get("veto")),
        "applied_at": "aggregation_layer_only",
    }
    if not status["configured"]:
        status.update(essential=[], unacceptable=[], valid=True,
                      would_zero_task=False)
        return status

    veto_cfg = groundtruth.get("veto") or {}
    ess = list(veto_cfg.get("essential", []))
    unacc = list(veto_cfg.get("unacceptable", []))
    status["essential"] = ess
    status["unacceptable"] = unacc

    try:
        from framework.aggregate import apply_veto, validate_veto
        validate_veto(groundtruth)
        status["valid"] = True
    except ValueError as exc:
        # Bad config — record loudly rather than crash mid-eval. Aggregation
        # would otherwise trip silently on every seed; flag here so CI sees it.
        status["valid"] = False
        status["error"] = str(exc)
        status["would_zero_task"] = None
        return status

    if score_json is None:
        status["would_zero_task"] = None
        return status

    _, would_zero = apply_veto(score_json, groundtruth)
    status["would_zero_task"] = bool(would_zero)
    return status


# ---------------------------------------------------------------------------
# Combined scoring
# ---------------------------------------------------------------------------
def combined_score(
    *,
    factoid_result: dict,        # scorer.py output as dict
    pytest_result: PytestResult,
    pytest_weight_map: dict[str, float] | None = None,
    factoid_alpha: float = 0.9,
    groundtruth: dict | None = None,
) -> dict:
    """Blend factoid score with pytest artifact score.

    ``groundtruth`` (optional) is used ONLY to stamp a non-mutating
    ``veto_status`` descriptor onto the result — it never changes the numeric
    combined_score. Pass the task's full groundtruth dict so essential/
    unacceptable veto validity can be checked here instead of only at
    aggregation time.

    v2 (#22, 魏-3b + 吴-3): the decision "is there a pytest layer" is now made
    by looking for ≥1 PASSED-or-FAILED line, not by whether per_test is
    non-empty. All-SKIP or all-ERROR runs (== no decidable tests) treat the
    pytest layer as absent (effective_alpha = 1.0) instead of silently
    zeroing the model.

    v2 (#23): also splits `declared_alpha` (config) from `effective_alpha`
    (what actually got applied) so the reader can tell "α=1.0 because no
    pytest configured" from "α=0.9 as configured".

    v3 (吴-4): "pytest layer broken" (per_test collected but all ERROR/SKIP
    AND a weight_map exists) now scores pytest_score=0 with
    effective_alpha=declared_alpha, instead of falling back to α=1.0. Pre-v3,
    breaking conftest would silently boost the score.

    v3 (李-2): ``pytest_details`` now includes the per-test roster
    (``per_test``: list of {name, status}) and weight-matching diagnostics so
    reports and downstream tooling can see which tests earned/dropped weight.
    """
    factoid_score = factoid_result.get("total_score", 0.0)
    declared_alpha = factoid_alpha
    pytest_status: str
    weight_details: dict = {
        "matched_names": [],
        "unmatched_names": [],
        "matched_weight_keys": [],
        "unused_weight_keys": [],
    }

    has_weight_map = bool(pytest_weight_map)
    has_per_test = bool(pytest_result.per_test)
    has_executable = (
        has_per_test
        and not pytest_result.no_executable_tests
        and has_weight_map
    )
    if has_executable:
        pytest_score, earned, total, weight_details = weighted_pytest_score(
            pytest_result, pytest_weight_map or {}
        )
        effective_alpha = factoid_alpha
        pytest_status = "scored"
    else:
        earned, total = 0.0, 0.0
        if not has_per_test:
            if has_weight_map:
                # v4 (王-2): task declares a pytest layer (weight_map exists)
                # but collection yielded zero lines — treat as broken, not
                # absent. Without this, a completely broken test suite would
                # give effective_alpha=1.0 and silently boost the score.
                pytest_score = 0.0
                effective_alpha = declared_alpha
                pytest_status = "pytest_broken_no_collection"
            else:
                # Legitimately absent — task has no pytest layer.
                pytest_score = 0.0
                effective_alpha = 1.0
                pytest_status = "no_tests_collected"
        elif not has_weight_map:
            # Author did not declare weights — treat pytest as absent.
            pytest_score = 0.0
            effective_alpha = 1.0
            pytest_status = "no_weight_map"
        else:
            # v3 (吴-4): per_test collected, weight_map declared, but no
            # PASSED/FAILED line — pytest layer is broken (conftest error,
            # collection failure, all-SKIP). Score as 0 with declared α so
            # the model does not benefit from a broken artifact layer.
            pytest_score = 0.0
            effective_alpha = declared_alpha
            pytest_status = "pytest_broken"

    combined = effective_alpha * factoid_score + (1 - effective_alpha) * pytest_score

    result = {
        "combined_score": round(combined, 4),
        "factoid_score": round(factoid_score, 4),
        "pytest_score": round(pytest_score, 4),
        # v2 (#23): expose both fields so reports can distinguish config vs applied.
        "factoid_alpha": effective_alpha,
        "declared_alpha": declared_alpha,
        "effective_alpha": effective_alpha,
        "pytest_status": pytest_status,
        "pytest_details": {
            "passed": pytest_result.passed,
            "failed": pytest_result.failed,
            "skipped": pytest_result.skipped,
            "errored": pytest_result.errored,
            "earned_weight": round(earned, 2),
            "total_weight": round(total, 2),
            # v3 (李-2): per-test roster (name + status only, no raw_output).
            "per_test": [
                {"name": t["name"], "status": t["status"]}
                for t in pytest_result.per_test
            ],
            # v3 (吴-1): weight-matching diagnostics.
            "matched_names": weight_details.get("matched_names", []),
            "unmatched_names": weight_details.get("unmatched_names", []),
            "matched_weight_keys": weight_details.get("matched_weight_keys", []),
            "unused_weight_keys": weight_details.get("unused_weight_keys", []),
        },
        "factoid_details": factoid_result,
    }
    result["veto_status"] = (
        evaluate_veto(score_json=result, groundtruth=groundtruth)
        if groundtruth is not None else {"configured": False}
    )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Wrap Harbor pytest into weighted scoring")
    parser.add_argument("--test-dir", required=True, help="Path to tests/ folder")
    parser.add_argument("--workspace", required=True, help="Workspace with agent output")
    parser.add_argument("--factoid-result", required=True, help="scorer.py output JSON")
    parser.add_argument("--pytest-weight-map", default=None, help="JSON file with test-name → weight")
    parser.add_argument("--factoid-alpha", type=float, default=0.9)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    weights: dict[str, float] = {}
    if args.pytest_weight_map:
        with open(args.pytest_weight_map, "r", encoding="utf-8") as f:
            weights = json.load(f)

    pytest_res = run_pytest(Path(args.test_dir), Path(args.workspace))
    with open(args.factoid_result, "r", encoding="utf-8") as f:
        factoid_res = json.load(f)

    result = combined_score(
        factoid_result=factoid_res,
        pytest_result=pytest_res,
        pytest_weight_map=weights,
        factoid_alpha=args.factoid_alpha,
    )
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "factoid_details"}, ensure_ascii=False, indent=2))
