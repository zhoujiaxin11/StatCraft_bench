"""
framework/aggregate.py
======================
Multi-run aggregation layer: turn per-trial score.json files into
statistically defensible model scores.

Why this exists
---------------
runner.py produces one score.json per (task × model × seed) trial. A single
run is NOT a reliable estimate of model ability — CausalDS showed the same
task run 3× has "at-least-1-correct" 76-91% but "all-3-correct" only 50-62%.
So we must:

  1. Aggregate the ≥5 seeds of a task into a STABLE score (mean) + a
     CONSISTENCY measure (are the 5 runs tight or all-over-the-place?).
  2. Aggregate tasks into a MODEL score by normalizing per task first, then
     equal-weight averaging (HiBayES: never pre-average unequal-weight tasks).
  3. Attach a bootstrap 95% CI (resampling tasks) so model ranking is
     compared by CI OVERLAP, not by point estimate (low-sample ranks are noisy).

Design sources
--------------
- HiBayES: keep item-level scores, interval on the total, compare by overlap
- SkillsBench: pass-rate = mean over ≥5 runs; report absolute + normalized gain
- CausalDS: capability ≠ reliability — report a consistency signal separately

Usage
-----
    # aggregate one task's seeds for one model
    python -m framework.aggregate task \\
        --task tasks/social_stats/159_gaokao_reform --model gpt-5.5

    # aggregate a whole model across all tasks
    python -m framework.aggregate model \\
        --tasks-root tasks --model gpt-5.5

    # compare two models by CI overlap
    python -m framework.aggregate compare \\
        --tasks-root tasks --model-a gpt-5.5 --model-b claude-sonnet-4.5
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class TaskAggregate:
    """One task's ≥N seeds for one model, collapsed to a stable estimate."""
    task_id: str
    model: str
    n_runs: int
    scores: list[float]              # raw combined_score per seed (usable only)
    stable_score: float              # mean over usable seeds (the "单题稳定分")
    std: float                       # dispersion across usable seeds
    consistency: float | None        # 1 - normalized std; None when n_runs < 3
    pass_at_1: float                 # fraction of runs >= pass_threshold
    pass_all: float                  # 1.0 if ALL runs passed else 0.0
    bimodal: bool | None             # unstable / mixed-outcome flag; None when n_runs < 3
    vetoed_runs: int = 0             # runs zeroed by essential/unacceptable veto
    # v2 (#28, 吴-5 / QA-5): crash accounting so scores measure ability, not
    # network luck. Crashed runs are EXCLUDED from stable_score / std /
    # consistency / pass_*.
    n_total: int = 0                 # every trial found on disk
    n_crashed: int = 0
    crash_rate: float = 0.0
    crash_statuses: dict[str, int] = field(default_factory=dict)
    # v3 (吴-3): split infra vs model_fail accounting so reports can tell
    # "we lost this trial to network noise" from "the model gave up".
    n_infra: int = 0                 # dropped from mean (infra crashed)
    n_model_fail: int = 0            # counted as 0 in mean (model gave up)
    infra_statuses: dict[str, int] = field(default_factory=dict)
    model_fail_statuses: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelAggregate:
    """One model across all its tasks."""
    model: str
    n_tasks: int
    per_task_stable: dict[str, float]
    overall_score: float             # equal-weight mean of per-task stable scores
    ci_low: float | None
    ci_high: float | None
    mean_consistency: float          # avg consistency across tasks
    flagged_bimodal: list[str] = field(default_factory=list)
    # v2 (#29, 吴-6 / QA-6): flag insufficient-data outputs.
    insufficient_tasks: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Veto rules (essential / unacceptable one-vote-veto)
# ---------------------------------------------------------------------------
def validate_veto(groundtruth: dict) -> None:
    """Fail fast if veto references field keys not in weights/scoring.

    Without this check, a typo silently zeros every run (essential field
    missing from per_field → default 0 → veto trips), which is worse than
    a loud error.
    """
    veto = groundtruth.get("veto")
    if not veto:
        return
    known = set(groundtruth.get("weights", {}).keys()) \
        | set(groundtruth.get("scoring", {}).keys())
    unknown = []
    for kind in ("essential", "unacceptable"):
        for f in veto.get(kind, []):
            if f not in known:
                unknown.append(f"{kind}:{f}")
    if unknown:
        raise ValueError(
            f"veto references unknown fields (not in weights/scoring): {unknown}"
        )


def apply_veto(score_json: dict, groundtruth: dict) -> tuple[float, bool]:
    """Return (possibly-zeroed score, was_vetoed).

    Reads optional `veto` block from groundtruth:
        "veto": {
            "essential":     ["part2_reform_impact.reform_direction", ...],
            "unacceptable":  ["part4_causal.wrong_sign_flag", ...]
        }
    - essential:    field score must be > 0, else the whole task is zeroed.
    - unacceptable: field score must be 0 (i.e. the bad thing was NOT done);
                    if the field scored > 0 it means the forbidden mistake was
                    made → task zeroed.

    Veto keys are looked up in the per-field breakdown of the factoid scorer.
    If no veto block exists, the score passes through unchanged.
    """
    veto = groundtruth.get("veto")
    if not veto:
        return score_json.get("combined_score", 0.0), False

    per_field = (
        score_json.get("factoid_details", {}).get("per_field", {})
    )
    combined = score_json.get("combined_score", 0.0)

    for f in veto.get("essential", []):
        if per_field.get(f, 0.0) <= 0.0:
            return 0.0, True

    for f in veto.get("unacceptable", []):
        if per_field.get(f, 0.0) > 0.0:
            return 0.0, True

    return combined, False


# ---------------------------------------------------------------------------
# Trial collection
# ---------------------------------------------------------------------------
# v3 (吴-3 / 罗-3 / 魏-3): distinguish two failure modes so scores measure
# ability, not infrastructure luck:
#   * INFRA_STATUSES     : the trial did not measure the model at all
#                          (network died, executor crashed, scorer bug). These
#                          are EXCLUDED from stable_score / std / consistency
#                          / pass_* — they contribute only to crash_rate.
#   * MODEL_FAIL_STATUSES: the model failed to produce a valid answer within
#                          its own budget (max turns, no output, invalid
#                          JSON, agent got stuck). These COUNT as 0 in the
#                          usable pool — the model earned 0 for the task.
#
# Pre-v3 both buckets were collapsed into CRASH_STATUSES and dropped, which
# meant "not answering" (max_turns / no_output) scored higher than "answering
# badly", and the ranking could flip.
# v4 (吴-3): import from the single source of truth.
from framework.statuses import (
    INFRA_STATUSES,
    MODEL_FAIL_STATUSES,
    CRASH_STATUSES,
    bucket as _status_bucket,
)


def collect_trial_scores(
    task_dir: Path,
    model: str,
    groundtruth: dict | None = None,
) -> list[dict]:
    """Gather per-trial score records for `model`.

    Returns a list of dicts with keys:
        score      : combined_score after veto
        vetoed     : bool
        status     : str | None
        bucket     : "ok" | "infra" | "model_fail"
        usable     : bool  (True when bucket != "infra"; "model_fail" IS usable
                            and contributes a 0 to the mean)

    Trials are directories named like `20260731_140353_gpt-5.5_seed2`. We match
    on the model token to avoid mixing models, and skip oracle trials.
    """
    trials_dir = task_dir / "trials"
    records: list[dict] = []
    if not trials_dir.exists():
        return records

    def _bucket(st: str) -> str:
        b = _status_bucket(st)
        # "unknown" is treated as infra (conservative: don't credit model)
        return "infra" if b == "unknown" else b

    for trial in sorted(trials_dir.iterdir()):
        if not trial.is_dir():
            continue
        name = trial.name
        if "oracle" in name:
            continue
        if f"_{model}_seed" not in name:
            continue
        score_path = trial / "score.json"
        if not score_path.exists():
            continue
        try:
            with open(score_path, "r", encoding="utf-8") as fh:
                sj = json.load(fh)
        except OSError:
            # v4 (李-2): disk I/O failure is infra — not the model's fault.
            # Drop from the ability estimate (usable=False).
            records.append({
                "score": 0.0, "vetoed": False,
                "status": "bad_io", "bucket": "infra", "usable": False,
            })
            continue
        except json.JSONDecodeError:
            # Agent wrote invalid JSON — model failure, counts as 0.
            records.append({
                "score": 0.0, "vetoed": False,
                "status": "bad_json", "bucket": "model_fail", "usable": True,
            })
            continue
        if groundtruth is not None:
            s, vetoed = apply_veto(sj, groundtruth)
        else:
            s = sj.get("combined_score", 0.0)
            vetoed = False
        status = sj.get("status")
        # Legacy score.json without status field: infer no_output when the
        # scorer left an "error" marker; otherwise assume OK.
        if status is None:
            if isinstance(sj.get("error"), str):
                status = "no_output"
            else:
                status = "ok"
        bucket = _bucket(status)
        # v3 (吴-3): "usable" means "contributes to ability estimate". Infra
        # failures are dropped; model-fail counts as 0.
        usable = bucket != "infra"
        # Model failures score 0 regardless of what score.json says (defensive:
        # a partially-written score.json with a stale combined_score would
        # otherwise leak into the mean).
        if bucket == "model_fail":
            s = 0.0
        records.append({
            "score": float(s),
            "vetoed": bool(vetoed),
            "status": status,
            "bucket": bucket,
            "usable": usable,
        })
    return records


# ---------------------------------------------------------------------------
# Task-level aggregation
# ---------------------------------------------------------------------------
def aggregate_task(
    task_dir: Path,
    model: str,
    *,
    pass_threshold: float | None = None,
    groundtruth: dict | None = None,
) -> TaskAggregate | None:
    """Collapse one task's seeds for one model into a stable estimate."""
    gt = groundtruth
    if gt is None:
        gt = _load_task_groundtruth(task_dir)
    if gt is not None:
        validate_veto(gt)
    if pass_threshold is None:
        pass_threshold = _read_pass_threshold(task_dir, gt)

    raw = collect_trial_scores(task_dir, model, groundtruth=gt)
    if not raw:
        return None

    n_total = len(raw)
    usable = [r for r in raw if r["usable"]]
    infra = [r for r in raw if r["bucket"] == "infra"]
    model_fail = [r for r in raw if r["bucket"] == "model_fail"]
    n_infra = len(infra)
    n_model_fail = len(model_fail)
    # v3 (吴-3): crash_rate is now infra-only. Model-fail contributes 0 to the
    # mean but isn't "crashed" — the model measurable behaved and lost.
    crash_rate = round(n_infra / n_total, 4) if n_total else 0.0
    infra_statuses: dict[str, int] = {}
    for r in infra:
        key = r["status"] or "unknown"
        infra_statuses[key] = infra_statuses.get(key, 0) + 1
    model_fail_statuses: dict[str, int] = {}
    for r in model_fail:
        key = r["status"] or "unknown"
        model_fail_statuses[key] = model_fail_statuses.get(key, 0) + 1
    # Legacy crash_statuses = union (retained for backward-compat callers).
    crash_statuses: dict[str, int] = {**infra_statuses, **model_fail_statuses}
    n_crashed = n_infra + n_model_fail

    if not usable:
        # Every trial was an infra crash — ability is unmeasured, not 0.
        return TaskAggregate(
            task_id=_read_task_id(task_dir),
            model=model,
            n_runs=0,
            scores=[],
            stable_score=0.0,
            std=0.0,
            consistency=None,
            pass_at_1=0.0,
            pass_all=0.0,
            bimodal=None,
            vetoed_runs=sum(1 for r in raw if r["vetoed"]),
            n_total=n_total,
            n_crashed=n_crashed,
            crash_rate=crash_rate,
            crash_statuses=crash_statuses,
            n_infra=n_infra,
            n_model_fail=n_model_fail,
            infra_statuses=infra_statuses,
            model_fail_statuses=model_fail_statuses,
        )

    scores = [r["score"] for r in usable]
    vetoed_count = sum(1 for r in usable if r["vetoed"])
    n = len(scores)
    mean = statistics.fmean(scores)
    sd = statistics.pstdev(scores) if n > 1 else 0.0

    # Consistency and bimodal are unreliable at low N.
    if n >= 3:
        consistency = max(0.0, 1.0 - (sd / 0.5))
        passes = [1 if s >= pass_threshold else 0 for s in scores]
        pass_at_1 = statistics.fmean(passes)
        pass_all = 1.0 if all(passes) else 0.0
        # v2 (#27, 魏-2): bimodal means "not everyone-passed and not everyone-failed"
        # — i.e. the seeds disagree, which is exactly the instability signal.
        # Pre-v2 the polarity was inverted (flagged rock-stable tasks as unstable).
        bimodal = not (all(p == 1 for p in passes) or all(p == 0 for p in passes))
    else:
        import warnings
        warnings.warn(
            f"Only {n} usable run(s) for {model} on {task_dir.name}; "
            "consistency/bimodal not reliable. Use ≥5 seeds.",
            stacklevel=2,
        )
        consistency = None
        passes = [1 if s >= pass_threshold else 0 for s in scores]
        pass_at_1 = statistics.fmean(passes)
        pass_all = 1.0 if all(passes) else 0.0
        bimodal = None

    return TaskAggregate(
        task_id=_read_task_id(task_dir),
        model=model,
        n_runs=n,
        scores=[round(s, 4) for s in scores],
        stable_score=round(mean, 4),
        std=round(sd, 4),
        consistency=round(consistency, 4) if consistency is not None else None,
        pass_at_1=round(pass_at_1, 4),
        pass_all=pass_all,
        bimodal=bimodal,
        vetoed_runs=vetoed_count,
        n_total=n_total,
        n_crashed=n_crashed,
        crash_rate=crash_rate,
        crash_statuses=crash_statuses,
        n_infra=n_infra,
        n_model_fail=n_model_fail,
        infra_statuses=infra_statuses,
        model_fail_statuses=model_fail_statuses,
    )


# ---------------------------------------------------------------------------
# Bootstrap CI (resample tasks)
# ---------------------------------------------------------------------------
def bootstrap_ci(
    task_scores: list[float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    min_tasks: int = 5,
) -> tuple[float | None, float | None]:
    """95% CI for the equal-weight mean, by resampling TASKS with replacement.

    This is the HiBayES-flavored uncertainty: the spread comes from which tasks
    you happened to include, which is the real source of ranking instability at
    20-50 tasks.

    v2 (#29, 吴-6 / QA-6): return (None, None) when there are fewer than
    `min_tasks` tasks. A zero-width CI on a single task reads like "extremely
    confident" when the truth is "not enough data to bootstrap".
    """
    if len(task_scores) < min_tasks:
        return None, None

    rng = random.Random(seed)
    k = len(task_scores)
    means: list[float] = []
    for _ in range(n_boot):
        sample = [task_scores[rng.randrange(k)] for _ in range(k)]
        means.append(statistics.fmean(sample))
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return round(lo, 4), round(hi, 4)


# ---------------------------------------------------------------------------
# Model-level aggregation
# ---------------------------------------------------------------------------
def aggregate_model(
    tasks_root: Path,
    model: str,
    *,
    n_boot: int = 2000,
    min_tasks_for_ci: int = 5,
) -> ModelAggregate:
    """Aggregate one model across every task under tasks_root.

    Normalizes PER TASK first (each task's stable score is already in [0,1]),
    then equal-weight averages — so a task with more checkpoints does not get
    silently upweighted (HiBayES warning).

    v2 (#29): fewer than ``min_tasks_for_ci`` tasks → CI returns None rather
    than a fake zero-width interval.
    """
    per_task: dict[str, float] = {}
    consistencies: list[float] = []
    flagged: list[str] = []

    for task_dir in _iter_task_dirs(tasks_root):
        agg = aggregate_task(task_dir, model)
        if agg is None:
            continue
        # Skip tasks with no usable runs; they carry no ability signal.
        if agg.n_runs == 0:
            continue
        per_task[agg.task_id] = agg.stable_score
        if agg.consistency is not None:
            consistencies.append(agg.consistency)
        if agg.bimodal:
            flagged.append(agg.task_id)

    scores = list(per_task.values())
    overall = statistics.fmean(scores) if scores else 0.0
    ci_low, ci_high = bootstrap_ci(scores, n_boot=n_boot, min_tasks=min_tasks_for_ci)
    mean_cons = statistics.fmean(consistencies) if consistencies else 0.0

    return ModelAggregate(
        model=model,
        n_tasks=len(per_task),
        per_task_stable={k: round(v, 4) for k, v in per_task.items()},
        overall_score=round(overall, 4),
        ci_low=ci_low,
        ci_high=ci_high,
        mean_consistency=round(mean_cons, 4),
        flagged_bimodal=flagged,
        insufficient_tasks=(len(scores) < min_tasks_for_ci),
    )


def compare_models(
    agg_a: ModelAggregate,
    agg_b: ModelAggregate,
    *,
    min_shared_tasks: int = 5,
    n_boot: int = 2000,
) -> dict:
    """Paired-difference bootstrap test: are two models statistically different?

    Instead of comparing marginal CIs (which ignores the paired structure and
    inflates uncertainty), we directly bootstrap the PER-TASK score difference:
        diff_i = score_a_i − score_b_i, for each shared task
    and report a 95% CI on mean(diff). If the CI excludes zero, the models
    are significantly different.

    v2 (#29): with fewer than ``min_shared_tasks`` shared tasks we refuse to
    declare a winner — the paired bootstrap is not reliable.
    """
    shared_tasks = sorted(
        set(agg_a.per_task_stable.keys()) & set(agg_b.per_task_stable.keys())
    )
    if not shared_tasks:
        return {
            "model_a": agg_a.model,
            "model_b": agg_b.model,
            "error": "no shared tasks to compare",
            "verdict": "insufficient_data",
        }
    if len(shared_tasks) < min_shared_tasks:
        diffs = [
            agg_a.per_task_stable[t] - agg_b.per_task_stable[t] for t in shared_tasks
        ]
        return {
            "model_a": agg_a.model,
            "model_b": agg_b.model,
            "score_a": agg_a.overall_score,
            "score_b": agg_b.overall_score,
            "n_shared_tasks": len(shared_tasks),
            "mean_diff_a_minus_b": round(statistics.fmean(diffs), 4) if diffs else None,
            "ci_diff_95": [None, None],
            "significant": False,
            "verdict": "insufficient_data",
            "reason": (
                f"only {len(shared_tasks)} shared task(s); need ≥ {min_shared_tasks} "
                "for a reliable paired bootstrap."
            ),
        }
    diffs = [agg_a.per_task_stable[t] - agg_b.per_task_stable[t] for t in shared_tasks]
    mean_diff = statistics.fmean(diffs)

    # Paired bootstrap: resample task indices WITH replacement
    rng = random.Random(0)
    k = len(diffs)
    boot_means: list[float] = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(k)] for _ in range(k)]
        boot_means.append(statistics.fmean(sample))
    boot_means.sort()
    alpha = 0.05
    ci_low = boot_means[int((alpha / 2) * n_boot)]
    ci_high = boot_means[int((1 - alpha / 2) * n_boot) - 1]

    sig = ci_low > 0 or ci_high < 0
    if sig:
        better = agg_a.model if mean_diff > 0 else agg_b.model
        verdict = f"{better} is significantly higher (paired diff CI excludes 0)"
    else:
        # v2 (#29): "inconclusive" — the data does not resolve a difference,
        # which is not the same as "equal".
        verdict = "inconclusive (paired diff CI includes 0; data does not distinguish)"

    return {
        "model_a": agg_a.model,
        "model_b": agg_b.model,
        "score_a": agg_a.overall_score,
        "score_b": agg_b.overall_score,
        "n_shared_tasks": len(shared_tasks),
        "mean_diff_a_minus_b": round(mean_diff, 4),
        "ci_diff_95": [round(ci_low, 4), round(ci_high, 4)],
        "significant": sig,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _iter_task_dirs(tasks_root: Path):
    """Yield every directory that contains a task.toml under tasks_root."""
    for toml_path in sorted(tasks_root.rglob("task.toml")):
        yield toml_path.parent


def _read_task_id(task_dir: Path) -> str:
    with open(task_dir / "task.toml", "rb") as fh:
        return _toml_load(fh)["task"]["id"]


def _toml_load(fh):
    """Load TOML with the stdlib parser (3.11+) or the tomli backport (3.10-)."""
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    return tomllib.load(fh)


def _read_pass_threshold(task_dir: Path, groundtruth: dict | None) -> float:
    """pass_threshold priority: groundtruth > task.toml [scoring] > 0.5 default."""
    if groundtruth and "pass_threshold" in groundtruth:
        return float(groundtruth["pass_threshold"])
    try:
        with open(task_dir / "task.toml", "rb") as fh:
            cfg = _toml_load(fh)
        return float(cfg.get("scoring", {}).get("pass_threshold", 0.5))
    except (OSError, KeyError, ValueError):
        return 0.5


def _load_task_groundtruth(task_dir: Path) -> dict | None:
    """Best-effort load of the task's public groundtruth JSON (for veto block)."""
    task_id = _read_task_id(task_dir)
    # tasks_root/../groundtruth/public/<id>.json — walk up to find groundtruth/
    for parent in task_dir.parents:
        gt = parent / "groundtruth" / "public" / f"{task_id}.json"
        if gt.exists():
            try:
                with open(gt, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except (OSError, json.JSONDecodeError):
                return None
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-run trial scores")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_task = sub.add_parser("task", help="Aggregate one task's seeds for one model")
    p_task.add_argument("--task", required=True)
    p_task.add_argument("--model", required=True)

    p_model = sub.add_parser("model", help="Aggregate one model across all tasks")
    p_model.add_argument("--tasks-root", required=True)
    p_model.add_argument("--model", required=True)
    p_model.add_argument("--n-boot", type=int, default=2000)

    p_cmp = sub.add_parser("compare", help="Compare two models by CI overlap")
    p_cmp.add_argument("--tasks-root", required=True)
    p_cmp.add_argument("--model-a", required=True)
    p_cmp.add_argument("--model-b", required=True)
    p_cmp.add_argument("--n-boot", type=int, default=2000)

    args = parser.parse_args()

    if args.cmd == "task":
        agg = aggregate_task(Path(args.task).resolve(), args.model)
        print(json.dumps(agg.to_dict() if agg else {"error": "no trials found"},
                         ensure_ascii=False, indent=2))
    elif args.cmd == "model":
        agg = aggregate_model(Path(args.tasks_root).resolve(), args.model, n_boot=args.n_boot)
        print(json.dumps(agg.to_dict(), ensure_ascii=False, indent=2))
    elif args.cmd == "compare":
        root = Path(args.tasks_root).resolve()
        a = aggregate_model(root, args.model_a, n_boot=args.n_boot)
        b = aggregate_model(root, args.model_b, n_boot=args.n_boot)
        print(json.dumps(compare_models(a, b, n_boot=args.n_boot), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
