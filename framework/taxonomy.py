"""
framework/taxonomy.py
=====================
Single source of truth for the task taxonomy
vocabulary — ``scenario`` and ``difficulty``.

Why this module exists
----------------------
TASK_AUTHORING.md declares a closed set of 12 scenarios ("从下面 12 个中选一个")
and 3 difficulty tiers, but until now NOTHING validated either one. A task.toml
could declare ``scenario = "medicall_health"``, live under
``tasks/medical_health/``, and be listed in the registry as ``medical_health``
— three different answers to "what is this task", with no error anywhere. That
is the mechanism behind item 14: after a task is reworked, the thing
being scored quietly drifts away from the thing that was selected, and
per-scenario aggregates silently split into a real bucket and a typo bucket.

The check is deliberately a WARNING, never a hard failure — item 14 asks for
「补白名单校验并做成告警而非硬失败」. A taxonomy label has no effect on a
trial's score, so refusing to run over one would cost model time for a
bookkeeping mistake.

This mirrors ``statuses.py``: the vocabulary lives in exactly one place so
item 17 ("scenario 白名单尚未更新") is a one-line edit here rather than a hunt
through prose, task.toml files and the registry. The contents below are copied
verbatim from TASK_AUTHORING.md §三 — updating the vocabulary itself is a group
decision, not something this module should invent.
"""
from __future__ import annotations

import json
from pathlib import Path

try:
    import tomllib as _tomllib  # Python 3.11+
except ModuleNotFoundError:  # 3.10 and earlier
    import tomli as _tomllib  # type: ignore[no-redef]


# The 12 scenarios from TASK_AUTHORING.md §三. Keep in sync with that document;
# this frozenset is what the code actually enforces. The Chinese gloss next to
# each key is informational only — the enforced identifier is the ASCII slug.
SCENARIOS = frozenset({
    "medical_health",         # 医疗·生命健康
    "finance_economics",      # 金融·经济
    "retail_ecommerce",       # 零售·电商·消费营销
    "industrial_energy",      # 工业·能源·制造
    "transport_logistics",    # 交通·物流
    "environment_climate",    # 环境·气候·生态·农业监测
    "geo_hazards",            # 自然灾害·极值事件·地球物理
    "education_academia",     # 教育·学术
    "society_policy",          # 社会·公共政策·人口
    "tech_internet",          # 科技·互联网·媒体信息
    "sports_entertainment",   # 体育·娱乐·文旅
    "bio_chem_materials",     # 生物·化学·材料
})

# TASK_AUTHORING.md §四 / §八.
DIFFICULTIES = frozenset({"easy", "hard", "extreme"})

# TASK_AUTHORING.md §七: the 40-tag closed capability vocabulary. Every
# task.toml [taxonomy].capabilities entry AND every capability_map value in
# groundtruth JSON must be a member. Adding a tag = spec-change PR + edit both
# this frozenset and §七 in one go. Same drift argument as SCENARIOS: a typo
# here silently splits per-capability aggregates into a real bucket and a typo
# bucket, and nothing else in the codebase would catch it.
CAPABILITIES = frozenset({
    # data handling
    "data_cleaning", "data_format_handling", "data_manipulation",
    "data_deduplication", "missing_value_handling",
    # descriptive statistics
    "descriptive_statistics", "weighted_statistics", "grouped_comparison",
    "cross_tabulation",
    # inferential statistics
    "hypothesis_testing", "parametric_test", "nonparametric_test",
    "multiple_comparison", "power_analysis",
    # regression / modeling
    "regression_modeling", "time_series_features", "feature_engineering",
    "model_selection", "hyperparameter_tuning", "cross_validation",
    # causal inference
    "did_estimation", "synthetic_control", "placebo_test",
    "parallel_trend_test", "propensity_matching", "iv_estimation",
    # advanced statistics
    "evt_tail_index", "panel_ols_inference", "bayesian_inference",
    "survival_analysis", "mixed_effects_model",
    # distance / distribution
    "wasserstein_distance", "kl_divergence", "distribution_test",
    # uncertainty
    "conformal_prediction", "bootstrap_ci", "bayesian_posterior",
    # engineering / safety
    "leakage_prevention", "code_correctness", "numerical_stability",
    "reproducibility", "output_format_compliance",
})


def validate_task_taxonomy(task_dir: Path, cfg: dict | None = None) -> list[str]:
    """Cross-check a task's taxonomy against the whitelist and its own layout.

    Returns a list of human-readable warnings; an empty list means clean.
    Never raises for a taxonomy problem and never fails a run — see the module
    docstring on why this is advisory.

    Three checks, in the order they catch real drift:

    1. ``scenario`` / ``difficulty`` are in the whitelist. A typo here splits a
       scenario's aggregate in two without any visible error.
    2. ``scenario`` equals the parent directory name. The layout contract is
       ``tasks/<scenario>/<task_id>/``, so a disagreement means the task moved
       (or was copied during a rework) without its metadata following.
    3. The registry entry, if one exists, agrees with task.toml on both
       ``scenario`` and ``difficulty``. This is the "评分对象漂移" case from
       item 14: the task was selected into the round under one label and is
       being scored under another.
    """
    task_dir = Path(task_dir)
    problems: list[str] = []
    if cfg is None:
        try:
            with open(task_dir / "task.toml", "rb") as f:
                cfg = _tomllib.load(f)
        except (OSError, ValueError) as exc:
            return [f"task.toml unreadable, taxonomy not checked: {exc}"]

    taxonomy = cfg.get("taxonomy", {}) or {}
    task_id = (cfg.get("task", {}) or {}).get("id", task_dir.name)
    scenario = taxonomy.get("scenario")
    difficulty = taxonomy.get("difficulty")

    # 1. whitelist membership
    if scenario is None:
        problems.append("[taxonomy].scenario is missing (required by TASK_AUTHORING.md)")
    elif scenario not in SCENARIOS:
        problems.append(
            f"[taxonomy].scenario={scenario!r} is not in the whitelist "
            f"(allowed: {sorted(SCENARIOS)}) — a typo here silently splits the "
            "per-scenario aggregate into two buckets"
        )
    if difficulty is None:
        problems.append("[taxonomy].difficulty is missing (required by TASK_AUTHORING.md)")
    elif difficulty not in DIFFICULTIES:
        problems.append(
            f"[taxonomy].difficulty={difficulty!r} is not in the whitelist "
            f"(allowed: {sorted(DIFFICULTIES)})"
        )

    # 2. declared scenario vs on-disk location
    parent = task_dir.parent.name
    if scenario is not None and parent and parent != "tasks" and scenario != parent:
        problems.append(
            f"[taxonomy].scenario={scenario!r} but the task lives under "
            f"tasks/{parent}/ — the directory and the metadata disagree about "
            "which scenario this task belongs to"
        )

    # 3. registry agreement
    problems.extend(_registry_disagreements(task_dir, task_id, scenario, difficulty))
    return problems


def _registry_disagreements(
    task_dir: Path, task_id: str, scenario, difficulty
) -> list[str]:
    """Compare task.toml against every registry entry that names this task."""
    root = task_dir.parent.parent.parent  # tasks/<scenario>/<id> → repo root
    registry_dir = root / "registry"
    if not registry_dir.is_dir():
        return []
    out: list[str] = []
    for reg_path in sorted(registry_dir.glob("*.json")):
        try:
            with open(reg_path, "r", encoding="utf-8") as f:
                reg = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue  # a broken registry is not this check's business
        entries = list(reg.get("tasks", []) or []) + list(reg.get("hidden_tasks", []) or [])
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("task_id") != task_id:
                continue
            reg_scenario = entry.get("scenario")
            reg_difficulty = entry.get("difficulty")
            if reg_scenario is not None and scenario is not None and reg_scenario != scenario:
                out.append(
                    f"{reg_path.name} lists scenario={reg_scenario!r} but task.toml "
                    f"says {scenario!r} — the task was selected under one label and "
                    "is being scored under another"
                )
            if (
                reg_difficulty is not None
                and difficulty is not None
                and reg_difficulty != difficulty
            ):
                out.append(
                    f"{reg_path.name} lists difficulty={reg_difficulty!r} but "
                    f"task.toml says {difficulty!r} — difficulty must stay fixed "
                    "across a task rework (item 14)"
                )
    return out
