"""
framework/ci_validate.py
========================
Structural + content validation for the tasks/ + groundtruth/ + registry/
tree. Runs offline (no Docker, no models, no data files needed) so both
contributors and CI can catch drift in <5 s before any long-running gate
kicks in.

Two subcommands, matching the CI layout in `.github/workflows/ci.yml`:

- ``--structure`` (L1, fast)
    * every ``tasks/<scenario>/<task_id>/`` has the five required files
      declared in TASK_AUTHORING.md §二
    * ``task.toml`` parses and has the six required sections + basic fields
    * ``schema.json`` and ``groundtruth/public/<task_id>.json`` are valid JSON
    * groundtruth ``weights`` / ``scoring`` / ``values`` key sets are internally
      consistent (no orphan weights, no scored-but-unweighted fields)
    * every ``registry/*.json`` is valid; registry↔tasks/ agree **both ways**
      — every registry entry has a task on disk, every task on disk is listed
      by at least one registry (this is what stops the "20 tasks in the
      description but 1 on disk" drift this check exists to catch)
    * registry ``difficulty_distribution`` matches counts of tasks it lists

- ``--content`` (L2, still fast, semantic)
    * ``taxonomy.validate_task_taxonomy`` per task (warnings surface here as
      errors — CI is the one place we want the drift to block a merge)
    * every ``[taxonomy].capabilities`` entry ⊆ ``CAPABILITIES``
    * every capability inside groundtruth's ``capability_map`` ⊆ ``CAPABILITIES``
    * ``validate_veto`` from aggregate.py (fails loud on typos)
    * scoring rule types are ones scorer.py actually implements

Both subcommands collect every problem and print them together, then exit 1
if any survived. A single ``python3 -m framework.ci_validate --structure
--content`` runs both back-to-back — that's what CONTRIBUTING.md's Step 1
should point to for the "local pre-flight" check.

The heavy checks (oracle ≥ 0.99, pytest) are NOT here — those need to
actually run code and are the L3 job in the workflow.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    import tomllib as _tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as _tomllib  # type: ignore[no-redef]

from framework.taxonomy import (
    CAPABILITIES,
    DIFFICULTIES,
    SCENARIOS,
    validate_task_taxonomy,
)


# TASK_AUTHORING.md §二: the five required per-task files. Dockerfile and
# tests/ are optional, so they are not in this set.
_REQUIRED_TASK_FILES = (
    "task.toml",
    "instruction.md",
    "schema.json",
    "environment",              # directory, existence-only
    "solution/solve.py",
)

# TASK_AUTHORING.md §四: task.toml top-level sections that must exist.
_REQUIRED_TOML_SECTIONS = (
    "task",
    "taxonomy",
    "environment",
    "resources",
    "required_outputs",
    "scoring",
)

# TASK_AUTHORING.md §九: closed vocabulary of scoring rule types actually
# implemented in scorer.py. Kept here (rather than imported) because ci_validate
# is meant to run without needing to import scorer.py's numeric machinery —
# a single frozenset is cheaper and the sync is a spec-change PR anyway.
_KNOWN_RULE_TYPES = frozenset({
    "exact_number", "exact_string", "graded",
    "min_threshold", "max_threshold",
    "bool", "enum", "accept_set",
    "list_ordered", "list_set", "object_keys_match",
    "no_leakage_keywords", "contains", "presence",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iter_task_dirs(root: Path):
    """Yield every ``tasks/<scenario>/<task_id>/`` directory that contains a
    ``task.toml`` — the marker file, not the dir name, is what makes a task."""
    tasks_root = root / "tasks"
    if not tasks_root.is_dir():
        return
    for scenario_dir in sorted(p for p in tasks_root.iterdir() if p.is_dir()):
        for task_dir in sorted(p for p in scenario_dir.iterdir() if p.is_dir()):
            if (task_dir / "task.toml").is_file():
                yield task_dir


def _load_toml(path: Path):
    try:
        with open(path, "rb") as f:
            return _tomllib.load(f), None
    except (OSError, ValueError) as exc:
        return None, f"{path}: {exc}"


def _load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{path}: {exc}"


def _rel(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Structure checks (L1)
# ---------------------------------------------------------------------------

def check_structure(root: Path) -> list[str]:
    """Return every structural problem; empty list = clean."""
    problems: list[str] = []
    tasks_on_disk: dict[str, dict] = {}   # task_id -> {scenario, difficulty, task_dir}

    task_dirs = list(_iter_task_dirs(root))
    if not task_dirs:
        problems.append("tasks/: no task directories with task.toml found")

    for task_dir in task_dirs:
        rel = _rel(root, task_dir)

        # required files
        for name in _REQUIRED_TASK_FILES:
            target = task_dir / name
            if name.endswith("/") or "/" not in name and target.is_dir():
                ok = target.is_dir()
            elif "/" in name:
                ok = target.is_file()
            else:
                ok = target.is_file() or target.is_dir()
            if not ok:
                problems.append(f"{rel}: missing required entry {name!r}")

        # task.toml required sections + basic fields
        cfg, err = _load_toml(task_dir / "task.toml")
        if err:
            problems.append(f"{rel}/task.toml: unreadable ({err})")
            continue
        for section in _REQUIRED_TOML_SECTIONS:
            if section not in cfg:
                problems.append(f"{rel}/task.toml: missing section [{section}]")

        task_id = (cfg.get("task", {}) or {}).get("id")
        if not task_id:
            problems.append(f"{rel}/task.toml: [task].id missing")
            task_id = task_dir.name  # for downstream cross-checks
        elif task_id != task_dir.name:
            problems.append(
                f"{rel}/task.toml: [task].id={task_id!r} != directory name "
                f"{task_dir.name!r}"
            )

        tax = cfg.get("taxonomy", {}) or {}
        scenario = tax.get("scenario")
        difficulty = tax.get("difficulty")
        caps = tax.get("capabilities", []) or []
        if not isinstance(caps, list) or not (1 <= len(caps) <= 15):
            problems.append(
                f"{rel}/task.toml: [taxonomy].capabilities must be a list of "
                f"1-15 tags (got {len(caps) if isinstance(caps, list) else type(caps).__name__})"
            )

        env = cfg.get("environment", {}) or {}
        if not env.get("image"):
            problems.append(f"{rel}/task.toml: [environment].image is required")

        req = cfg.get("required_outputs", {}) or {}
        files = req.get("files", []) or []
        if not isinstance(files, list) or "outcome.json" not in files:
            problems.append(
                f"{rel}/task.toml: [required_outputs].files must include 'outcome.json'"
            )

        # schema.json is valid JSON
        schema_path = task_dir / "schema.json"
        if schema_path.is_file():
            _, err = _load_json(schema_path)
            if err:
                problems.append(f"{rel}/schema.json: invalid JSON ({err})")

        # register for cross-checks
        tasks_on_disk[task_id] = {
            "scenario": scenario,
            "difficulty": difficulty,
            "task_dir": task_dir,
        }

    # groundtruth files
    problems.extend(_check_groundtruth(root, tasks_on_disk))

    # registry ↔ tasks/ two-way consistency
    problems.extend(_check_registries(root, tasks_on_disk))

    return problems


def _check_groundtruth(root: Path, tasks_on_disk: dict[str, dict]) -> list[str]:
    problems: list[str] = []
    for task_id, info in tasks_on_disk.items():
        gt_path = root / "groundtruth" / "public" / f"{task_id}.json"
        if not gt_path.is_file():
            problems.append(
                f"groundtruth/public/{task_id}.json: missing (every task needs a "
                f"public groundtruth per TASK_AUTHORING.md §二)"
            )
            continue
        gt, err = _load_json(gt_path)
        if err:
            problems.append(f"groundtruth/public/{task_id}.json: invalid JSON ({err})")
            continue

        # key-set consistency
        weights = gt.get("weights", {}) or {}
        scoring = gt.get("scoring", {}) or {}
        values = gt.get("values", {}) or {}
        values_flat = _flatten_dotted(values)
        # Every weight key must resolve to either a scoring rule or a value.
        for key in weights:
            if key not in scoring and key not in values_flat:
                problems.append(
                    f"groundtruth/public/{task_id}.json: weight key {key!r} has "
                    f"no matching scoring rule or value"
                )
        # A scored field with no weight would score 0 and just be dead — flag it.
        for key in scoring:
            if key not in weights:
                problems.append(
                    f"groundtruth/public/{task_id}.json: scoring rule {key!r} "
                    f"has no weight in [weights] — it would be silently ignored"
                )
    return problems


def _flatten_dotted(obj, prefix: str = "") -> set[str]:
    """Turn nested ``{"part1": {"total": 1}}`` into ``{"part1.total"}`` so it
    can be compared against dotted weight keys."""
    out: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            nested = _flatten_dotted(v, f"{prefix}{k}.")
            if nested:
                out.update(nested)
            out.add(f"{prefix}{k}")
    return out


def _check_registries(root: Path, tasks_on_disk: dict[str, dict]) -> list[str]:
    problems: list[str] = []
    reg_dir = root / "registry"
    if not reg_dir.is_dir():
        problems.append("registry/: directory missing")
        return problems

    registries = sorted(reg_dir.glob("*.json"))
    if not registries:
        problems.append("registry/: no *.json files")
        return problems

    union_of_registry_task_ids: set[str] = set()
    for reg_path in registries:
        reg, err = _load_json(reg_path)
        if err:
            problems.append(f"{_rel(root, reg_path)}: invalid JSON ({err})")
            continue

        entries = reg.get("tasks", []) or []
        listed_ids: set[str] = set()
        counts_by_difficulty: dict[str, int] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                problems.append(f"{_rel(root, reg_path)}: tasks[] contains a non-object entry")
                continue
            tid = entry.get("task_id")
            if not tid:
                problems.append(f"{_rel(root, reg_path)}: an entry is missing task_id")
                continue
            listed_ids.add(tid)
            union_of_registry_task_ids.add(tid)
            counts_by_difficulty[entry.get("difficulty", "?")] = (
                counts_by_difficulty.get(entry.get("difficulty", "?"), 0) + 1
            )

            info = tasks_on_disk.get(tid)
            if info is None:
                problems.append(
                    f"{_rel(root, reg_path)}: entry {tid!r} has no matching "
                    f"tasks/*/{tid}/ directory on disk"
                )
                continue
            # scenario / difficulty agreement is enforced by taxonomy.validate;
            # here we only check the disk path claim.
            claimed = entry.get("path")
            if claimed:
                actual = _rel(root, info["task_dir"])
                if claimed != actual:
                    problems.append(
                        f"{_rel(root, reg_path)}: entry {tid!r} claims path "
                        f"{claimed!r} but task actually lives at {actual!r}"
                    )

        # difficulty_distribution honesty (this was the specific drift the
        # honesty check this catches automatically).
        declared = reg.get("difficulty_distribution")
        if isinstance(declared, dict):
            # keys we care about are only the DIFFICULTIES vocabulary
            declared_norm = {k: int(declared.get(k, 0)) for k in DIFFICULTIES}
            actual_norm = {k: counts_by_difficulty.get(k, 0) for k in DIFFICULTIES}
            if declared_norm != actual_norm:
                problems.append(
                    f"{_rel(root, reg_path)}: difficulty_distribution "
                    f"{declared_norm} disagrees with counts of listed tasks "
                    f"{actual_norm}"
                )

    # every task on disk must appear in at least one registry, else it will
    # never be selected into a round and there is no reason for it to exist.
    for tid in tasks_on_disk:
        if tid not in union_of_registry_task_ids:
            problems.append(
                f"tasks/*/{tid}/: on disk but not listed in any registry/*.json "
                f"— a task that no registry declares is unreachable"
            )
    return problems


# ---------------------------------------------------------------------------
# Content checks (L2)
# ---------------------------------------------------------------------------

def check_content(root: Path) -> list[str]:
    problems: list[str] = []
    for task_dir in _iter_task_dirs(root):
        rel = _rel(root, task_dir)
        cfg, err = _load_toml(task_dir / "task.toml")
        if err:
            # structure check already flagged this; skip silently in content.
            continue

        # 1. scenario / difficulty / directory / registry alignment
        for w in validate_task_taxonomy(task_dir, cfg):
            problems.append(f"{rel}: {w}")

        # 2. capabilities ⊆ CAPABILITIES
        caps = (cfg.get("taxonomy", {}) or {}).get("capabilities", []) or []
        for cap in caps:
            if cap not in CAPABILITIES:
                problems.append(
                    f"{rel}/task.toml: capability {cap!r} is not in the "
                    f"controlled vocabulary — see TASK_AUTHORING.md §七 / "
                    f"framework/taxonomy.py::CAPABILITIES"
                )

        # 3. groundtruth checks
        task_id = (cfg.get("task", {}) or {}).get("id", task_dir.name)
        gt_path = root / "groundtruth" / "public" / f"{task_id}.json"
        if gt_path.is_file():
            gt, gt_err = _load_json(gt_path)
            if gt_err or not isinstance(gt, dict):
                continue
            problems.extend(_check_groundtruth_content(root, gt_path, gt))

    return problems


def _check_groundtruth_content(root: Path, gt_path: Path, gt: dict) -> list[str]:
    problems: list[str] = []
    rel = _rel(root, gt_path)

    # capability_map values ⊆ CAPABILITIES
    cap_map = gt.get("capability_map", {}) or {}
    for field_key, caps in cap_map.items():
        if not isinstance(caps, list):
            problems.append(
                f"{rel}: capability_map[{field_key!r}] must be a list, "
                f"got {type(caps).__name__}"
            )
            continue
        for cap in caps:
            if cap not in CAPABILITIES:
                problems.append(
                    f"{rel}: capability_map[{field_key!r}] contains "
                    f"{cap!r}, which is not in the controlled vocabulary "
                    f"(§七 / CAPABILITIES)"
                )

    # scoring rule types are known
    scoring = gt.get("scoring", {}) or {}
    for field_key, rule in scoring.items():
        if not isinstance(rule, dict):
            problems.append(f"{rel}: scoring[{field_key!r}] must be an object")
            continue
        rtype = rule.get("type")
        if rtype not in _KNOWN_RULE_TYPES:
            problems.append(
                f"{rel}: scoring[{field_key!r}].type={rtype!r} is not "
                f"implemented in scorer.py (allowed: {sorted(_KNOWN_RULE_TYPES)})"
            )

    # veto sanity — reuse the loader used by aggregate.py so behaviour matches.
    try:
        from framework.aggregate import validate_veto  # local import: heavy module
        validate_veto(gt)
    except ValueError as exc:
        problems.append(f"{rel}: {exc}")
    except Exception as exc:  # pragma: no cover - aggregate import shouldn't crash
        problems.append(f"{rel}: validate_veto crashed: {exc!r}")

    return problems


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _root_from_here() -> Path:
    return Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python3 -m framework.ci_validate",
        description="Offline structure + content validator for tasks/, "
                    "groundtruth/ and registry/. Used by CI's L1 + L2 gates.",
    )
    ap.add_argument("--structure", action="store_true",
                    help="Run L1 structure checks (files, TOML, JSON, "
                         "registry↔tasks/ consistency).")
    ap.add_argument("--content", action="store_true",
                    help="Run L2 content checks (taxonomy whitelist, "
                         "capability vocabulary, veto, rule types).")
    ap.add_argument("--root", type=Path, default=None,
                    help="Repo root; defaults to the parent of framework/.")
    args = ap.parse_args(argv)

    if not (args.structure or args.content):
        ap.error("pick at least one of --structure / --content")

    root = args.root or _root_from_here()
    problems: list[str] = []
    ran: list[str] = []

    if args.structure:
        ran.append("--structure")
        problems.extend(check_structure(root))
    if args.content:
        ran.append("--content")
        problems.extend(check_content(root))

    if problems:
        for p in problems:
            print(f"::error::{p}")
        print(
            f"\nFAILED — {len(problems)} problem(s) across {' '.join(ran)}",
            file=sys.stderr,
        )
        return 1

    print(f"OK — {' '.join(ran)} passed at {root}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
