"""
framework/runner.py
===================
Single-task execution entry point.

Wraps: workspace prep → agent run → factoid score → optional pytest → merged report.

Design sources
--------------
- 159's run_eval.py: workspace symlink, trajectory capture, report generation
- Harbor's trial abstraction: one trial = one (task × agent × model × seed)
- SkillsBench's oracle-must-pass rule: --check-oracle flag re-scores solution/solve.py

Usage
-----
    # 1. Verify oracle passes (CI gate)
    python framework/runner.py \\
        --task tasks/social_stats/159_gaokao_reform \\
        --agent oracle \\
        --check-oracle

    # 2. Run one model
    python framework/runner.py \\
        --task tasks/social_stats/159_gaokao_reform \\
        --model claude-sonnet-4.5 \\
        --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from framework.agent_adapter import HttpApiAgent, SubprocessExecutor, load_model_config
from framework.docker_executor import DockerExecutor, check_docker_available, image_exists
from framework.scorer import score_task, validate_groundtruth_consistency
from framework.verifier_wrapper import combined_score, evaluate_veto, run_pytest


ROOT = Path(__file__).resolve().parent.parent


try:
    import tomllib as _tomllib  # Python 3.11+
except ModuleNotFoundError:  # 3.10 and earlier
    import tomli as _tomllib  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# task.toml single source of truth
# ---------------------------------------------------------------------------
def load_task_config(task_dir: Path) -> dict:
    """Load task.toml and return the merged configuration dict.

    Returns a flat dict with keys like 'max_turns', 'timeout_per_step',
    'memory_gb', 'cpus', 'image', 'pass_threshold', 'required_outputs',
    'wall_timeout'.
    All runner code should read from this instead of hardcoding values.

    v4 (魏-4): ``wall_timeout`` is now surfaced and consumed by the runner
    as a real wall-clock watchdog. If not specified, defaults to
    max_turns × timeout_per_step + 300 (generous safety margin).
    """
    with open(task_dir / "task.toml", "rb") as f:
        cfg = _tomllib.load(f)
    resources = cfg.get("resources", {})
    scoring = cfg.get("scoring", {})
    env = cfg.get("environment", {})
    max_turns = int(resources.get("max_turns", 50))
    timeout_per_step = int(resources.get("timeout_per_step", 180))
    # v4 (魏-4): wall_timeout replaces the old ghost "timeout" field.
    # Falls back to timeout (legacy) then derived default.
    wall_timeout = int(resources.get("wall_timeout",
                       resources.get("timeout",
                                     max_turns * timeout_per_step + 300)))
    return {
        "max_turns": max_turns,
        "timeout_per_step": timeout_per_step,
        "memory_gb": int(resources.get("memory_gb", 8)),
        "cpus": int(resources.get("cpus", 4)),
        "image": env.get("image"),
        "pass_threshold": float(scoring.get("pass_threshold", 0.5)),
        "required_outputs": list(cfg.get("required_outputs", {}).get("files", [])),
        "wall_timeout": wall_timeout,
    }


# ---------------------------------------------------------------------------
# Workspace prep
# ---------------------------------------------------------------------------
def cleanup_workspace(workspace: Path, *, keep_full: bool = False) -> None:
    """Delete bulky input data after scoring — keeps outputs + trajectory.

    Kept:    outcome.json, output/*, instruction.md, schema.json (small)
    Removed: workspace/environment/  (tens of MB per trial)
    Skipped if --keep-full-workspace was passed.
    """
    if keep_full:
        return
    env_dir = workspace / "environment"
    if not env_dir.exists() or env_dir.is_symlink():
        return
    # Restore write permission before rmtree; otherwise chmod 0o444 blocks it.
    for p in env_dir.rglob("*"):
        if p.is_file() and not p.is_symlink():
            try:
                p.chmod(0o644)
            except OSError:
                pass
    try:
        shutil.rmtree(env_dir)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Human-readable eval report
# ---------------------------------------------------------------------------
def _read_task_title(task_dir: Path) -> str:
    try:
        with open(task_dir / "task.toml", "rb") as f:
            return _tomllib.load(f)["task"].get("title", task_dir.name)
    except Exception:
        return task_dir.name


def generate_report(
    *,
    trial_root: Path,
    task_dir: Path,
    workspace: Path,
    agent: str,
    model: str | None,
    api_base: str | None,
    seed: int,
    docker_image: str | None,
    duration_sec: float,
    score_result: dict,
    trajectory_path: Path | None,
) -> Path:
    """Write a human-readable eval_report.txt inspired by 007's PEV-Eval format.

    Sections:
      1. Metadata / summary
      2. Agent-generated code (extracted from trajectory.jsonl)
      3. Output files check
      4. Score breakdown (combined + factoid + pytest)
      5. Per-capability radar
      6. Per-field details (failed ones highlighted)
    """
    from datetime import datetime as _dt

    title = _read_task_title(task_dir)
    lines: list[str] = []

    def header(text: str) -> None:
        lines.extend(["", "=" * 70, f"  {text}", "=" * 70, ""])

    combined = score_result.get("combined_score", 0.0)
    factoid = score_result.get("factoid_score", 0.0)
    pytest_score = score_result.get("pytest_score")
    factoid_alpha = score_result.get("factoid_alpha", 1.0)

    # -----------------------------
    # 1. HEADER
    # -----------------------------
    lines.extend([
        "=" * 70,
        f"  Bench Eval Report  —  {title}",
        "=" * 70,
        f"  Time:          {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Task ID:       {task_dir.name}",
        f"  Agent:         {agent}",
        f"  Model:         {model or '(oracle)'}",
        f"  API Base:      {api_base or 'N/A'}",
        f"  Seed:          {seed}",
        f"  Docker Image:  {docker_image or '(host mode)'}",
        f"  Workspace:     {workspace}",
        f"  Duration:      {duration_sec:.1f}s",
        "",
        f"  >>> Combined Score:  {combined:.4f}  ({combined * 100:.2f}%)",
        f"      Factoid:         {factoid:.4f}  (weight α = {factoid_alpha})",
    ])
    if pytest_score is not None:
        pytd = score_result.get("pytest_details", {})
        lines.append(
            f"      Pytest:          {pytest_score:.4f}  "
            f"(passed {pytd.get('passed', 0)}/"
            f"{pytd.get('passed', 0) + pytd.get('failed', 0) + pytd.get('skipped', 0) + pytd.get('errored', 0)})"
        )

    # -----------------------------
    # 2. AGENT-GENERATED CODE
    # -----------------------------
    header("1. AGENT GENERATED CODE")
    code_blocks_dumped = 0
    if trajectory_path and trajectory_path.exists():
        import json as _json
        import re

        with open(trajectory_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = _json.loads(line)
                except Exception:
                    continue
                if entry.get("role") != "assistant":
                    continue
                content = entry.get("content", "")
                blocks = re.findall(r"```python\s*\n(.*?)```", content, re.DOTALL)
                for block in blocks:
                    code_blocks_dumped += 1
                    lines.append(f"--- Code Block #{code_blocks_dumped} ---")
                    lines.append(block.strip())
                    lines.append("")
    if code_blocks_dumped == 0:
        lines.append("(no code blocks captured — see trajectory.jsonl)")

    # -----------------------------
    # 3. OUTPUT FILES
    # -----------------------------
    header("2. OUTPUT FILES")
    # v2 (#20, 李-4): drive the check purely from task.toml required_outputs.
    # No hardcoded baseline — 131 doesn't declare predictions.csv, so the old
    # code printed a bogus [MISSING] for every 131 report.
    check_paths: list[str] = []
    try:
        with open(task_dir / "task.toml", "rb") as f:
            cfg = _tomllib.load(f)
        check_paths = list(dict.fromkeys(
            cfg.get("required_outputs", {}).get("files", [])
        ))
    except Exception:
        pass
    if not check_paths:
        # If no explicit list, still verify the canonical outcome.json exists
        # (unless the task is pytest-only, in which case leaving this empty
        # is the correct answer).
        check_paths = ["outcome.json"]

    for rel in check_paths:
        p = workspace / rel
        if p.exists():
            size = p.stat().st_size
            lines.append(f"  [FOUND    ] {rel}  ({size} bytes)")
        else:
            lines.append(f"  [MISSING  ] {rel}")

    # v3 (吴-5 / 魏-5): surface missing_outputs and the run's status so
    # per-artifact scoring is visible in the report.
    missing_outputs = score_result.get("missing_outputs") or []
    status = score_result.get("status")
    if missing_outputs:
        lines.append("")
        lines.append(f"  Missing / empty required outputs ({len(missing_outputs)}):")
        for rel in missing_outputs:
            lines.append(f"    - {rel}")
    if status and status != "ok":
        lines.append("")
        lines.append(f"  Run status: {status}")

    # -----------------------------
    # 4. FACTOID BREAKDOWN (per-capability)
    # -----------------------------
    header("3. CAPABILITY BREAKDOWN (per-capability score)")
    fd = score_result.get("factoid_details", {})
    per_cap = fd.get("per_capability", {})
    if per_cap:
        max_name = max(len(k) for k in per_cap) if per_cap else 0
        for cap, s in sorted(per_cap.items(), key=lambda kv: -kv[1]):
            bar = "█" * int(round(s * 20))
            lines.append(f"  {cap:<{max_name}}  {s:.4f}  {bar}")
    else:
        lines.append("  (no capability data)")

    # -----------------------------
    # 5. PER-FIELD DETAILS
    # -----------------------------
    header("4. PER-FIELD DETAILS")
    details = fd.get("details", [])
    if not details:
        lines.append("  (no per-field details available)")
    else:
        lines.append(f"  {'Field':<50}  {'Score':>7}  {'Weight':>7}  Reason")
        lines.append(f"  {'-' * 50}  {'-' * 7}  {'-' * 7}  {'-' * 40}")
        for d in details:
            marker = " " if d["score"] >= 0.99 else ("~" if d["score"] > 0 else "X")
            lines.append(
                f"  {marker} {d['field']:<48}  "
                f"{d['score']:>7.4f}  {d['weight']:>7.1f}  {d['reason']}"
            )

    # -----------------------------
    # 6. PYTEST DETAILS
    # -----------------------------
    if pytest_score is not None:
        header("5. PYTEST DETAILS")
        pytd = score_result.get("pytest_details", {})
        lines.append(
            f"  Passed:   {pytd.get('passed', 0)}  "
            f"Failed: {pytd.get('failed', 0)}  "
            f"Skipped: {pytd.get('skipped', 0)}  "
            f"Errored: {pytd.get('errored', 0)}"
        )
        lines.append(
            f"  Weight:   {pytd.get('earned_weight', 0)} / {pytd.get('total_weight', 0)}"
        )
        # v3 (李-2): per-test roster.
        per_test = pytd.get("per_test") or []
        if per_test:
            lines.append("")
            lines.append(f"  {'Test':<60}  Status")
            lines.append(f"  {'-' * 60}  {'-' * 8}")
            for t in per_test:
                name = t.get("name", "?")
                status = t.get("status", "?")
                marker = " " if status == "PASSED" else ("X" if status == "FAILED" else "~")
                lines.append(f"  {marker} {name:<58}  {status}")
        # v3 (吴-1): weight-map health.
        unmatched = pytd.get("unmatched_names") or []
        unused = pytd.get("unused_weight_keys") or []
        if unmatched:
            lines.append("")
            lines.append(f"  [!] {len(unmatched)} test(s) had no matching weight_map entry:")
            for n in unmatched:
                lines.append(f"      - {n}")
        if unused:
            lines.append(f"  [!] {len(unused)} weight_map key(s) matched no test:")
            for k in unused:
                lines.append(f"      - {k}")

    # -----------------------------
    # 7. FOOTER
    # -----------------------------
    header("Summary")
    lines.append(f"  Final combined score: {combined:.4f}  ({combined * 100:.2f}%)")
    lines.append(f"  Details JSON:         {trial_root / 'score.json'}")
    if trajectory_path:
        lines.append(f"  Trajectory:           {trajectory_path}")
    lines.append("")

    out_path = trial_root / "eval_report.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def prepare_workspace(task_dir: Path, trial_root: Path) -> Path:
    """Create trial workspace with data + instruction copied read-only.

    Data protection:
      - Host mode (no Docker): environment/ files are COPIED into the workspace
        and chmod'ed to 0o444 (read-only). Prevents the agent from writing
        through to the source data — a real risk with symlinks.
      - Docker mode: the runner will still bind-mount env with `:ro`; the copy
        here is redundant but harmless.
    """
    workspace = trial_root / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    # Copy instruction & schema (writable — agent might inspect but not modify)
    for name in ("instruction.md", "schema.json"):
        src = task_dir / name
        if src.exists():
            shutil.copy2(src, workspace / name)

    # Copy environment/ deeply, follow symlinks (so we get real files, not links
    # that could tunnel to the source), then chmod every file read-only.
    env_src = task_dir / "environment"
    env_dst = workspace / "environment"
    if env_src.exists() and not env_dst.exists():
        # symlinks=False → resolve symlinks into real copies
        shutil.copytree(env_src, env_dst, symlinks=False)
        for p in env_dst.rglob("*"):
            if p.is_file() and not p.is_symlink():
                try:
                    p.chmod(0o444)
                except OSError:
                    pass

    return workspace


# ---------------------------------------------------------------------------
# Agent invocation
# ---------------------------------------------------------------------------
def run_oracle(task_dir: Path, workspace: Path, *, image: str | None = None) -> None:
    """Execute solution/solve.py inside workspace.

    If `image` is provided, runs inside a Docker container; otherwise runs on
    the host directly.
    """
    solve = task_dir / "solution" / "solve.py"
    if not solve.exists():
        raise FileNotFoundError(f"Oracle not found: {solve}")

    if image:
        # Docker mode: mount the whole task_dir (for solve.py + environment/)
        # and workspace (for outputs).
        cmd = [
            "docker", "run", "--rm",
            "--network", "none",
            "--cap-drop=ALL",
            "--security-opt", "no-new-privileges",
            "--pids-limit", "256",
            "--memory", "8g", "--cpus", "4",
            "-v", f"{workspace.resolve()}:/workspace",
            "-v", f"{(task_dir / 'solution').resolve()}:/task_solution:ro",
            "-v", f"{(task_dir / 'environment').resolve()}:/workspace/environment:ro",
            "-w", "/workspace",
            image,
            "python", "/task_solution/solve.py",
        ]
    else:
        cmd = [sys.executable, str(solve)]

    proc = subprocess.run(
        cmd,
        cwd=str(workspace) if not image else None,
        capture_output=True,
        text=True,
        timeout=900,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Oracle failed:\n{proc.stdout}\n{proc.stderr}")


def run_agent(
    task_dir: Path,
    workspace: Path,
    model_name: str,
    models_yaml: Path,
    *,
    image: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    model_id: str | None = None,
    trial_root: Path | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    """Run HttpApiAgent for the given model.

    If `image` is provided, agent's Python code executions happen inside a
    Docker container (via DockerExecutor). Otherwise SubprocessExecutor runs
    code on the host.

    The agent process itself (LLM API calls) always runs on the host — API
    keys never enter the container.

    If api_base + api_key are provided, they override the models.yaml lookup.
    Useful for ad-hoc testing with a specific endpoint without editing config.

    Resource limits (max_turns / timeout_per_step / memory_gb / cpus) are read
    from `task.toml` via `load_task_config()` — the runner should never
    hardcode these, so different tasks can tune their own budgets.

    v2:
      * #4  ``max_tokens`` (CLI-driven) overrides models.yaml when given.
      * #5  trajectory streams to disk incrementally.
      * #6  exec logs (code + full output) persist under ``<trial>/exec_logs``.
      * #13 returns ``terminated_by`` for score.json._meta.

    v3 (王-1): ``temperature`` (CLI-driven) overrides models.yaml when given.
        Needed for vendors like Kimi-K2.6 whose gateway rejects temperature=0.
    """
    if api_base and api_key:
        from framework.agent_adapter import ModelConfig
        cfg = ModelConfig(
            name=model_name,
            api_base=api_base,
            api_key=api_key,
            model_id=model_id or model_name,
        )
    else:
        cfg = load_model_config(models_yaml, model_name)
    if max_tokens is not None:
        cfg.max_tokens = max_tokens  # CLI override wins over models.yaml (v2 #4)
    if temperature is not None:
        cfg.temperature = float(temperature)  # v3 (王-1)

    task_cfg = load_task_config(task_dir)
    max_turns = task_cfg["max_turns"]
    timeout_per_step = task_cfg["timeout_per_step"]
    memory = f"{task_cfg['memory_gb']}g"
    cpus = str(task_cfg["cpus"])

    # v2 (#6): persistent per-block code + output logs live alongside the trial.
    exec_log_dir = (trial_root / "exec_logs") if trial_root is not None else None

    executor: SubprocessExecutor | DockerExecutor
    if image:
        executor = DockerExecutor(
            image=image,
            workspace=workspace,
            data_dir=task_dir / "environment",
            memory=memory,
            cpus=cpus,
            timeout_per_step=timeout_per_step,
            allow_network=False,
            log_dir=exec_log_dir,
        )
    else:
        executor = SubprocessExecutor(
            workspace,
            timeout=timeout_per_step,
            log_dir=exec_log_dir,
        )

    # v2 (#5): stream trajectory to disk so mid-run crashes still leave a trail.
    trajectory_path = (trial_root / "trajectory.jsonl") if trial_root is not None else None

    agent = HttpApiAgent(
        cfg,
        max_turns=max_turns,
        executor=executor,
        trajectory_path=trajectory_path,
    )
    agent.setup(task_dir, workspace)
    terminated_by = "unknown"
    try:
        terminated_by = agent.run() or "unknown"
    finally:
        agent.teardown()
    # If streaming was on, dump_trajectory is a no-op; otherwise this is the
    # backstop that keeps pre-v2 callers working.
    if trajectory_path is None:
        agent.dump_trajectory(workspace.parent / "trajectory.jsonl")
    # Stash side-channel info for the caller (score.json._meta).
    run_agent.last_terminated_by = terminated_by  # type: ignore[attr-defined]
    run_agent.last_usage_summary = dict(agent.usage_summary)  # type: ignore[attr-defined]
    return terminated_by


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
# v2 (#18): status vocabulary carried in every score.json.  aggregate.py filters
# on this to separate ability signal from infrastructure noise (see #28).
# v3 additions:
#   * STATUS_EXECUTOR_CRASH (李-5): host/docker executor blew up mid-block —
#     infra failure, not a model signal.
#   * STATUS_NO_PROGRESS / STATUS_STUCK_LOOP (侯-3): circuit-breaker signals —
#     model failed to advance or emitted identical replies. Model_fail bucket.
STATUS_OK = "ok"
STATUS_API_ERROR = "api_error"
STATUS_NO_OUTPUT = "no_output"
STATUS_BAD_JSON = "bad_json"
STATUS_MAX_TURNS = "max_turns"
STATUS_SCORER_ERROR = "scorer_error"
STATUS_EXECUTOR_CRASH = "executor_crash"
STATUS_NO_PROGRESS = "no_progress"
STATUS_STUCK_LOOP = "stuck_loop"


def score_run(
    task_id: str,
    workspace: Path,
    task_dir: Path,
    groundtruth_dir: Path,
    *,
    task_cfg: dict | None = None,
) -> dict:
    """Score a completed run: factoid + optional pytest.

    v2 changes:
      * #15 罗-7: outcome.json is only required when ``factoid_alpha > 0``;
        artifact-only tasks (pytest layer alone) validate ``required_outputs.files``.
      * #16 罗-5: scorer exceptions are caught and reported as ``scorer_error``
        rather than aborting the runner.
      * #18: every return path stamps a ``status`` field (STATUS_*).

    v3 (吴-5 / 魏-5): missing / empty required_outputs no longer zero the
    entire trial. Only outcome.json (when the factoid layer is active) forces
    an early return; other missing artifacts drop their dependent tests to 0
    via the pytest layer and are surfaced under ``missing_outputs``.
    """
    gt_path = groundtruth_dir / "public" / f"{task_id}.json"
    if not gt_path.exists():
        raise FileNotFoundError(f"Groundtruth missing: {gt_path}")

    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)
    factoid_alpha = float(gt.get("factoid_alpha", 0.9))
    pytest_weights = gt.get("pytest_weight_map", {})

    if task_cfg is None:
        try:
            task_cfg = load_task_config(task_dir)
        except Exception:
            task_cfg = {}
    required_outputs: list[str] = list(task_cfg.get("required_outputs", []) or [])
    if factoid_alpha > 0 and "outcome.json" not in required_outputs:
        required_outputs.append("outcome.json")

    # v2 (#15): validate every declared required output up front.
    missing: list[str] = []
    empty: list[str] = []
    for rel in required_outputs:
        p = workspace / rel
        if not p.exists():
            missing.append(rel)
        elif p.is_file() and p.stat().st_size == 0:
            empty.append(rel)

    answer_path = workspace / "outcome.json"

    # v3 (吴-5 / 魏-5): only outcome.json (with factoid active) forces an
    # early return; other missing outputs are recorded and scoring proceeds
    # so pytest can grade the artifacts that DO exist.
    missing_outputs = sorted(set(missing) | set(empty))
    outcome_missing = ("outcome.json" in missing) or ("outcome.json" in empty)
    if factoid_alpha > 0 and outcome_missing:
        return {
            "combined_score": 0.0,
            "factoid_score": 0.0,
            "status": STATUS_NO_OUTPUT,
            "missing_outputs": missing_outputs,
            "error": (
                "outcome.json missing/empty; "
                f"required outputs missing/empty: {missing_outputs}"
            ),
        }

    # If factoid layer is active, outcome.json must parse as JSON.
    # v3.3 (Luo-4): tolerate bare NaN/Infinity literals by mapping them to null
    # (via parse_constant) instead of treating a mostly-correct file with one
    # non-standard numeric token as an unparseable failure. Genuinely malformed
    # structure still raises JSONDecodeError and keeps the bad_json status.
    if factoid_alpha > 0 and answer_path.exists():
        try:
            raw = answer_path.read_text(encoding="utf-8")
            json.loads(raw, parse_constant=lambda _c: None)
        except (OSError, json.JSONDecodeError) as exc:
            return {
                "combined_score": 0.0,
                "factoid_score": 0.0,
                "status": STATUS_BAD_JSON,
                "missing_outputs": missing_outputs,
                "error": f"outcome.json unparseable: {exc}",
            }

    # v2 (#16): guard the whole scoring block so a scorer bug becomes a
    # labeled 0, not a runner crash.
    try:
        if factoid_alpha > 0:
            factoid_result = score_task(answer_path=answer_path, groundtruth_path=gt_path)
        else:
            # Pytest-only task — synthesize a zero-scored factoid result.
            from framework.scorer import ScoringResult
            factoid_result = ScoringResult(
                total_score=0.0, total_weight=0.0, earned_weight=0.0,
                per_field={}, per_capability={}, details=[],
            )

        # Optional pytest for artifact checks
        test_dir = task_dir / "tests"
        pytest_res = run_pytest(test_dir, workspace) if test_dir.exists() else None

        if pytest_res is not None:
            # v4 (吴-1): always route through combined_score() so its internal
            # three-branch logic (scored / no_weight_map / pytest_broken) is
            # exercised. Pre-v4, empty per_test bypassed combined_score entirely,
            # making the pytest_broken branch dead code.
            result = combined_score(
                factoid_result=factoid_result.to_dict(),
                pytest_result=pytest_res,
                pytest_weight_map=pytest_weights,
                factoid_alpha=factoid_alpha,
                groundtruth=gt,
            )
        else:
            result = {
                "combined_score": round(factoid_result.total_score, 4),
                "factoid_score": round(factoid_result.total_score, 4),
                "factoid_details": factoid_result.to_dict(),
                "pytest_score": None,
            }
            result["veto_status"] = evaluate_veto(score_json=result, groundtruth=gt)
    except Exception as exc:
        return {
            "combined_score": 0.0,
            "factoid_score": 0.0,
            "status": STATUS_SCORER_ERROR,
            "missing_outputs": missing_outputs,
            "error": f"{type(exc).__name__}: {exc}",
        }

    result["status"] = STATUS_OK
    # v3 (吴-5 / 魏-5): always stamp missing_outputs so reports can surface
    # partial-artifact runs even when the factoid layer produced a score.
    result["missing_outputs"] = missing_outputs
    return result


# ---------------------------------------------------------------------------
# CI: oracle must pass
# ---------------------------------------------------------------------------
def _write_fail_score(trial_root: Path, status: str, error: str) -> None:
    """v4 (侯-2): persist a minimal score.json for early-exit failures so that
    the trial directory is never empty on disk."""
    import json as _json
    payload = {
        "combined_score": 0.0,
        "factoid_score": 0.0,
        "status": status,
        "error": error,
    }
    try:
        (trial_root / "score.json").write_text(
            _json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError:
        pass


def check_oracle(
    task_dir: Path,
    groundtruth_dir: Path,
    trial_root: Path,
    *,
    image: str | None = None,
) -> bool:
    """Sanity: solution/solve.py must score >= 0.99 (i.e., near-perfect).

    v2 (#30, 魏-8): also cross-checks that ``values`` and ``scoring[k].gt`` agree.
    """
    workspace = prepare_workspace(task_dir, trial_root)
    run_oracle(task_dir, workspace, image=image)
    task_id = _read_task_id(task_dir)

    # v2 (#30): fail loudly on values / scoring drift before scoring proceeds.
    gt_path = groundtruth_dir / "public" / f"{task_id}.json"
    if gt_path.exists():
        try:
            with open(gt_path, "r", encoding="utf-8") as f:
                gt = json.load(f)
            problems = validate_groundtruth_consistency(gt)
            if problems:
                print("[oracle-check] groundtruth values ≠ scoring.gt:")
                for p in problems:
                    print(f"  - {p}")
                return False
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[oracle-check] could not read groundtruth: {exc}")

    result = score_run(task_id, workspace, task_dir, groundtruth_dir)
    score = result.get("combined_score", 0.0)
    print(f"[oracle-check] task={task_id} score={score:.4f} status={result.get('status')}")
    return score >= 0.99


def _read_task_id(task_dir: Path) -> str:
    with open(task_dir / "task.toml", "rb") as f:
        return _tomllib.load(f)["task"]["id"]


def _read_task_image(task_dir: Path) -> str | None:
    """Read the domain image tag from task.toml [environment].image."""
    with open(task_dir / "task.toml", "rb") as f:
        cfg = _tomllib.load(f)
    return cfg.get("environment", {}).get("image")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="Task directory")
    parser.add_argument("--groundtruth-dir", default=str(ROOT / "groundtruth"))
    parser.add_argument("--models-yaml", default=str(ROOT / "configs" / "models.yaml"))
    parser.add_argument("--agent", default="http_api", choices=["http_api", "oracle"])
    parser.add_argument("--model", default=None, help="Model name (for http_api agent)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-oracle", action="store_true", help="Just verify oracle scores 1.0")
    parser.add_argument("--use-docker", action="store_true",
                        help="Run agent code inside a Docker container. Image comes from task.toml [environment].image")
    parser.add_argument("--image", default=None,
                        help="Override the Docker image (takes precedence over task.toml)")
    parser.add_argument("--api-base", default=None,
                        help="Override LLM API base URL (bypasses models.yaml)")
    parser.add_argument("--api-key", default=None,
                        help="Override LLM API key (bypasses models.yaml)")
    parser.add_argument("--model-id", default=None,
                        help="Override model identifier sent to the API (defaults to --model)")
    parser.add_argument("--keep-full-workspace", action="store_true",
                        help="Keep the full workspace including copied input data (default: cleanup)")
    parser.add_argument("--allow-host-mode", action="store_true",
                        help="Allow http_api agent to run on the host without Docker. "
                             "STRONGLY DISCOURAGED for real evaluation: agent code can read "
                             "groundtruth/, tests/, and other repo files. Use only for local debug.")
    parser.add_argument(
        "--trial-tag",
        default=None,
        help="v3.3 (Hou-3 / Wei-3): stamp 'bench.trial=<tag>' on every docker "
             "container so batch.py can reap THIS exact container after a "
             "batch-level timeout, instead of leaking it.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Override the LLM max_tokens (default: models.yaml / vendor default)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override the LLM temperature (v3 王-1). "
                             "Needed for vendors that reject temperature=0.")
    args = parser.parse_args()

    task_dir = Path(args.task).resolve()
    gt_dir = Path(args.groundtruth_dir).resolve()

    # v4 (侯-2): compute trial_root and mkdir BEFORE image/Docker checks,
    # so that early failures leave a trace on disk.
    task_id = _read_task_id(task_dir)
    tag = args.agent if args.agent == "oracle" else f"{args.model}_seed{args.seed}"
    # Resolve Docker image if requested
    image: str | None = None
    if args.use_docker or args.image:
        image = args.image or _read_task_image(task_dir)
        if not image:
            raise SystemExit("--use-docker requires task.toml [environment].image or --image flag")
    if image:
        tag += "_docker"
    trial_root = Path(args.output_dir) if args.output_dir else (
        task_dir / "trials" / f"{datetime.now():%Y%m%d_%H%M%S}_{tag}"
    )
    trial_root.mkdir(parents=True, exist_ok=True)

    # Now do Docker availability/image checks (trial_root already exists for failure recording)
    if image:
        if not check_docker_available():
            _write_fail_score(trial_root, "docker_unavailable", "Docker daemon not reachable")
            raise SystemExit("Docker daemon not reachable. Start Docker Desktop first.")
        if not image_exists(image):
            _write_fail_score(trial_root, "image_missing", f"Image {image!r} not built locally")
            raise SystemExit(
                f"Image {image!r} not built locally.\n"
                f"Run: docker build -f docker/base.Dockerfile -t bench-base:v1 . && "
                f"docker build -f docker/social.Dockerfile -t bench-social:v1 ."
            )
        print(f"[runner] Docker mode: image={image}")

    # Refuse to run the http_api agent on the host by default: nothing prevents
    # LLM-generated code from reading groundtruth/ or tests/. Oracle can still
    # run on the host (trusted, human-written) — and --allow-host-mode is a
    # documented escape hatch for local debug.
    if args.agent == "http_api" and image is None and not args.allow_host_mode:
        raise SystemExit(
            "Refusing to run http_api agent on the host without --use-docker.\n"
            "Reason: agent-generated Python has full filesystem access, so\n"
            "  groundtruth/public/*.json and tests/ are reachable.\n"
            "Either pass --use-docker (recommended), or acknowledge the risk with\n"
            "--allow-host-mode (do NOT use for benchmark numbers)."
        )
    if args.agent == "http_api" and image is None and args.allow_host_mode:
        print("[runner] WARNING: host mode is on. Groundtruth is on the honor system.",
              file=sys.stderr)

    # Refuse to run ANYTHING against an inconsistent groundtruth. Checked here —
    # before the agent/oracle runs — so a broken GT costs zero model time/tokens
    # instead of surfacing as a silent wrong score after a full trial.
    gt_path = gt_dir / "public" / f"{task_id}.json"
    if gt_path.exists():
        with open(gt_path, "r", encoding="utf-8") as f:
            _gt = json.load(f)
        _problems = validate_groundtruth_consistency(_gt)
        if _problems:
            raise SystemExit(
                f"[runner] refusing to run: groundtruth {task_id} is inconsistent:\n  - "
                + "\n  - ".join(_problems)
            )

    if args.check_oracle:
        ok = check_oracle(task_dir, gt_dir, trial_root, image=image)
        sys.exit(0 if ok else 1)

    print(f"[runner] task={task_id} agent={args.agent} model={args.model}")
    workspace = prepare_workspace(task_dir, trial_root)

    t0 = time.time()
    terminated_by = None
    usage_summary = None

    # v4 (魏-4): wall-clock watchdog. If the task declares wall_timeout, we
    # enforce it via signal.alarm (Unix) or threading fallback so a stuck agent
    # cannot run indefinitely and block the batch pool.
    task_cfg = load_task_config(task_dir)
    wall_timeout = task_cfg.get("wall_timeout", 0)
    _watchdog_fired = False

    def _watchdog_handler(*_a):
        nonlocal _watchdog_fired
        _watchdog_fired = True
        raise TimeoutError(f"[runner] wall_timeout={wall_timeout}s exceeded")

    if wall_timeout > 0 and hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _watchdog_handler)
        signal.alarm(wall_timeout)

    # v4 (魏-3): inject seed into random sources so multi-seed runs produce
    # independent samples. Without this, temperature=0 + no seed injection
    # gives identical trials across all seeds, making std≈0 / CI fake-narrow.
    import random as _rng
    _rng.seed(args.seed)
    # v3.3 hotfix: PIN Python hash randomization to a CONSTANT instead of tying it
    # to args.seed. Tying it to the trial-seed (the earlier v4 魏‑3 wiring) made any
    # participant/reference code that iterates dicts/sets produce DIFFERENT results
    # per seed — not sampler noise, but arbitrary algorithmic reshuffling that flipped
    # e.g. causal control-group selection and swung DID estimates wildly across seeds.
    # Hashing is disabled here for cross-trial stability; intended per-seed variance
    # still comes from BENCH_SEED below + numpy/random seeding in executors.
    os.environ["PYTHONHASHSEED"] = "0"
    try:
        import numpy as _np
        _np.random.seed(args.seed)
    except ImportError:
        pass
    # v3.3 (Wei-3 / Hou-3): expose per-trial identity for BOTH executors.
    # SubprocessExecutor inherits these env vars directly; DockerExecutor turns
    # them into `-e ...=` on its long-lived container AND a `--label bench.trial`
    # that batch.py uses to reap exactly this container when it kills our proc.
    os.environ["BENCH_SEED"] = str(args.seed)
    os.environ["BENCH_TRIAL_TAG"] = (
        args.trial_tag if args.trial_tag is not None else trial_root.name
    )

    if args.agent == "oracle":
        run_oracle(task_dir, workspace, image=image)
    else:
        if not args.model:
            raise SystemExit("--model required for http_api agent")
        try:
            run_agent(
                task_dir, workspace, args.model, Path(args.models_yaml),
                image=image,
                api_base=args.api_base,
                api_key=args.api_key,
                model_id=args.model_id,
                trial_root=trial_root,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        except TimeoutError:
            _watchdog_fired = True
            print(f"[runner] WATCHDOG: wall_timeout={wall_timeout}s exceeded, "
                  "terminating agent run.", file=sys.stderr)
        finally:
            # Cancel alarm regardless of outcome.
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
        # v2 (#13, #3): pick up side-channel info run_agent stashed on itself.
        terminated_by = getattr(run_agent, "last_terminated_by", None)
        usage_summary = getattr(run_agent, "last_usage_summary", None)
        if _watchdog_fired:
            terminated_by = "wall_timeout"
    duration = time.time() - t0

    task_cfg_for_scoring = task_cfg
    result = score_run(task_id, workspace, task_dir, gt_dir, task_cfg=task_cfg_for_scoring)
    # v2 (#18, #13, #3): stamp metadata for downstream aggregation & audit.
    #
    # v4 (侯-6): FIELD SEMANTICS (authoritative documentation):
    #   - result["status"] is the AUTHORITATIVE scoring field. aggregate.py
    #     uses ONLY this to classify trials into buckets (ok/infra/model_fail).
    #     It answers: "did we get a valid score from this trial?"
    #   - result["_meta"]["terminated_by"] is DIAGNOSTIC. It records WHY the
    #     agent stopped (final_answer, stuck_loop, wall_timeout, etc.) but does
    #     NOT directly affect scoring. Use it for post-hoc analysis and reports.
    #   The mapping below translates terminated_by → status only when the
    #   termination reason implies the trial's score is invalid/unreliable.
    result["_meta"] = {
        "task_id": task_id,
        "agent": args.agent,
        "model": args.model,
        "seed": args.seed,
        "docker_image": image,
        "duration_sec": round(duration, 2),
        "terminated_by": terminated_by,
        "usage": usage_summary,
    }
    # If run_agent raised an api_error and score_run still ran (unlikely but
    # possible when the exception propagates before returning), prefer that
    # signal over score_run's status.
    if terminated_by == "api_error" and result.get("status") != "scorer_error":
        result["status"] = "api_error"
    elif terminated_by == "executor_crash" and result.get("status") in (
        None, "no_output", "scorer_error"
    ):
        # v4 (吴-2): removed "ok" from this whitelist. If score_run already
        # succeeded and returned status="ok", the trial produced a valid score
        # — don't overwrite it with executor_crash just because the agent had
        # a late-stage infra fault after outcome.json was already written.
        result["status"] = "executor_crash"
    elif terminated_by in ("stuck_loop", "no_progress") and result.get("status") in (
        None, "no_output"
    ):
        # v3/v3.3 (Wu-2): circuit-breaker termination where NO valid score was
        # produced yet -> model_fail (counts as 0, not infra-dropped). If
        # score_run DID return status="ok" (outcome.json validly written), it is
        # preserved here too — same protection extended as for executor_crash,
        # removing the prior asymmetry that zeroed an actually-completed trial.
        result["status"] = terminated_by
    elif terminated_by == "max_turns_exhausted" and result.get("status") == "no_output":
        # Agent ran out of turns and failed to write required outputs — that's
        # the max_turns signal, not the (redundant) "no_output".
        result["status"] = "max_turns"

    out_path = trial_root / "score.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[runner] score={result.get('combined_score', 0.0):.4f}  → {out_path}")

    # v4 (侯-5): append a one-line summary to trials/index.jsonl so batch
    # results can be glanced without traversing per-trial dirs.
    try:
        index_path = trial_root.parent / "index.jsonl"
        index_entry = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "model": args.model,
            "seed": args.seed,
            "score": result.get("combined_score", 0.0),
            "status": result.get("status"),
            "terminated_by": terminated_by,
            "trial_dir": trial_root.name,
        }
        with open(index_path, "a", encoding="utf-8") as idx_f:
            idx_f.write(json.dumps(index_entry, ensure_ascii=False) + "\n")
    except OSError:
        pass  # Non-critical — don't crash the run for an index write failure.

    # Human-readable report (mirrors PEV-Eval format from the original 159 setup)
    report_path = generate_report(
        trial_root=trial_root,
        task_dir=task_dir,
        workspace=workspace,
        agent=args.agent,
        model=args.model,
        api_base=args.api_base,
        seed=args.seed,
        docker_image=image,
        duration_sec=duration,
        score_result=result,
        trajectory_path=trial_root / "trajectory.jsonl",
    )
    print(f"[runner] report → {report_path}")

    cleanup_workspace(workspace, keep_full=args.keep_full_workspace)


if __name__ == "__main__":
    main()
