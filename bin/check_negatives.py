#!/usr/bin/env python3
"""bin/check_negatives.py — standalone negative-coverage check.

Runs every negatives/*.py for a task in an isolated sandbox, scores the
outcome.json each one produces with the task's own scorer, and reports whether
the checkpoints can actually be knocked down. A checkpoint that only an empty
answer can move, or a veto that never fires, is a checkpoint you cannot trust.

Deliberately self-contained: uses ONLY framework.scorer + framework.aggregate,
which already exist in this repo. No gtsplit / compat / runner dependency, and
it does not touch the task's real outcome.json (each negative runs in a temp
dir with the environment/ data linked in).

    python3 bin/check_negatives.py --task social_stats/159_gaokao_reform
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from framework.aggregate import apply_veto  # noqa: E402
from framework.scorer import score_task  # noqa: E402


def _find_task(task: str) -> Path:
    p = REPO / "tasks" / task
    if not p.is_dir():
        sys.exit(f"task dir not found: {p}")
    return p


def _find_groundtruth(task_id: str) -> Path:
    p = REPO / "groundtruth" / "public" / f"{task_id}.json"
    if not p.is_file():
        sys.exit(f"groundtruth not found: {p}")
    return p


def _run_negative(neg: Path, task_dir: Path, env: dict | None = None) -> tuple[dict | None, str]:
    """Run one negative in an isolated sandbox; return (outcome_dict, status)."""
    with tempfile.TemporaryDirectory() as tmp:
        sandbox = Path(tmp)
        # Link the environment/ data in so `./environment/*.csv` resolves,
        # without copying large files.
        env_src = task_dir / "environment"
        if env_src.is_dir():
            (sandbox / "environment").symlink_to(env_src)
        import os
        run_env = {**os.environ, **(env or {})}
        try:
            proc = subprocess.run(
                [sys.executable, str(neg)],
                cwd=sandbox,
                capture_output=True,
                text=True,
                timeout=300,
                env=run_env,
            )
        except subprocess.TimeoutExpired:
            return None, "TIMEOUT"
        if proc.returncode != 0:
            tail = proc.stderr.strip().splitlines()[-1:] or ["(no stderr)"]
            return None, f"EXIT{proc.returncode}: {tail[0]}"
        out = sandbox / "outcome.json"
        if not out.is_file():
            return None, "NO_OUTCOME"
        try:
            return json.loads(out.read_text(encoding="utf-8")), "OK"
        except json.JSONDecodeError as e:
            return None, f"BAD_JSON: {e}"


def _score(outcome: dict, gt_path: Path):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(outcome, tf, ensure_ascii=False)
        ans_path = tf.name
    try:
        return score_task(answer_path=ans_path, groundtruth_path=gt_path)
    finally:
        Path(ans_path).unlink(missing_ok=True)


def _check_perturb(neg: Path, task_dir: Path, gt_path: Path, weights: dict) -> list[str]:
    """Run perturb_one.py once per field; a field that keeps full credit after
    being displaced past its own band has a tolerance that accepts anything.
    Returns the list of such 'too-wide' fields."""
    oracle = task_dir / "solution" / "solve.py"
    if not oracle.is_file():
        return ["<no oracle: solution/solve.py missing>"]
    survived = []
    for field in weights:
        outcome, status = _run_negative(
            neg, task_dir,
            env={"HARBOR_PERTURB_FIELD": field, "HARBOR_ORACLE_PATH": str(oracle.resolve())},
        )
        if outcome is None:
            continue  # field not probeable / oracle omitted it
        result = _score(outcome, gt_path)
        if result.per_field.get(field, 0.0) > 0.0:
            survived.append(field)
    return survived


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="e.g. social_stats/159_gaokao_reform")
    ap.add_argument("--ceiling", type=float, default=0.5,
                    help="a negative scoring above this is flagged as too generous")
    args = ap.parse_args()

    task_dir = _find_task(args.task)
    task_id = task_dir.name
    gt_path = _find_groundtruth(task_id)
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    weights = gt.get("weights", {})

    neg_dir = task_dir / "negatives"
    negs = sorted(neg_dir.glob("*.py"))
    if not negs:
        sys.exit(f"no negatives found in {neg_dir}")

    print("=" * 74)
    print(f"NEGATIVE COVERAGE  {task_id}   ({len(negs)} negatives)")
    print("=" * 74)
    print(f"  {'negative':<26}{'score':>8}{'after veto':>12}  vetoed")

    # field -> list of negatives that knocked it to 0
    knocked_by: dict[str, list[str]] = {f: [] for f in weights}
    crashed: list[tuple[str, str]] = []
    too_generous: list[tuple[str, float]] = []
    any_vetoed = False
    perturb_survivors: list[str] = []

    for neg in negs:
        name = neg.name

        # perturb_one.py is parameterised: it needs one run per field with the
        # oracle wired in. Handled separately as a tolerance probe, below.
        if name == "perturb_one.py":
            perturb_survivors = _check_perturb(neg, task_dir, gt_path, weights)
            print(f"  {name:<26}{'(probe)':>8}{'':>12}  "
                  f"ran {len(weights)} field displacements")
            continue

        outcome, status = _run_negative(neg, task_dir)
        if outcome is None:
            crashed.append((name, status))
            print(f"  {name:<26}{'—':>8}{'—':>12}  (DID NOT RUN: {status})")
            continue

        result = _score(outcome, gt_path)
        per_field = result.per_field
        score_json = {
            "combined_score": result.total_score,
            "factoid_details": {"per_field": per_field},
        }
        after, vetoed = apply_veto(score_json, gt)
        any_vetoed = any_vetoed or vetoed

        for f in weights:
            if per_field.get(f, 0.0) <= 0.0:
                knocked_by[f].append(name)

        if result.total_score > args.ceiling:
            too_generous.append((name, result.total_score))

        print(f"  {name:<26}{result.total_score:>8.4f}{after:>12.4f}  {vetoed}")

    # ---- Coverage verdict ---------------------------------------------------
    dead = [f for f, movers in knocked_by.items() if not movers]
    dead_weight = sum(float(weights[f]) for f in dead)

    print()
    covered = len(weights) - len(dead)
    print(f"checkpoints knocked down by >=1 negative: {covered}/{len(weights)}")

    ok = True
    if dead:
        print()
        print(f"FAIL — {dead_weight:.0f} weight sits on checkpoints no negative can move:")
        for f in dead:
            print(f"   x {f}  (w={weights[f]}, {gt.get('scoring', {}).get(f, {}).get('type')})")
        ok = False

    if gt.get("veto") and not any_vetoed:
        print()
        print("FAIL — a veto block is declared but no negative ever triggered it "
              "(veto_probe.py should).")
        ok = False

    if perturb_survivors:
        print()
        real = [f for f in perturb_survivors if not f.startswith("<")]
        notes = [f for f in perturb_survivors if f.startswith("<")]
        if real:
            sw = sum(float(weights[f]) for f in real)
            print(f"FAIL — {sw:.0f} weight on {len(real)} checkpoint(s) whose tolerance "
                  f"accepted a value displaced past its own declared band:")
            for f in real:
                print(f"   x {f}  (w={weights[f]}, "
                      f"{gt.get('scoring', {}).get(f, {}).get('type')})")
            ok = False
        for n in notes:
            print(f"   . perturb probe skipped: {n}")

    if too_generous:
        print()
        print(f"WARN — {len(too_generous)} negative(s) scored above ceiling "
              f"{args.ceiling} (too generous?):")
        for name, sc in too_generous:
            print(f"   ! {name}: {sc:.4f}")

    if crashed:
        print()
        print(f"NOTE — {len(crashed)} negative(s) did not run (excluded from coverage):")
        for name, st in crashed:
            print(f"   . {name}: {st}")

    print()
    if ok and not crashed:
        print("OK — every checkpoint is knocked down by a negative, veto fires.")
        sys.exit(0)
    if ok and crashed:
        print("OK, WITH GAPS — coverage complete among negatives that ran.")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()
