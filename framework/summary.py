"""
framework/summary
=================
Render a markdown overview across one task's trial index.

``runner.py`` already appends a JSONL line per finished trial to
``<task>/trials/index.jsonl``. This module is the missing *read* side: it rolls
those rows up into a per-model table so you can glance at who ran and how they
did without traversing every ``score.json``.

Usage::

    python3 -m framework.summary --task tasks/education_academia/159_gaokao_reform

The ability estimate mirrors aggregate.py's intent: infra-fault trials are
DROPPED from the mean; model_fail trials count as 0.
"""
from __future__ import annotations

import argparse
import json
import statistics as _stats
from pathlib import Path


def _load_rows(task_dir: Path) -> list[dict]:
    idx = task_dir / "trials" / "index.jsonl"
    if not idx.exists():
        return []
    out: list[dict] = []
    for ln in idx.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            continue  # skip partial/corrupt tail lines from an interrupted run
    return out


# Single source of truth — framework/statuses.py is stdlib-only, so importing
# it here costs nothing and guarantees the model_fail set can never drift from
# the one aggregate.py and the runner actually use.
from framework.statuses import MODEL_FAIL_STATUSES


def _usable(status: str | None) -> bool:
    """True when the row contributes to the ability estimate."""
    if status == "ok":
        return True
    # Any model_fail status is usable but scores ~0. Infra / unknown are dropped.
    return status in MODEL_FAIL_STATUSES


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, help="Task directory containing trials/")
    args = ap.parse_args()

    task_dir = Path(args.task).resolve()
    rows = _load_rows(task_dir)

    print(f"# Trials summary — {task_dir.name}\n")
    print(f"_source: {task_dir}/trials/index.jsonl_\n")
    if not rows:
        print("_(no trials/index.jsonl yet — run the runner first)_")
        return

    groups: dict[str, dict] = {}
    for r in rows:
        m = str(r.get("model") or "?")
        g = groups.setdefault(m, {"scored": [], "infra": [], "total": []})
        g["total"].append(r)
        score = float(r.get("score", 0) or 0)
        st = r.get("status")
        if _usable(st):
            g["scored"].append(score)
        else:
            g["infra"].append(r)

    print("| model | runs | scored(n) | infra-dropped | best | worst |")
    print("|---|---|---|---|---|---|")
    for m in sorted(groups):
        s: list[float] = groups[m]["scored"]
        n_infra = len(groups[m]["infra"])
        n_total = len(groups[m]["total"])
        best = f"{max(s):.3f}" if s else "-"
        worst = f"{min(s):.3f}" if s else "-"
        if len(s) >= 2:
            spread = f"{_stats.mean(s):.3f}±{_stats.pstdev(s):.3f} ({len(s)})"
        elif len(s) == 1:
            spread = f"{s[0]:.3f} ({1})"
        else:
            # every trial was infra — ability was never
            # measured. Print N/A rather than a dash that reads like a 0.
            spread = "N/A (ability not measured)"
        print(f"| {m} | {n_total} | {spread} | {n_infra} | {best} | {worst} |")

    print("\n## dropped trials\n")
    any_dropped = False
    for m in sorted(groups):
        for r in groups[m]["infra"]:
            any_dropped = True
            print(f"- `{r.get('trial_dir','?')}` model={m} "
                  f"status={r.get('status')} terminated_by={r.get('terminated_by')}")
    if not any_dropped:
        print("_(none)_")


if __name__ == "__main__":
    main()
