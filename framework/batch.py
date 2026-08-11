"""
framework/batch.py
==================
Batch runner: fan out (task × model × seed) triples over a process pool.

v3 : the single-task ``runner.py`` intentionally stays single-task.
This wrapper does the cartesian product and calls the runner as a subprocess
per trial, so:

  * a crash in one trial cannot poison the others
  * Docker's own concurrency limits + the OS cap on subprocess handles are
    the only ceiling — no shared Python state to corrupt
  * every existing runner flag can be passed through unchanged

Usage
-----
    # every task in social_stats, 3 models, seeds 0..4, 4 parallel workers
    python3 -m framework.batch \
        --tasks-glob 'tasks/social_stats/*' \
        --models gpt-5.5,glm-5.2,doubao \
        --seeds 5 \
        --parallel 4 \
        --use-docker

    # explicit seed list
    python3 -m framework.batch \
        --tasks tasks/social_stats/159_gaokao_reform \
        --models gpt-5.5 \
        --seeds 0,2,4 \
        --parallel 2 --use-docker
"""
from __future__ import annotations

import argparse
import concurrent.futures as _futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _parse_seeds(spec: str) -> list[int]:
    """Accept ``N`` (→ 0..N-1), ``a-b`` (inclusive range) or a comma list."""
    spec = spec.strip()
    if "," in spec:
        return sorted({int(x.strip()) for x in spec.split(",") if x.strip()})
    if "-" in spec and not spec.startswith("-"):
        a, b = spec.split("-", 1)
        lo, hi = int(a), int(b)
        return list(range(lo, hi + 1))
    n = int(spec)
    return list(range(n))


def _expand_tasks(task_args: list[str], glob_arg: str | None) -> list[Path]:
    tasks: list[Path] = []
    for t in task_args or []:
        p = Path(t).resolve()
        if not (p / "task.toml").exists():
            raise SystemExit(f"--tasks item is not a task dir (no task.toml): {p}")
        tasks.append(p)
    if glob_arg:
        # Glob is evaluated relative to ROOT (project root) so callers can
        # write 'tasks/social_stats/*' regardless of cwd.
        for p in sorted(ROOT.glob(glob_arg)):
            if (p / "task.toml").exists():
                tasks.append(p.resolve())
    # De-duplicate preserving order.
    seen: set[Path] = set()
    out: list[Path] = []
    for t in tasks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    if not out:
        raise SystemExit("No tasks matched. Pass --tasks or --tasks-glob.")
    return out


def _build_cmd(
    task: Path,
    model: str,
    seed: int,
    *,
    passthrough: list[str],
) -> list[str]:
    cmd = [
        sys.executable, "-m", "framework.runner",
        "--task", str(task),
        "--agent", "http_api",
        "--model", model,
        "--seed", str(seed),
    ]
    cmd.extend(passthrough)
    return cmd


def _reap_trial_containers(trial_tag: str) -> None:
    """v3.3 : best-effort kill of any docker container labelled
    ``bench.trial=<trial_tag>`` so a batch-level timeout that orphans the child
    runner does NOT leak its long-lived agent container until external teardown.
    Reaps BY LABEL ONLY — sibling trials' containers are never touched."""
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(trial_tag))
    if not safe:
        return
    try:
        listing = subprocess.run(
            ["docker", "ps", "--quiet", "--filter", f"label=bench.trial={safe}"],
            capture_output=True, text=True, timeout=15,
        )
        ids = [x.strip() for x in listing.stdout.splitlines() if x.strip()]
        if ids:
            subprocess.run(
                ["docker", "rm", "-f", *ids],
                capture_output=True, timeout=30,
            )
    except (subprocess.TimeoutExpired, OSError):
        pass


def _run_one(cmd: list[str], log_root: Path, tag: str,
             timeout: int = 7200) -> tuple[str, int, str]:
    """Run a single subprocess; capture stdout+stderr to file. Return (tag, rc, log_path).

    v4/v3.3 : added a wall-clock ``timeout`` (default 7200s). When it fires we
    SIGKILL our own python child — but its long-lived agent container keeps running,
    because teardown() never ran in the orphaned child. So on TIMEOUT we also reap any
    containers carrying this trial's ``bench.trial`` label before returning rc=-1.
    """
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{tag}.log"
    with open(log_path, "w", encoding="utf-8") as fh:
        try:
            proc = subprocess.run(
                cmd,
                stdout=fh,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
            )
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            fh.write(f"\n\n[TIMEOUT after {timeout}s — killed by batch runner]\n")
            # The child runner was killed mid-run → it never tore down its agent
            # container(s). Best-effort reap by label so they don't linger eating CPU.
            _reap_trial_containers(tag)
            rc = -1
    return tag, rc, str(log_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fan out (task × model × seed) trials over a process pool."
    )
    parser.add_argument("--tasks", nargs="*", default=[],
                        help="Explicit task directories.")
    parser.add_argument("--tasks-glob", default=None,
                        help="Glob relative to project root, e.g. 'tasks/social_stats/*'.")
    parser.add_argument("--models", required=True,
                        help="Comma-separated model names, e.g. 'gpt-5.5,glm-5.2'.")
    parser.add_argument("--seeds", required=True,
                        help="Seed spec: 'N' → 0..N-1, 'a-b' inclusive, or '0,2,4'.")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Max concurrent trials. Watch Docker/RAM.")
    parser.add_argument("--log-dir", default=None,
                        help="Where per-trial stdout goes. Default: <root>/.bench/batch_logs/<ts>.")

    # Passthrough flags — kept minimal on purpose; add more as needed.
    parser.add_argument("--use-docker", action="store_true")
    parser.add_argument("--image", default=None)
    parser.add_argument("--allow-host-mode", action="store_true")
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-config", default=None,
                        help="v4 : JSON file mapping model_name → "
                             "{api_base, api_key_env, model_id?}. Overrides "
                             "--api-base/--api-key per model.")
    parser.add_argument("--model-id", default=None,
                        help="Override the API-side model id for ALL --models (only useful when --models is a single name).")
    parser.add_argument("--models-yaml", default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--keep-full-workspace", action="store_true")

    args = parser.parse_args()

    tasks = _expand_tasks(args.tasks, args.tasks_glob)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = _parse_seeds(args.seeds)

    if not models:
        raise SystemExit("No models provided (comma-separated).")
    if not seeds:
        raise SystemExit("No seeds provided.")

    # v4: load per-model API config if provided.
    # Format: {"model_name": {"api_base": "...", "api_key_env": "ENV_VAR", "model_id": "..."}}
    api_config: dict[str, dict] = {}
    if args.api_config:
        with open(args.api_config, "r", encoding="utf-8") as _f:
            api_config = json.load(_f)

    passthrough: list[str] = []
    if args.use_docker:
        passthrough.append("--use-docker")
    if args.image:
        passthrough.extend(["--image", args.image])
    if args.allow_host_mode:
        passthrough.append("--allow-host-mode")
    if args.api_base:
        passthrough.extend(["--api-base", args.api_base])
    if args.api_key:
        passthrough.extend(["--api-key", args.api_key])
    if args.model_id:
        passthrough.extend(["--model-id", args.model_id])
    if args.models_yaml:
        passthrough.extend(["--models-yaml", args.models_yaml])
    if args.max_tokens is not None:
        passthrough.extend(["--max-tokens", str(args.max_tokens)])
    if args.temperature is not None:
        passthrough.extend(["--temperature", str(args.temperature)])
    if args.keep_full_workspace:
        passthrough.append("--keep-full-workspace")

    log_root = Path(args.log_dir) if args.log_dir else (
        ROOT / ".bench" / "batch_logs" / time.strftime("%Y%m%d_%H%M%S")
    )
    # v3.3: per-invocation unique suffix so two batches sharing an identical
    # (task,model,seed) combo cannot collide on the bench.trial label we reap by below.
    _launch_ts = str(int(time.time()))

    combos: list[tuple[Path, str, int, str, list[str]]] = []
    for task in tasks:
        for model in models:
            # v4: per-model API override from --api-config
            model_passthrough = list(passthrough)
            if model in api_config:
                mcfg = api_config[model]
                if "api_base" in mcfg:
                    # Remove global --api-base if present, replace with per-model
                    model_passthrough = [x for x in model_passthrough if x != "--api-base"]
                    # Also remove the value after --api-base
                    _cleaned: list[str] = []
                    _skip = False
                    for x in model_passthrough:
                        if _skip:
                            _skip = False
                            continue
                        if x == "--api-base":
                            _skip = True
                            continue
                        _cleaned.append(x)
                    model_passthrough = _cleaned
                    model_passthrough.extend(["--api-base", mcfg["api_base"]])
                if "api_key_env" in mcfg:
                    key = os.environ.get(mcfg["api_key_env"], "")
                    if key:
                        _cleaned2: list[str] = []
                        _skip2 = False
                        for x in model_passthrough:
                            if _skip2:
                                _skip2 = False
                                continue
                            if x == "--api-key":
                                _skip2 = True
                                continue
                            _cleaned2.append(x)
                        model_passthrough = _cleaned2
                        model_passthrough.extend(["--api-key", key])
                if "model_id" in mcfg:
                    _cleaned3: list[str] = []
                    _skip3 = False
                    for x in model_passthrough:
                        if _skip3:
                            _skip3 = False
                            continue
                        if x == "--model-id":
                            _skip3 = True
                            continue
                        _cleaned3.append(x)
                    model_passthrough = _cleaned3
                    model_passthrough.extend(["--model-id", mcfg["model_id"]])
            for seed in seeds:
                # v3.3: this same string travels both ways — it
                # becomes --trial-tag -> bench.trial=<tag> docker LABEL on every
                # container the child runner starts, so _run_one can reap EXACTLY
                # those containers if our process times out. The launch-ts suffix
                # guarantees global uniqueness across concurrent batches.
                tag = f"{task.name}__{model}__seed{seed}__{_launch_ts}"
                trial_passthrough = model_passthrough + ["--trial-tag", tag]
                cmd = _build_cmd(task, model, seed, passthrough=trial_passthrough)
                combos.append((task, model, seed, tag, cmd))

    total = len(combos)
    print(f"[batch] {total} trial(s): "
          f"{len(tasks)} task × {len(models)} model × {len(seeds)} seed  "
          f"| parallel={args.parallel}  | logs → {log_root}")

    workers = max(1, int(args.parallel))
    ok, fail = 0, 0
    failures: list[tuple[str, int, str]] = []
    t0 = time.time()

    with _futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(_run_one, cmd, log_root, tag): (tag, task, model, seed)
            for (task, model, seed, tag, cmd) in combos
        }
        done = 0
        for fut in _futures.as_completed(futs):
            tag, task, model, seed = futs[fut]
            try:
                _, rc, log_path = fut.result()
            except Exception as exc:
                rc, log_path = -1, f"(exception: {exc})"
            done += 1
            if rc == 0:
                ok += 1
                marker = "OK "
            else:
                fail += 1
                failures.append((tag, rc, log_path))
                marker = f"FAIL(rc={rc})"
            elapsed = time.time() - t0
            print(f"[batch] ({done}/{total})  {marker}  {tag}  "
                  f"| {elapsed:.1f}s cumulative  | {log_path}")

    print("")
    print(f"[batch] done in {time.time() - t0:.1f}s  |  ok={ok}  fail={fail}  total={total}")
    if failures:
        print("[batch] failing trials:")
        for tag, rc, log_path in failures:
            print(f"  - {tag}  rc={rc}  log={log_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
