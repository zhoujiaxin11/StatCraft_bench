"""
framework/batch.py
==================
Batch runner: fan out (task × model × seed) triples over a process pool.

the single-task ``runner.py`` intentionally stays single-task.
This wrapper does the cartesian product and calls the runner as a subprocess
per trial, so:

  * a crash in one trial cannot poison the others
  * Docker's own concurrency limits + the OS cap on subprocess handles are
    the only ceiling — no shared Python state to corrupt
  * every existing runner flag can be passed through unchanged

Usage
-----
    # every task in education_academia, 3 models, seeds 0..4, 4 parallel workers
    python3 -m framework.batch \
        --tasks-glob 'tasks/education_academia/*' \
        --models gpt-5.5,glm-5.2,doubao \
        --seeds 5 \
        --parallel 4 \
        --use-docker

    # explicit seed list
    python3 -m framework.batch \
        --tasks tasks/education_academia/159_gaokao_reform \
        --models gpt-5.5 \
        --seeds 0,2,4 \
        --parallel 2 --use-docker

    #disk-tight machine: one trial at a time, and refuse to start
    # a trial with less than 5 GiB free (checked before EVERY trial, not once).
    python3 -m framework.batch \
        --tasks-glob 'tasks/education_academia/*' \
        --models gpt-5.5 --seeds 3 \
        --serial --min-free-gb 5 --use-docker
"""
from __future__ import annotations

import argparse
import concurrent.futures as _futures
import json
import os
import shutil
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
        # write 'tasks/education_academia/*' regardless of cwd.
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
    """ best-effort kill of any docker container labelled
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


# 「磁盘紧张下并发跑题 → 遵循串行实验约束，单题跑完再进
# 下一题」. A trial that runs out of disk halfway does not fail loudly: it leaves a
# truncated workspace, an unwritable trajectory and an unscorable trial, and with
# --parallel N it takes N trials down at once. The floor below is a precondition
# check, not a cleanup — nothing here ever deletes anything.
DEFAULT_MIN_FREE_GB = 5.0


def _free_gb(path: Path) -> float:
    """Free space on the filesystem holding ``path``, in GiB."""
    try:
        return shutil.disk_usage(str(path)).free / (1024 ** 3)
    except OSError:
        return float("inf")  # can't measure → don't block on it


def _run_one(cmd: list[str], log_root: Path, tag: str,
             timeout: int = 7200, min_free_gb: float = 0.0) -> tuple[str, int, str]:
    """Run a single subprocess; capture stdout+stderr to file. Return (tag, rc, log_path).

    added a wall-clock ``timeout`` (default 7200s). When it fires we
    SIGKILL our own python child — but its long-lived agent container keeps running,
    because teardown() never ran in the orphaned child. So on TIMEOUT we also reap any
    containers carrying this trial's ``bench.trial`` label before returning rc=-1.

    ``min_free_gb`` is re-checked HERE, immediately before the
    child is spawned, not only once at batch start. Disk is consumed by the trials
    themselves (workspaces, exec_logs, trajectories, docker layers), so a floor
    checked once up front is stale by trial 3. A trial that would start under the
    floor is NOT started: rc=-2 and a one-line reason in its log. Skipping a trial
    that would have died with a truncated workspace costs nothing; letting it start
    costs the model time AND leaves an unscorable trial behind.
    """
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"{tag}.log"
    if min_free_gb > 0:
        free = _free_gb(log_root)
        if free < min_free_gb:
            with open(log_path, "w", encoding="utf-8") as fh:
                fh.write(
                    f"[SKIPPED before launch — {free:.2f} GiB free < "
                    f"--min-free-gb {min_free_gb:.2f}]\n"
                    "No child runner was started, so no model tokens were spent.\n"
                    "Free disk space and re-run this trial.\n"
                )
            return tag, -2, str(log_path)
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
                        help="Glob relative to project root, e.g. 'tasks/education_academia/*'.")
    parser.add_argument("--models", required=True,
                        help="Comma-separated model names, e.g. 'gpt-5.5,glm-5.2'.")
    parser.add_argument("--seeds", required=True,
                        help="Seed spec: 'N' → 0..N-1, 'a-b' inclusive, or '0,2,4'.")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Max concurrent trials. Watch Docker/RAM.")
    # 串行实验约束的两个开关。默认值刻意保持基线行为
    # （--parallel 的默认已经是 1，磁盘门默认关闭），所以现有命令行一行都不用改。
    parser.add_argument("--min-free-gb", type=float, default=0.0,
                        help=" refuse to START a trial when free disk "
                             "on the log filesystem is below this many GiB (checked "
                             f"per trial, not once). Suggested: {DEFAULT_MIN_FREE_GB}. "
                             "0 (default) disables the check — baseline behaviour.")
    parser.add_argument("--serial", action="store_true",
                        help=" force one trial at a time, overriding "
                             "--parallel. Explicit form of 「单题跑完再进下一题」 for "
                             "disk-tight machines.")
    parser.add_argument("--log-dir", default=None,
                        help="Where per-trial stdout goes. Default: <root>/.comate/batch_logs/<ts>.")

    # Passthrough flags — kept minimal on purpose; add more as needed.
    parser.add_argument("--use-docker", action="store_true")
    parser.add_argument("--image", default=None)
    parser.add_argument("--allow-host-mode", action="store_true")
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-config", default=None,
                        help=" JSON file mapping model_name → "
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

    # load per-model API config if provided.
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
        ROOT / ".comate" / "batch_logs" / time.strftime("%Y%m%d_%H%M%S")
    )
    # per-invocation unique suffix so two batches sharing an identical
    # (task,model,seed) combo cannot collide on the bench.trial label we reap by below.
    _launch_ts = str(int(time.time()))

    combos: list[tuple[Path, str, int, str, list[str]]] = []
    for task in tasks:
        for model in models:
            # per-model API override from --api-config
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
                # this same string travels both ways — it
                # becomes --trial-tag -> bench.trial=<tag> docker LABEL on every
                # container the child runner starts, so _run_one can reap EXACTLY
                # those containers if our process times out. The launch-ts suffix
                # guarantees global uniqueness across concurrent batches.
                tag = f"{task.name}__{model}__seed{seed}__{_launch_ts}"
                trial_passthrough = model_passthrough + ["--trial-tag", tag]
                cmd = _build_cmd(task, model, seed, passthrough=trial_passthrough)
                combos.append((task, model, seed, tag, cmd))

    total = len(combos)
    workers = max(1, int(args.parallel))

    # serial constraint + disk floor.
    #
    # item 15 is 「遵循串行实验约束，单题跑完再进下一题」. Two things are
    # needed for that to be more than a convention:
    #
    #   * --serial makes it explicit and unambiguous (parallel=1 regardless of
    #     what --parallel says), so a stale shell-history command with
    #     --parallel 4 cannot quietly reintroduce concurrency.
    #   * when a disk floor is set, concurrency is downgraded to 1 automatically:
    #     N concurrent trials consume disk N times as fast, and the failure mode
    #     is not a clean error but N truncated workspaces.
    #
    # Both are opt-in: with no new flags this block prints nothing and changes
    # nothing, so every existing batch command behaves exactly as on the baseline.
    if args.serial and workers > 1:
        print(f"[batch] --serial: overriding --parallel {workers} → 1 "
              "(单题跑完再进下一题)")
        workers = 1
    elif args.min_free_gb > 0 and workers > 1:
        print(f"[batch] --min-free-gb {args.min_free_gb:g} set: downgrading "
              f"--parallel {workers} → 1 — concurrent trials burn disk {workers}× "
              "as fast, and running out mid-trial truncates every workspace at once.")
        workers = 1

    if args.min_free_gb > 0:
        free_now = _free_gb(log_root.parent if log_root.parent.exists() else ROOT)
        print(f"[batch] disk floor: {free_now:.2f} GiB free, "
              f"--min-free-gb {args.min_free_gb:g}")
        if free_now < args.min_free_gb:
            # Refuse the whole batch up front rather than skipping trial by trial:
            # there is no point spawning anything when the floor is already broken.
            raise SystemExit(
                f"[batch] REFUSING to start: only {free_now:.2f} GiB free, below "
                f"--min-free-gb {args.min_free_gb:g}. Free disk space first — a "
                "trial that runs out of disk mid-run leaves an unscorable workspace "
                "instead of failing loudly."
            )

    print(f"[batch] {total} trial(s): "
          f"{len(tasks)} task × {len(models)} model × {len(seeds)} seed  "
          f"| parallel={workers}  | logs → {log_root}")

    ok, fail = 0, 0
    failures: list[tuple[str, int, str]] = []
    t0 = time.time()

    with _futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(_run_one, cmd, log_root, tag,
                        7200, args.min_free_gb): (tag, task, model, seed)
            for (task, model, seed, tag, cmd) in combos
        }
        done = 0
        skipped = 0
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
            elif rc == -2:
                # a disk-floor skip is NOT a trial failure —
                # nothing ran, no tokens were spent, and reporting it as FAIL would
                # make a full-disk batch look like a broken framework.
                skipped += 1
                failures.append((tag, rc, log_path))
                marker = "SKIP(disk)"
            else:
                fail += 1
                failures.append((tag, rc, log_path))
                marker = f"FAIL(rc={rc})"
            elapsed = time.time() - t0
            print(f"[batch] ({done}/{total})  {marker}  {tag}  "
                  f"| {elapsed:.1f}s cumulative  | {log_path}")

    print("")
    print(f"[batch] done in {time.time() - t0:.1f}s  |  ok={ok}  fail={fail}  "
          f"skipped={skipped}  total={total}")
    if skipped:
        print(f"[batch] {skipped} trial(s) never started: free disk fell below "
              f"--min-free-gb {args.min_free_gb:g}. Free space and re-run just those.")
    if failures:
        print("[batch] failing trials:")
        for tag, rc, log_path in failures:
            print(f"  - {tag}  rc={rc}  log={log_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
