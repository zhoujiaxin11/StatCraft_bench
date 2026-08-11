"""
framework/docker_executor.py
============================
Docker-based code executor for the HttpApiAgent.

Design
------
- **Host-mode agent**: the main agent process (LLM calls, prompt logic) stays
  on the host. Only the *Python code produced by the LLM* runs inside a
  container. API keys never enter the container.
- **Long-lived container per task**: `setup()` starts one container that stays
  alive for the whole task run; each `run(code)` uses `docker exec` to execute
  a small script inside. `teardown()` stops the container (which self-removes
  via --rm).
- **--network none**: the container has no outbound network access, so agent
  code can't call arbitrary URLs.
- **Read-only data mount**: environment/ is mounted RO so the agent can't
  overwrite input data.
- **Writable workspace**: workspace/ is mounted RW; the agent's outcome.json
  and any artifacts land there and are visible to the host.

Same interface as SubprocessExecutor: `run(code) -> (ok, output)`.
"""
from __future__ import annotations

import os
import subprocess
import time
import uuid
from pathlib import Path


class DockerExecutor:
    """Long-lived container + `docker exec` code execution."""

    def __init__(
        self,
        *,
        image: str,
        workspace: Path,
        data_dir: Path | None = None,
        memory: str = "4g",
        cpus: str = "2",
        timeout_per_step: int = 120,
        allow_network: bool = False,
        read_only_rootfs: bool = True,
        log_dir: Path | None = None,
    ):
        self.image = image
        self.workspace = Path(workspace).resolve()
        self.data_dir = Path(data_dir).resolve() if data_dir else None
        self.memory = memory
        self.cpus = cpus
        # Also expose `timeout` for parity with SubprocessExecutor (agent reads it).
        self.timeout_per_step = timeout_per_step
        self.timeout = timeout_per_step
        self.allow_network = allow_network
        self.read_only_rootfs = read_only_rootfs
        # v2 (#6): optional persistence of code + output for post-mortem audit.
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        # Unique per-run container name to allow concurrent trials
        self.container_name = f"bench_{uuid.uuid4().hex[:12]}"
        self._started = False
        # v3.3: bench.trial=<tag> once setup() runs; lets batch.py reap
        # the exact container after a batch-level timeout instead of leaking it.
        self.trial_label: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def setup(self) -> None:
        """Start the long-lived container."""
        cmd = [
            "docker", "run", "-d", "--rm",
            "--name", self.container_name,
            "--memory", self.memory,
            "--cpus", self.cpus,
            "--cap-drop=ALL",
            "--security-opt", "no-new-privileges",
            "--pids-limit", "256",
        ]
        # v3.3: propagate host-seeded randomness + a trial tag so
        # agent code inside the container reproduces deterministically AND every
        # container carries a label for precise post-timeout reaping by batch.py.
        # PYTHONHASHSEED must be present BEFORE the python interpreter starts, so
        # it is injected here as a container env var (each `docker exec` child of
        # this long-lived container inherits it).
        self.trial_label = None
        for _key in ("PYTHONHASHSEED", "BENCH_SEED"):
            _val = os.environ.get(_key)
            if _val is not None:
                cmd.extend(["-e", f"{_key}={_val}"])
        _tag_raw = os.environ.get("BENCH_TRIAL_TAG")
        if _tag_raw:
            _safe_tag = "".join(
                c if c.isalnum() or c in "-_." else "_" for c in str(_tag_raw)
            )
            if _safe_tag:
                cmd.extend(["--label", f"bench.trial={_safe_tag}"])
                self.trial_label = _safe_tag
        cmd += [
            "-v", f"{self.workspace}:/workspace",
            "-w", "/workspace",
        ]
        if not self.allow_network:
            cmd.extend(["--network", "none"])
        if self.read_only_rootfs:
            # Root filesystem read-only; /workspace stays writable via the bind
            # mount, and /tmp is a tmpfs so libraries that need scratch space
            # (matplotlib cache, joblib) still work.
            cmd.extend([
                "--read-only",
                "--tmpfs", "/tmp:rw,exec,size=1g",
                "--tmpfs", "/home/runner:rw,size=256m",
            ])
        if self.data_dir:
            cmd.extend(["-v", f"{self.data_dir}:/workspace/environment:ro"])
        # Keep container alive; overriding CMD to a no-op that sleeps forever.
        cmd.extend([self.image, "tail", "-f", "/dev/null"])

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Failed to start container (image={self.image!r}):\n{proc.stderr}"
            )
        self._started = True
        self._assert_data_mounted()

    def _assert_data_mounted(self) -> None:
        """Fail loudly if environment/ was requested but is empty inside the
        container. A silently-broken bind mount (wrong path, docker quirk) leaves
        the agent with no input data, which reads as a low score instead of an
        infra error — so it is caught here, before any model time is spent."""
        if not self.data_dir:
            return
        probe = subprocess.run(
            ["docker", "exec", self.container_name, "python", "-c",
             "import os; print('NDATA', len(os.listdir('/workspace/environment')))"],
            capture_output=True, text=True, timeout=30,
        )
        out = (probe.stdout or "") + (probe.stderr or "")
        if probe.returncode != 0 or "NDATA 0" in out:
            self.teardown()
            raise RuntimeError(
                f"environment/ mount is empty or unreadable inside container "
                f"(image={self.image!r}, data_dir={self.data_dir}). "
                f"probe output:\n{out.strip()}"
            )

    def teardown(self) -> None:
        """Stop the container (self-removes due to --rm)."""
        if not self._started:
            return
        subprocess.run(
            ["docker", "stop", "-t", "2", self.container_name],
            capture_output=True, timeout=30,
        )
        self._started = False

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, *_):
        self.teardown()

    # ------------------------------------------------------------------
    # Code execution
    # ------------------------------------------------------------------
    def run(self, code: str, *, log_name: str | None = None) -> tuple[bool, str]:
        """Write `code` to a script inside the mounted workspace, then run it
        inside the container via `docker exec`. Returns (ok, combined_output).

        v2 (#6): when `log_name` and `self.log_dir` are set, persists code +
        full output for post-hoc audit — mirrors SubprocessExecutor.
        """
        if not self._started:
            self.setup()

        # Write into workspace/ on host — container sees it at /workspace/
        script_name = f"_step_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}.py"
        script_path = self.workspace / script_name
        # v3.3: prepend a tiny seeding prologue so ANY random/numpy usage
        # in agent-generated code reproduces deterministically under BENCH_SEED.
        seeded_code = _seed_prologue() + "\n" + code
        script_path.write_text(seeded_code, encoding="utf-8")

        # Per-step PID file inside the container's tmpfs: the launcher records
        # its own shell PID before exec-ing python so we can kill the EXACT timed-
        # out process instead of pattern-matching (which miskills sibling steps).
        pidfile = f"/tmp/.bench_pid_{script_path.stem}"
        try:
            proc = subprocess.run(
                ["docker", "exec", self.container_name,
                 "sh", "-c",
                 f"echo $$ > {pidfile}; exec python {script_name}"],
                capture_output=True, text=True,
                timeout=self.timeout_per_step,
                encoding="utf-8", errors="replace",
            )
            output = (proc.stdout or "")
            if proc.stderr:
                output += "\n[stderr]\n" + proc.stderr
            ok = proc.returncode == 0
            out_stripped = output.strip()
        except subprocess.TimeoutExpired:
            # v3.3: kill ONLY this step's process via its recorded PID.
            # Never `pkill -f <name>` — concurrent timed-out zombies share the
            # "_step_*" suffix and would be killed together.
            self._kill_pid_in_container(pidfile)
            ok = False
            out_stripped = (
                f"[timeout after {self.timeout_per_step}s — "
                "timed-out process killed]"
            )
        finally:
            try:
                script_path.unlink()
            except OSError:
                pass

        if self.log_dir is not None and log_name:
            # Local persistence to avoid coupling with framework.agent_adapter.
            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                (self.log_dir / f"{log_name}.py").write_text(code, encoding="utf-8")
                (self.log_dir / f"{log_name}.out").write_text(
                    f"# ok={ok}\n{out_stripped}", encoding="utf-8"
                )
            except OSError:
                pass
        return ok, out_stripped

    def _kill_pid_in_container(self, pidfile: str) -> None:
        """v3.3 : best-effort SIGTERM→SIGKILL of ONLY the PID recorded in
        ``pidfile``. Never pattern-matches script names (which miskill siblings)."""
        for sig in ("TERM", "KILL"):
            try:
                subprocess.run(
                    ["docker", "exec", self.container_name,
                     "sh", "-c",
                     "[ -f {pf} ] && kill -{sg} \"$(cat {pf})\" 2>/dev/null || true".format(
                         pf=pidfile, sg=sig)],
                    capture_output=True, timeout=6,
                )
            except subprocess.TimeoutExpired:
                break  # docker exec itself hung — give up on further attempts.
        try:
            subprocess.run(
                ["docker", "exec", self.container_name, "rm", "-f", pidfile],
                capture_output=True, timeout=4,
            )
        except subprocess.TimeoutExpired:
            pass


def _seed_prologue() -> str:
    """Python source prepended ahead of each executed block so random/numpy usage
    inside agent-generated code reproduces deterministically across BENCH_SEED values."""
    try:
        s = int(os.environ.get("BENCH_SEED") or os.environ.get("SEED") or "0")
    except ValueError:
        s = 0
    lines = [
        "# framework determinism prologue (BENCH_SEED)",
        "import random as _bs_r",
        f"_bs_r.seed({s})",
        "try:",
        "    import numpy as _bs_n",
        f"    _bs_n.random.seed({s})",
        "except Exception:",
        "    pass",
    ]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def check_docker_available() -> bool:
    """Return True if `docker` CLI is installed and daemon is reachable."""
    try:
        proc = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=10
        )
        return proc.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def image_exists(image: str, retries: int = 5, delay: float = 0.7) -> bool:
    """Return True if the given image tag exists locally.

    v4 : tolerate Docker CLI concurrency jitter by retrying with backoff,
    catching TimeoutExpired/OSError that previously crashed callers.
    v3.3 补强: jitter 也常以 **非异常的 rc≠0** 出现（而非仅 timeout）。
      原实现一遇非零就立刻返 False 不再重试 —— 实测在容器频繁启停的启动窗口里偶发假阴，
      把整批 trial 打死。现在：rc≠0 时若输出含明确“No such image / reference …”
      才判为真正缺失并快返；否则视为瞬态、按指数退避重试到耗尽为止。
    """
    ABSENT_MARKERS = (
        b"no such image", b"reference does not exist", b"manifest unknown",
        b"no manifest found",
    )
    for attempt in range(retries):
        try:
            proc = subprocess.run(
                ["docker", "image", "inspect", image],
                capture_output=True, timeout=10,
            )
            if proc.returncode == 0:
                return True
            blob = ((proc.stderr or b"") + (proc.stdout or b"")).lower()
            if any(m in blob for m in ABSENT_MARKERS):
                return False  # genuinely absent — don't waste time retrying
            # rc≠0 但无缺席标记 ⇒ 视作瞬态抖动，落入下方退避重试
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass
        if attempt < retries - 1:
            time.sleep(delay * (attempt + 1))  # linear-ish backoff
    return False
