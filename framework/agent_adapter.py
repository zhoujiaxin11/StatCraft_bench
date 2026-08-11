"""
framework/agent_adapter.py
==========================
HTTP-API agent adapter compatible with Harbor's Agent interface.

Wraps 159's original `minimal_agent.py` philosophy (LLM emits Python code,
executor runs it, results feed back) into a Harbor-compatible Agent shape.

Design sources
--------------
- 159's minimal_agent.py: HTTP-API cross-model, plain code interpretation loop
- Harbor's BaseAgent contract: `.setup()` / `.step(observation)` / `.teardown()`
- smolagents (DABStep's official harness): Jupyter Kernel Gateway code execution

v2 notes (质检版修订)
--------------------
  * #1: retry with exponential backoff (deadline-aware)
  * #2: verify workspace/outcome.json before accepting [FINAL_ANSWER]
  * #3: ChatResponse dataclass carrying usage / stop_reason
  * #4: max_tokens defaults to None (vendor default)
  * #5: trajectory streamed to disk on every event
  * #6: trajectory records full tool output + exec_logs persistence
  * #7: three-level fallback code-block extraction
  * #8: optional MAX_BLOCKS_PER_TURN cap (default: unlimited)
  * #9: [FINAL_ANSWER] must be on its own line, outside code fences
  * #10-12: system prompt aligned with the above
  * #13: run() returns terminated_by
  * #14: last-turn nudge before max_turns exhausts
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


# ---------------------------------------------------------------------------
# Harbor-compatible Agent contract (structural)
# ---------------------------------------------------------------------------
class HarborAgent(Protocol):
    def setup(self, task_dir: Path, workspace: Path) -> None: ...
    def step(self, observation: str) -> str: ...
    def teardown(self) -> None: ...


# ---------------------------------------------------------------------------
# Model client (OpenAI-compatible HTTP API)
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    name: str
    api_base: str
    api_key: str
    model_id: str = ""
    temperature: float = 0.0
    # v2 (#4): None means "do not send max_tokens" — vendor default applies.
    max_tokens: int | None = None
    timeout: int = 300
    # v2 (#1): user-level retry budget for transient (429/5xx/connect/timeout).
    max_retries: int = 4

    def __post_init__(self):
        if not self.model_id:
            self.model_id = self.name


@dataclass
class ChatResponse:
    """v2 (#3): carry usage / stop_reason alongside content."""
    content: str
    stop_reason: str | None = None
    usage: dict | None = None
    model: str | None = None


_RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class ModelClient:
    """OpenAI-compatible chat completion caller."""

    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        try:
            from openai import OpenAI  # openai>=1.0
        except ImportError as e:
            raise RuntimeError("openai package required: pip install openai") from e
        self._client = OpenAI(
            base_url=cfg.api_base,
            api_key=cfg.api_key,
            timeout=cfg.timeout,
            max_retries=max(0, cfg.max_retries),
        )

    def chat(self, messages: list[dict], *, deadline: float | None = None) -> ChatResponse:
        """Streaming chat call with exponential-backoff retry (v2 #1).

        `deadline` (time.time seconds) truncates retry sleeps so the loop
        cannot outrun the per-step timeout budget.
        """
        try:
            from openai import APIConnectionError, APITimeoutError, APIStatusError
        except ImportError:  # pragma: no cover
            APIConnectionError = APITimeoutError = APIStatusError = Exception  # type: ignore

        attempt = 0
        last_exc: Exception | None = None
        while True:
            try:
                return self._do_chat(messages)
            except (APIConnectionError, APITimeoutError) as exc:  # type: ignore[misc]
                last_exc = exc
            except APIStatusError as exc:  # type: ignore[misc]
                status = getattr(exc, "status_code", None)
                if status not in _RETRY_STATUS:
                    raise
                last_exc = exc

            attempt += 1
            if attempt > self.cfg.max_retries:
                assert last_exc is not None
                raise last_exc

            delay = min(2 ** attempt + random.random(), 30.0)
            if deadline is not None:
                remaining = deadline - time.time()
                if remaining <= 0:
                    assert last_exc is not None
                    raise last_exc
                delay = min(delay, max(0.0, remaining - 0.5))
            time.sleep(delay)

    def _do_chat(self, messages: list[dict]) -> ChatResponse:
        kwargs: dict[str, Any] = dict(
            model=self.cfg.model_id,
            messages=messages,
            temperature=self.cfg.temperature,
            stream=True,
            stream_options={"include_usage": True},
        )
        if self.cfg.max_tokens is not None:
            kwargs["max_tokens"] = self.cfg.max_tokens

        try:
            stream = self._client.chat.completions.create(**kwargs)
        except TypeError:
            # Older gateway shims may reject stream_options.
            kwargs.pop("stream_options", None)
            stream = self._client.chat.completions.create(**kwargs)

        chunks: list[str] = []
        stop_reason: str | None = None
        usage: dict | None = None
        model_id: str | None = None
        for event in stream:
            u = getattr(event, "usage", None)
            if u is not None:
                usage = _usage_to_dict(u)
            m = getattr(event, "model", None)
            if m:
                model_id = m
            if not event.choices:
                continue
            choice = event.choices[0]
            delta = choice.delta
            if delta and getattr(delta, "content", None):
                chunks.append(delta.content)
            fr = getattr(choice, "finish_reason", None)
            if fr:
                stop_reason = fr

        return ChatResponse(
            content="".join(chunks),
            stop_reason=stop_reason,
            usage=usage,
            model=model_id or self.cfg.model_id,
        )


def _usage_to_dict(u: Any) -> dict:
    try:
        return u.model_dump()
    except AttributeError:
        pass
    return {
        "prompt_tokens": getattr(u, "prompt_tokens", None),
        "completion_tokens": getattr(u, "completion_tokens", None),
        "total_tokens": getattr(u, "total_tokens", None),
    }


# ---------------------------------------------------------------------------
# Code executor
# ---------------------------------------------------------------------------
def _persist_exec_log(log_dir: Path, log_name: str, code: str,
                      output: str, ok: bool) -> None:
    """Persist code + full output for post-hoc audit (v2 #6, )."""
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"{log_name}.py").write_text(code, encoding="utf-8")
        (log_dir / f"{log_name}.out").write_text(
            f"# ok={ok}\n{output}", encoding="utf-8"
        )
    except OSError:
        pass


class SubprocessExecutor:
    """Runs Python code in the workspace via host subprocess (no container)."""

    def __init__(self, workspace: Path, timeout: int = 180, log_dir: Path | None = None):
        self.workspace = workspace
        self.timeout = timeout
        self.log_dir = Path(log_dir) if log_dir else None
        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def setup(self) -> None:
        pass

    def teardown(self) -> None:
        pass

    def run(self, code: str, *, log_name: str | None = None) -> tuple[bool, str]:
        # v3.3: prepend a determinism prologue driven by BENCH_SEED so any
        # random/numpy usage inside agent code reproduces per trial-seed. The host
        # subprocess inherits PYTHONHASHSEED too (set by runner main).
        import os
        try:
            bs_seed = int(os.environ.get("BENCH_SEED") or os.environ.get("SEED") or "0")
        except ValueError:
            bs_seed = 0
        prologue = "\n".join([
            "# framework determinism prologue (BENCH_SEED)",
            "import random as _bs_r",
            f"_bs_r.seed({bs_seed})",
            "try:",
            "    import numpy as _bs_n",
            f"    _bs_n.random.seed({bs_seed})",
            "except Exception:",
            "    pass",
        ]) + "\n\n"
        script = self.workspace / f"_step_{int(time.time() * 1000)}.py"
        script.write_text(prologue + code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                cwd=str(self.workspace),
                capture_output=True,
                text=True,
                timeout=self.timeout,
                encoding="utf-8",
                errors="replace",
            )
            output = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
            ok = proc.returncode == 0
            out_stripped = output.strip()
        except subprocess.TimeoutExpired:
            ok = False
            out_stripped = f"[timeout after {self.timeout}s]"
        finally:
            try:
                script.unlink()
            except OSError:
                pass

        if self.log_dir is not None and log_name:
            _persist_exec_log(self.log_dir, log_name, code, out_stripped, ok)
        return ok, out_stripped


CodeExecutor = SubprocessExecutor  # backward-compat alias


# ---------------------------------------------------------------------------
# The Agent
# ---------------------------------------------------------------------------
# v2 (#10-12 )
SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a data-analysis agent. You solve the task by writing Python code
    that runs in the task workspace (Python 3.11 — do not use features
    introduced after 3.11 such as type parameter defaults or except*).
    Available libraries include pandas, numpy, scipy, statsmodels,
    scikit-learn, matplotlib.

    Response format
    ---------------
    In each turn you may output one OR more Python code blocks and, when the
    task is complete, a final-answer marker.

      1) Python code blocks to execute:
         ```python
         # your code here
         ```

      2) Final answer marker — after outcome.json has actually been written to
         disk, add on its own line, OUTSIDE any code block:
         [FINAL_ANSWER]

    Rules
    -----
    - Read data using relative paths (e.g. `./environment/*.csv`).
    - Write outputs to the current working directory (e.g. `./outcome.json`).
    - Do not attempt to read `tests/`, `groundtruth*`, or any scoring files.
    - `outcome.json` MUST match the schema field names and types EXACTLY.
      No missing keys, no extra keys, no renamed / case-changed keys.
    - Put `[FINAL_ANSWER]` on its own line, outside any code block. The
      framework only accepts it when it appears alone on a line; mentions
      inside code, strings, or comments are ignored.
    - Only send `[FINAL_ANSWER]` after the code that writes outcome.json has
      actually run to completion. If the framework detects that outcome.json
      is missing or unparseable it will reject your marker and ask for a fix.
    - Print intermediate results to help debug.
    """
).strip()


class HttpApiAgent:
    """Cross-model HTTP-API agent (Harbor-compatible)."""

    def __init__(
        self,
        model_cfg: ModelConfig,
        max_turns: int = 30,
        executor=None,
        *,
        max_blocks_per_turn: int | None = None,
        trajectory_path: Path | None = None,
        empty_streak_limit: int = 3,
        repeat_streak_limit: int = 3,
    ):
        self.model = ModelClient(model_cfg)
        self.max_turns = max_turns
        self.messages: list[dict] = []
        self.executor = executor
        self.trajectory: list[dict] = []
        self.max_blocks_per_turn = max_blocks_per_turn
        self._trajectory_path: Path | None = Path(trajectory_path) if trajectory_path else None
        self._trajectory_writer = None
        self._workspace: Path | None = None
        self._turn_idx: int = 0
        self.usage_summary: dict = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
        }
        # v2 (#13): filled by run(); consumed by runner.py.
        self.terminated_by: str | None = None
        self._per_step_timeout: int = 180
        self._pending_observation: str = ""
        # v3: circuit breaker counters.
        self._empty_streak: int = 0           # consecutive turns with no code AND no [FINAL_ANSWER]
        self._repeat_streak: int = 0          # consecutive turns with the same reply body
        self._last_reply_hash: str | None = None
        self._empty_streak_limit = int(empty_streak_limit)
        self._repeat_streak_limit = int(repeat_streak_limit)
        # v4: track last code execution time for mtime freshness check.
        self._last_exec_time: float | None = None

    def setup(self, task_dir: Path, workspace: Path) -> None:
        self._workspace = workspace
        instruction = (task_dir / "instruction.md").read_text(encoding="utf-8")
        schema = (task_dir / "schema.json").read_text(encoding="utf-8")
        if self.executor is None:
            self.executor = SubprocessExecutor(workspace, timeout=self._per_step_timeout)
        self.executor.setup()
        # v3.3: seed this process's own RNGs from exported BENCH_SEED so
        # adapter-side randomness (retry jitter, sampling) is deterministic per
        # trial-seed instead of collapsing across seeds when temperature == 0.
        try:
            import os as _os_seed
            _seed_val = int(
                _os_seed.environ.get("BENCH_SEED")
                or _os_seed.environ.get("PYTHONHASHSEED") or "0"
            )
        except ValueError:
            _seed_val = 0
        random.seed(_seed_val)
        try:
            import numpy as _np_seed
            _np_seed.random.seed(_seed_val)
        except ImportError:
            pass
        # Read per-step timeout from executor so retry deadlines stay in budget.
        for attr in ("timeout", "timeout_per_step"):
            v = getattr(self.executor, attr, None)
            if isinstance(v, (int, float)) and v > 0:
                self._per_step_timeout = int(v)
                break
        # v2 (#5): streaming trajectory writer.
        if self._trajectory_path is not None:
            self._trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            self._trajectory_writer = open(self._trajectory_path, "w", encoding="utf-8")
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"# Task\n{instruction}\n\n"
                    f"# Answer Schema (output field definitions)\n```json\n{schema}\n```"
                ),
            },
        ]

    def _record(self, entry: dict) -> None:
        """Append trajectory entry; stream to disk immediately (v2 #5)."""
        self.trajectory.append(entry)
        if self._trajectory_writer is not None:
            try:
                self._trajectory_writer.write(json.dumps(entry, ensure_ascii=False) + "\n")
                self._trajectory_writer.flush()
            except (OSError, ValueError):
                pass

    def step(self, observation: str = "") -> str:
        if observation:
            self.messages.append({"role": "user", "content": f"[Execution result]\n{observation}"})

        deadline = time.time() + max(self._per_step_timeout * 2, 30)
        try:
            resp = self.model.chat(self.messages, deadline=deadline)
        except Exception as exc:  # v2 #1 + #13
            self.terminated_by = "api_error"
            self._record({
                "role": "system",
                "event": "api_error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "turn_idx": self._turn_idx,
            })
            self._pending_observation = f"[API error] {type(exc).__name__}: {exc}"
            raise

        reply = resp.content
        self.messages.append({"role": "assistant", "content": reply})
        self._record({
            "role": "assistant",
            "content": reply,
            "stop_reason": resp.stop_reason,
            "usage": resp.usage,
            "model": resp.model,
            "turn_idx": self._turn_idx,
        })
        if resp.usage:
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                v = resp.usage.get(k)
                if isinstance(v, (int, float)):
                    self.usage_summary[k] += int(v)
            self.usage_summary["calls"] += 1

        # v3 / v3.3: repeat detection. A turn that declares
        # [FINAL_ANSWER] is routed to the verify branch below REGARDLESS of body
        # repetition — it must NOT feed the stuck-loop counter. Otherwise an
        # identical-but-valid repeated answer whose outcome verification is still
        # being nudged gets killed as stuck_loop before it can recover. Genuine
        # bare-text loops carry no marker, so they keep counting; max_turns stays
        # as the ultimate backstop either way.
        declared_final = _detect_final_answer(reply)
        if not declared_final:
            reply_hash = hashlib.sha1(reply.encode("utf-8", errors="replace")).hexdigest()
            if reply.strip() and self._last_reply_hash == reply_hash:
                self._repeat_streak += 1
            else:
                self._repeat_streak = 0
            self._last_reply_hash = reply_hash
            if self._repeat_streak >= self._repeat_streak_limit:
                self.terminated_by = "stuck_loop"
                self._record({
                    "role": "system",
                    "event": "stuck_loop",
                    "repeat_streak": self._repeat_streak,
                    "turn_idx": self._turn_idx,
                })
                self._pending_observation = (
                    f"[Framework note] Detected {self._repeat_streak} consecutive "
                    "identical replies — terminating to avoid a stuck loop."
                )
                return "done"  # signals run() to stop; terminated_by carries reason

        code_blocks = _extract_all_python(reply)

        # v3: if the reply was cut off by max_tokens ("stop_reason=length")
        # and the strict regex found nothing, try to salvage a truncated block:
        # take from the last opening ```python (or ```py) fence to the end and
        # mark it truncated=True. If successful, also stitch a hint into the
        # observation asking the model to finish the truncated code next turn.
        length_truncated = (resp.stop_reason == "length")
        truncated_note = ""
        truncated_block_idxs: set[int] = set()  # v4 : track salvaged blocks
        if length_truncated and not code_blocks:
            salvaged = _salvage_truncated_python(reply)
            if salvaged:
                code_blocks = [salvaged]
                truncated_block_idxs.add(0)
                truncated_note = (
                    "\n\n[Framework note] Your previous reply was cut off by "
                    "max_tokens with an unterminated ```python fence. The "
                    "framework executed the salvaged (truncated) code above. "
                    "Please continue by writing the remainder of the computation "
                    "in a NEW fully-terminated ```python fence."
                )
        elif length_truncated and code_blocks:
            # Reply had valid blocks but was still cut — warn so the model
            # knows to be shorter or wrap up.
            truncated_note = (
                "\n\n[Framework note] Your reply was cut off by max_tokens. "
                "Aim to close every ```python fence and wrap your work up in "
                "smaller chunks."
            )

        # v2 (#8): optional cap on blocks-per-turn.
        capped_note = ""
        if self.max_blocks_per_turn is not None and len(code_blocks) > self.max_blocks_per_turn:
            n_all = len(code_blocks)
            code_blocks = code_blocks[: self.max_blocks_per_turn]
            capped_note = (
                f"\n\n[Framework note] {n_all} code blocks emitted; only the first "
                f"{self.max_blocks_per_turn} were executed. Please concentrate your "
                f"work into a smaller number of runnable blocks."
            )

        outputs: list[str] = []
        any_failed = False
        for i, code in enumerate(code_blocks):
            log_name = f"block_{self._turn_idx:02d}_{i:02d}"
            # v3: executor faults (OSError, subprocess errors) are
            # infrastructure crashes, not the model's fault — tag them as
            # executor_crash so aggregate.py can drop them from the mean.
            try:
                try:
                    ok, output = self.executor.run(code, log_name=log_name)  # type: ignore[call-arg]
                except TypeError:
                    # Backward compat: executors without log_name kwarg.
                    ok, output = self.executor.run(code)  # type: ignore[union-attr]
                # v4: record execution timestamp for mtime freshness check.
                self._last_exec_time = time.time()
            except (OSError, subprocess.SubprocessError) as exc:
                self.terminated_by = "executor_crash"
                self._record({
                    "role": "system",
                    "event": "executor_crash",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "block_idx": i,
                    "turn_idx": self._turn_idx,
                })
                self._pending_observation = (
                    f"[Executor error] {type(exc).__name__}: {exc}"
                )
                raise
            if not ok:
                any_failed = True
            trunc = output if len(output) < 3000 else output[:3000] + "\n[... truncated]"
            # v2 (#6): trajectory keeps full output; observation to model is truncated.
            self._record({
                "role": "tool",
                "content": trunc,
                "content_full": output,
                "ok": ok,
                "truncated": i in truncated_block_idxs,  # v4
                "block_idx": i,
                "turn_idx": self._turn_idx,
                "log_name": log_name,
            })
            outputs.append(f"[Block {i + 1}/{len(code_blocks)} ok={ok}]\n{trunc}")

        pending = "\n\n".join(outputs) if outputs else ""
        if capped_note:
            pending += capped_note
        if truncated_note:
            pending += truncated_note

        # v2 (#9): anchor [FINAL_ANSWER] to its own line, outside fences.
        declared_final = _detect_final_answer(reply)

        if declared_final:
            # v3: accept [FINAL_ANSWER] whenever outcome.json is already
            # written, even if this turn had no code blocks. The write may have
            # happened in a previous turn (e.g. model wrote outcome, saw a
            # bounce, then re-sent [FINAL_ANSWER] alone).
            ok_final, reason = self._verify_final_answer()
            if any_failed:
                # Code failed this turn — reject regardless of outcome.json.
                pending += (
                    "\n\n[Framework note] Your code block failed with the error "
                    "above, so [FINAL_ANSWER] is rejected. Please read the "
                    "traceback, fix the issue, and try again. Do not send "
                    "[FINAL_ANSWER] unless your code actually ran to completion "
                    "and wrote outcome.json."
                )
                pending += self._workspace_listing_hint()
                self._pending_observation = pending
                self._empty_streak = 0  # code ran (though failed) → progress
                return "continue"
            if ok_final:
                self._pending_observation = pending
                return "done"
            # outcome.json missing / bad JSON.
            pending += (
                f"\n\n[Framework note] [FINAL_ANSWER] was rejected because "
                f"outcome.json {reason}. Please write a valid outcome.json and "
                f"try again."
            )
            pending += self._workspace_listing_hint()
            self._pending_observation = pending
            self._empty_streak = 0  # declared FINAL_ANSWER → not empty
            return "continue"

        if not code_blocks:
            pending = (
                "No Python code block detected. Please provide runnable Python in a "
                "```python fenced block, or output `[FINAL_ANSWER]` on its own line "
                "after your outcome.json has actually been written."
            )
            if truncated_note:
                pending += truncated_note
            # v3: count consecutive empty (no code, no final) turns.
            self._empty_streak += 1
            if self._empty_streak >= self._empty_streak_limit:
                self.terminated_by = "no_progress"
                self._record({
                    "role": "system",
                    "event": "no_progress",
                    "empty_streak": self._empty_streak,
                    "turn_idx": self._turn_idx,
                })
                self._pending_observation = (
                    f"[Framework note] {self._empty_streak} consecutive turns "
                    "produced neither a Python code block nor [FINAL_ANSWER] "
                    "— terminating to avoid a stuck loop."
                )
                return "done"
        else:
            # Reset the streak whenever the turn actually ran code.
            self._empty_streak = 0

        self._pending_observation = pending
        return "continue"

    def _verify_final_answer(self) -> tuple[bool, str]:
        """v2 (#2): check workspace/outcome.json exists and parses.

        v4 : also checks mtime — if outcome.json was written BEFORE
        the most recent code execution, it's likely stale (leftover from a
        prior step) and the model forgot to regenerate it.
        """
        if self._workspace is None:
            return True, ""
        path = self._workspace / "outcome.json"
        if not path.exists():
            return False, "was not written to the workspace"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return False, f"could not be parsed as JSON ({type(exc).__name__}: {exc})"
        if not isinstance(data, dict):
            return False, "must be a JSON object (dict) at the top level"
        # v4: mtime freshness check — detect stale outcome.json
        if hasattr(self, "_last_exec_time") and self._last_exec_time is not None:
            try:
                mtime = path.stat().st_mtime
                if mtime < self._last_exec_time:
                    return False, (
                        "appears stale (written before last code execution). "
                        "Your code ran but did not update outcome.json — "
                        "re-run the code that writes outcome.json."
                    )
            except OSError:
                pass
        return True, ""

    def _workspace_listing_hint(self) -> str:
        """v3 : attach a compact listing of the workspace root so the model
        can distinguish "code did not run" from "code ran but wrote wrong filename".

        Only files under workspace root (depth 1) are listed; size in bytes.
        outcome.json is highlighted at the top.
        """
        if self._workspace is None:
            return ""
        try:
            entries: list[tuple[str, int]] = []
            for p in sorted(self._workspace.iterdir()):
                if p.is_file():
                    try:
                        entries.append((p.name, p.stat().st_size))
                    except OSError:
                        entries.append((p.name, -1))
                elif p.is_dir():
                    entries.append((p.name + "/", -1))
        except OSError:
            return ""
        if not entries:
            return "\n\n[Framework note] Workspace root is empty."
        # Highlight outcome.json separately if present.
        outcome_line = ""
        for name, size in entries:
            if name == "outcome.json":
                outcome_line = f"  outcome.json  ({size} bytes)"
                break
        lines = [f"  {n}  ({s} bytes)" if s >= 0 else f"  {n}" for n, s in entries]
        hint = "\n\n[Framework note] Current workspace contents (files/dirs):\n" + "\n".join(lines)
        if outcome_line:
            hint += "\n  → outcome.json IS present; if you meant it as your answer, send `[FINAL_ANSWER]` alone."
        else:
            hint += "\n  → outcome.json is NOT present; write it before sending [FINAL_ANSWER]."
        return hint

    def run(self) -> str:
        """v2 (#13): returns terminated_by."""
        observation = ""
        try:
            for turn in range(self.max_turns):
                self._turn_idx = turn
                # v2 (#14): last-turn nudge.
                if turn == self.max_turns - 1 and turn > 0:
                    hint = (
                        f"[Framework note] This is the FINAL turn "
                        f"({turn + 1}/{self.max_turns}). If outcome.json is not "
                        "yet written, write it now in a single Python block and "
                        "then output [FINAL_ANSWER] on its own line."
                    )
                    observation = f"{observation}\n\n{hint}" if observation else hint
                try:
                    status = self.step(observation)
                except Exception:
                    # v3: preserve executor_crash / stuck_loop / no_progress
                    # signals set by step() before falling back to api_error.
                    self.terminated_by = self.terminated_by or "api_error"
                    return self.terminated_by
                observation = self._pending_observation
                self._pending_observation = ""
                if status == "done":
                    # v3: step() also returns "done" when the circuit
                    # breaker fires (stuck_loop / no_progress). In that case
                    # terminated_by is already set — DON'T overwrite it with
                    # final_answer.
                    if self.terminated_by in ("stuck_loop", "no_progress"):
                        return self.terminated_by
                    self.terminated_by = "final_answer"
                    return "final_answer"
            self.terminated_by = "max_turns_exhausted"
            return "max_turns_exhausted"
        finally:
            self._record({
                "role": "system",
                "event": "terminated",
                "terminated_by": self.terminated_by,
                "usage_summary": self.usage_summary,
                "turns_executed": self._turn_idx + 1,
            })

    def teardown(self) -> None:
        if self.executor is not None:
            self.executor.teardown()
        if self._trajectory_writer is not None:
            try:
                self._trajectory_writer.close()
            except OSError:
                pass
            self._trajectory_writer = None

    def dump_trajectory(self, path: Path) -> None:
        """v2 (#5): no-op when streaming already wrote everything."""
        p = Path(path)
        if self._trajectory_path is not None and p.resolve() == self._trajectory_path.resolve():
            return
        with open(p, "w", encoding="utf-8") as f:
            for entry in self.trajectory:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
# v2 (#7): three-level fallback for code block extraction.
_PYTHON_STRICT = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
_PYTHON_LOOSE = re.compile(r"```(?:py|python)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_FENCE_ANY_UNLABELED = re.compile(r"```[ \t]*\n(.*?)```", re.DOTALL)
_FENCE_LABELED = re.compile(r"```([A-Za-z][A-Za-z0-9+_-]*)\s*\n.*?```", re.DOTALL)


def _extract_all_python(text: str) -> list[str]:
    """Extract Python code blocks with three-level fallback (v2 #7).

    Order:
      1. ```python (canonical)
      2. ```py / ```Python (case-insensitive)
      3. Unlabeled ``` fences — but only when the reply has no OTHER labeled
         fence. This blocks ```json / ```bash / ```sql from tunneling into
         the Python executor.
    """
    blocks = [m.group(1) for m in _PYTHON_STRICT.finditer(text)]
    if blocks:
        return [b.strip() for b in blocks if b.strip()]
    blocks = [m.group(1) for m in _PYTHON_LOOSE.finditer(text)]
    if blocks:
        return [b.strip() for b in blocks if b.strip()]
    labeled = [m.group(1).lower() for m in _FENCE_LABELED.finditer(text)]
    if labeled and not any(lbl in {"py", "python"} for lbl in labeled):
        return []
    blocks = [m.group(1) for m in _FENCE_ANY_UNLABELED.finditer(text)]
    return [b.strip() for b in blocks if b.strip()]


_FINAL_ANSWER_RE = re.compile(r"(?m)^\s*\[FINAL_ANSWER\]\s*$")
_ANY_FENCE_RE = re.compile(r"```[A-Za-z0-9+_-]*\s*\n.*?```", re.DOTALL)


def _detect_final_answer(reply: str) -> bool:
    """v2 (#9): [FINAL_ANSWER] must be on its own line, outside code fences."""
    outside_code = _ANY_FENCE_RE.sub("", reply)
    return bool(_FINAL_ANSWER_RE.search(outside_code))


def _extract_python(text: str) -> str | None:
    blocks = _extract_all_python(text)
    return blocks[0] if blocks else None


# v3: salvage helper for max_tokens-truncated replies.
_TRUNCATED_OPEN_RE = re.compile(
    r"```(?:python|py)?\s*\n", re.IGNORECASE
)


def _salvage_truncated_python(text: str) -> str | None:
    """When ``resp.stop_reason == "length"`` and the strict regex found no
    complete ```python``` block, try to extract the body of the LAST opening
    fence up to end-of-text. Returns None when no salvageable block is found.

    Heuristics used:
      * odd count of ``` markers → likely unclosed fence
      * take everything after the LAST opening fence (```python / ```py /
        bare ```) as the truncated body
      * strip trailing whitespace; require ≥ 1 non-empty line
    """
    if text.count("```") % 2 == 0:
        return None  # fences look balanced — nothing to salvage
    last_open = None
    for m in _TRUNCATED_OPEN_RE.finditer(text):
        last_open = m
    if last_open is None:
        return None
    body = text[last_open.end():].rstrip()
    # Guard against runaway prose after a stray backtick — insist on at least
    # one line that looks vaguely like code (contains =, def, import, print,
    # return, or a common data-analysis keyword).
    if not body.strip():
        return None
    if not re.search(r"^(?:\s*)(?:import|from|def|class|print|return|for|while|if|with|[A-Za-z_][\w\.]*\s*=)",
                     body, re.MULTILINE):
        return None
    return body


def load_model_config(models_yaml: Path, name: str) -> ModelConfig:
    import os

    import yaml  # type: ignore[import-untyped]

    with open(models_yaml, "r", encoding="utf-8") as f:
        cfg_all = yaml.safe_load(f)
    for m in cfg_all.get("models", []):
        if m["name"] == name:
            api_key = os.environ.get(m.get("api_key_env", ""), m.get("api_key", ""))
            return ModelConfig(
                name=m["name"],
                api_base=m["api_base"],
                api_key=api_key,
                model_id=m.get("model_id", m["name"]),
                temperature=m.get("temperature", 0.0),
                # v2 (#4): default None; per-model override kicks in only when set.
                max_tokens=m.get("max_tokens"),
                max_retries=int(m.get("max_retries", 4)),
            )
    raise KeyError(f"Model {name!r} not found in {models_yaml}")
