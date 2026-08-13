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

Behavior notes
--------------
  * Deadline-aware retry with exponential backoff on chat completions.
  * `workspace/outcome.json` is verified (existence + parseable) before a
    `[FINAL_ANSWER]` marker is accepted.
  * `ChatResponse` dataclass carries `usage` / `stop_reason` so the runner
    can surface budget / termination reasons in `score.json`.
  * `max_tokens` defaults to `None` (vendor default) — the adapter refuses
    to invent a cap.
  * Trajectory is streamed to disk on every event; full tool output plus
    `exec_logs` are persisted for post-hoc audit.
  * Three-level fallback for code-block extraction (fenced → tagged →
    heuristic); `[FINAL_ANSWER]` must be on its own line, outside fences.
  * Optional `MAX_BLOCKS_PER_TURN` cap (default: unlimited).
  * `run()` returns `terminated_by` so aggregate.py can classify runs.
  * Last-turn nudge is emitted before `max_turns` is exhausted so the
    model has one guaranteed chance to submit.
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
    # 逐次下发的采样 seed。None = 不发这个字段。
    # 我们的多 seed 是「同一道题跑 N 次取分布」，之前 seed 只用来给 trial 目录起名
    # 和播种本地 random/numpy，压根没告诉模型侧 —— 于是同一个 seed 复跑并不比不同
    # seed 更可复现，replication_check 那一层就没了立足点。注意：网关是否真的认这个
    # 字段无法先验断定，所以配套记了 system_fingerprint + seed_sent 进轨迹，事后能
    # 反查「发了没有 / 认没认」。
    seed: int | None = None

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
    # vendor "thinking" text, kept ONLY for accounting and
    # audit. It is deliberately NOT concatenated into ``content`` and never goes
    # back into the message history: doing either would feed the model its own
    # chain-of-thought and change the task. The point of capturing it is that it
    # is billed as completion_tokens while contributing nothing to the visible
    # reply, so a run that keeps hitting stop_reason="length" with a short answer
    # is otherwise inexplicable from the trajectory alone.
    reasoning: str = ""
    # 网关/厂商回的采样指纹。配合 seed 用：seed
    # 发出去了不等于对方认，指纹变了就说明后端换了（权重、量化、部署都算），届时
    # 「同 seed 应当同分」这个假设本身就不成立，得靠它把不可复现的原因分开。
    system_fingerprint: str | None = None


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
            # pin the SDK's own retry budget to 0 and keep
            # retries in ONE place — chat() below. Pre-v4 the same models.yaml
            # `max_retries` fed both layers, so they multiplied: max_retries=4
            # meant 5 SDK attempts inside each of 5 app attempts = 25 requests,
            # with the SDK's sleeps happening inside a single app attempt where
            # chat()'s deadline cannot see or truncate them. That is how a plain
            # api_error could burn the whole wall budget and get reported as a
            # timeout instead. One layer, deadline-aware, is both fewer requests
            # and an honest terminal status.
            max_retries=0,
        )

    def chat(self, messages: list[dict], *, deadline: float | None = None) -> ChatResponse:
        """Streaming chat call with exponential-backoff retry (v2 #1).

        `deadline` (time.time seconds) truncates retry sleeps so the loop
        cannot outrun the per-step timeout budget.

        this is now the ONLY retry layer — the SDK's own
        retry is pinned to 0 in __init__. Consequently ``cfg.max_retries``
        finally means what models.yaml says it means (max_retries=4 → at most 5
        requests, not 25), and every backoff sleep is visible to ``deadline``.
        The loop also refuses to start an attempt it cannot finish inside the
        remaining budget, so a retry can no longer push a plain api_error past
        the wall-clock watchdog and get relabelled as a timeout.

        The retryable set is unchanged on purpose (see ``_RETRY_STATUS``);
        widening it is item 5, which is out of scope here.
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
                # give up when the remaining budget cannot
                # hold a sleep plus a plausible next attempt. Sleeping right up
                # to the deadline and then firing a request that is guaranteed to
                # be cut off wastes the budget and hides the real cause.
                if remaining <= 0 or remaining <= delay + 1.0:
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
        # 只在配了 seed 时才发这个字段，没配就一个字
        # 都不加 —— 有的网关见到不认识的键会直接 400。
        if getattr(self.cfg, "seed", None) is not None:
            kwargs["seed"] = self.cfg.seed

        try:
            stream = self._client.chat.completions.create(**kwargs)
        except TypeError:
            # Older gateway shims may reject stream_options.
            # seed 一并退掉。TypeError 是「这个 SDK/shim 的签名里没有这个
            # 参数」，seed 和 stream_options 都属于新加的可选键，退化时要一起摘掉，
            # 否则第二次调用还是同一个 TypeError。
            kwargs.pop("stream_options", None)
            kwargs.pop("seed", None)
            stream = self._client.chat.completions.create(**kwargs)

        chunks: list[str] = []
        reasoning_chunks: list[str] = []
        stop_reason: str | None = None
        usage: dict | None = None
        model_id: str | None = None
        system_fingerprint: str | None = None
        for event in stream:
            u = getattr(event, "usage", None)
            if u is not None:
                usage = _usage_to_dict(u)
            m = getattr(event, "model", None)
            if m:
                model_id = m
            # 指纹在流里任何一个 chunk 上都可能出现，
            # 取到就留最后一个非空值。
            sf = getattr(event, "system_fingerprint", None)
            if sf:
                system_fingerprint = sf
            if not event.choices:
                continue
            choice = event.choices[0]
            delta = choice.delta
            if delta and getattr(delta, "content", None):
                chunks.append(delta.content)
            # collect reasoning_content separately. Vendors
            # spell it reasoning_content (DeepSeek/GLM) or reasoning (others);
            # both are accounting-only here.
            if delta is not None:
                for attr in ("reasoning_content", "reasoning"):
                    rc = getattr(delta, attr, None)
                    if isinstance(rc, str) and rc:
                        reasoning_chunks.append(rc)
                        break
            fr = getattr(choice, "finish_reason", None)
            if fr:
                stop_reason = fr

        return ChatResponse(
            content="".join(chunks),
            stop_reason=stop_reason,
            usage=usage,
            model=model_id or self.cfg.model_id,
            reasoning="".join(reasoning_chunks),
            system_fingerprint=system_fingerprint,
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
    """Persist code + full output for post-hoc audit (v2 #6, 李-3)."""
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
        # prepend a determinism prologue driven by BENCH_SEED so any
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
        required_outputs: list[str] | None = None,
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
        # circuit breaker counters.
        self._empty_streak: int = 0           # consecutive turns with no code AND no [FINAL_ANSWER]
        self._repeat_streak: int = 0          # consecutive turns with the same reply body
        self._last_reply_hash: str | None = None
        self._empty_streak_limit = int(empty_streak_limit)
        self._repeat_streak_limit = int(repeat_streak_limit)
        # artifact-based progress signal. A turn that emits
        # no code block is NOT necessarily a stalled turn — a job started in an
        # earlier turn may still be writing the task's required outputs. We keep a
        # (mtime_ns, size) fingerprint per declared required output and treat any
        # new/changed file as progress, so the no_progress circuit breaker stops
        # killing runs that are demonstrably still producing deliverables.
        self._required_outputs: list[str] = list(required_outputs or [])
        self._artifact_fingerprint: dict[str, tuple[int, int]] = {}
        # per-turn accounting. Item 16 asks to re-check
        # max_turns after the protocol fixes (items 9/10) because "成功 run 被浪费
        # 的 turn 拖垮" — but pre-v4 there was no way to tell a run that needed 40
        # turns of real work from one that burned 30 turns on replies the framework
        # could not parse. These counters make that difference visible in
        # score.json / index.jsonl, so the new max_turns can be picked from data
        # instead of guessed. They are pure bookkeeping: nothing reads them to make
        # a control-flow decision.
        self.turn_stats: dict = {
            "turns_with_code": 0,      # a turn that actually ran at least one block
            "turns_no_code": 0,        # no block AND no [FINAL_ANSWER] — wasted
            "turns_exec_failed": 0,    # ran code, at least one block failed
            "turns_empty_reply": 0,    # gateway returned content="" (item 7)
            "turns_final_answer": 0,   # [FINAL_ANSWER] declared (accepted or not)
        }
        # parsed schema.json, filled by setup(). Used for the
        # advisory structure self-check at [FINAL_ANSWER] time.
        self._schema: dict | None = None
        #/ outcome.json 新鲜度检查的状态。原版存的是「上一次
        # 代码执行的时刻」并拿它跟文件 mtime 比大小；那个比较结构上恒成立（详见
        # _verify_final_answer 里的说明）。这里换成「出处」判据：交卷时那份
        # outcome.json 到底是这次 run 产出的，还是 setup 之前就躺在盘上的。
        self._outcome_fp_base: tuple[int, int] | None = None  # setup 时盘上就有的那份
        self._ran_any_code: bool = False        # setup 以来是否执行过任何代码块

    def setup(self, task_dir: Path, workspace: Path) -> None:
        self._workspace = workspace
        instruction = (task_dir / "instruction.md").read_text(encoding="utf-8")
        schema = (task_dir / "schema.json").read_text(encoding="utf-8")
        # keep the parsed schema for the advisory self-check
        # at [FINAL_ANSWER]. A malformed schema.json must not break the run, so a
        # parse failure just disables the check.
        try:
            parsed_schema = json.loads(schema)
            self._schema = parsed_schema if isinstance(parsed_schema, dict) else None
        except json.JSONDecodeError:
            self._schema = None
        if self.executor is None:
            self.executor = SubprocessExecutor(workspace, timeout=self._per_step_timeout)
        self.executor.setup()
        # seed this process's own RNGs from exported BENCH_SEED so
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
        # baseline the required-output fingerprint BEFORE the
        # first turn, so files that were already in the workspace at setup time
        # are not mistaken for progress the agent made.
        self._artifact_fingerprint = self._snapshot_artifacts()
        # 同一时机给 outcome.json 单独拍一张基线指纹。交卷时
        # 「这份 outcome.json 是不是这次 run 写出来的」就靠它判，不再比时间戳。
        self._outcome_fp_base = self._outcome_fingerprint()
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

    def _snapshot_artifacts(self) -> dict[str, tuple[int, int]]:
        """ (mtime_ns, size) per existing required output."""
        snap: dict[str, tuple[int, int]] = {}
        if self._workspace is None:
            return snap
        for rel in self._required_outputs:
            p = self._workspace / rel
            try:
                st = p.stat()
            except OSError:
                continue  # not written yet (or unreadable) — simply absent
            snap[rel] = (st.st_mtime_ns, st.st_size)
        return snap

    def _outcome_fingerprint(self) -> tuple[int, int] | None:
        """ outcome.json 的 (mtime_ns, size)，不存在时 None。

        单独一个函数是因为 outcome.json 的新鲜度判据和 required_outputs 的
        「有进展」判据用途不同：后者每轮都会刷新自己的基线（改动只算一次进展），
        前者要的是**整条 run 的**基线，不能被逐轮刷新掉。
        """
        if self._workspace is None:
            return None
        try:
            st = (self._workspace / "outcome.json").stat()
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def _artifact_progress(self) -> list[str]:
        """Names of required outputs that appeared or changed since last check.

        this is the second progress signal alongside "the turn
        emitted a code block". A long-running computation started in an earlier
        turn can keep writing deliverables while the model itself says nothing;
        pre-v4 those turns counted as empty and the no_progress circuit breaker
        killed a run that was in fact still producing the required artifacts.
        Calling this UPDATES the stored fingerprint, so each change is counted
        as progress exactly once.
        """
        if not self._required_outputs:
            return []
        current = self._snapshot_artifacts()
        changed = [
            rel for rel, fp in current.items()
            if self._artifact_fingerprint.get(rel) != fp
        ]
        self._artifact_fingerprint = current
        return changed

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
        # never put an empty assistant turn back into the
        # history. Several gateways return content="" (typically together with
        # stop_reason="tool_calls" or "length" after the whole budget went into
        # reasoning_content), and replaying `{"role":"assistant","content":""}`
        # on the next request makes the API reject the WHOLE conversation with a
        # 400 — one empty reply killed the run from that point on. We keep the
        # real (empty) reply for parsing and the trajectory, and store a visible
        # placeholder in the history so the transcript stays well-formed and the
        # model can see that its previous turn came back blank.
        empty_reply = not reply.strip()
        if empty_reply:
            history_content = (
                "[Framework note] (your previous reply arrived empty"
                + (f", stop_reason={resp.stop_reason}" if resp.stop_reason else "")
                + ")"
            )
        else:
            history_content = reply
        self.messages.append({"role": "assistant", "content": history_content})
        entry = {
            "role": "assistant",
            "content": reply,
            "stop_reason": resp.stop_reason,
            "usage": resp.usage,
            "model": resp.model,
            "turn_idx": self._turn_idx,
        }
        if empty_reply:
            entry["empty_reply"] = True
            entry["history_placeholder"] = history_content
            self.turn_stats["turns_empty_reply"] += 1  ## reasoning is recorded for accounting/audit only —
        # it is NOT appended to content and NOT sent back in the history.
        if getattr(resp, "reasoning", ""):
            entry["reasoning_content"] = resp.reasoning
            entry["reasoning_chars"] = len(resp.reasoning)
        # 记下「这轮发出去的 seed」和「对方回的采样
        # 指纹」。两个都是事后反查用的：seed_sent 证明我们确实下发了（而不是只拿来给
        # 目录起名），system_fingerprint 变了说明后端换了、同 seed 同分的前提不再成立。
        # 都用 getattr 兜着：测试里的假 client 没有 .cfg，ChatResponse 也可能是老形状。
        _seed_sent = getattr(getattr(self.model, "cfg", None), "seed", None)
        if _seed_sent is not None:
            entry["seed_sent"] = _seed_sent
        if getattr(resp, "system_fingerprint", None):
            entry["system_fingerprint"] = resp.system_fingerprint
        self._record(entry)
        if resp.usage:
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                v = resp.usage.get(k)
                if isinstance(v, (int, float)):
                    self.usage_summary[k] += int(v)
            self.usage_summary["calls"] += 1

        #/ repeat detection. A turn that declares
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

        # if the reply was cut off by max_tokens ("stop_reason=length")
        # and the strict regex found nothing, try to salvage a truncated block:
        # take from the last opening ```python (or ```py) fence to the end and
        # mark it truncated=True. If successful, also stitch a hint into the
        # observation asking the model to finish the truncated code next turn.
        length_truncated = (resp.stop_reason == "length")
        truncated_note = ""
        truncated_block_idxs: set[int] = set()  # track salvaged blocks
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

        # optional cap on blocks-per-turn.
        capped_note = ""
        if self.max_blocks_per_turn is not None and len(code_blocks) > self.max_blocks_per_turn:
            n_all = len(code_blocks)
            code_blocks = code_blocks[: self.max_blocks_per_turn]
            capped_note = (
                f"\n\n[Framework note] {n_all} code blocks emitted; only the first "
                f"{self.max_blocks_per_turn} were executed. Please concentrate your "
                f"work into a smaller number of runnable blocks."
            )

        # 记住「这一轮模型到底写了几个代码块」。
        # 下面的拼接会把 code_blocks 塌成一个元素，塌完之后就再也数不出来了——想统计
        # 「多代码块」这个形态的发生率，只能回头重新解析 reply。该计数字段直接
        # 写进轨迹（n_blocks），比重新解析可靠，并进来。取的是**限流之后**的数，
        # 也就是真正被执行的块数，consistent with the earlier revision。
        n_source_blocks = len(code_blocks)

        # multiple code blocks in ONE assistant message share
        # a namespace as far as the model is concerned — it writes `df = ...` in
        # block 1 and `df.head()` in block 2. The pre-v4 loop ran each block as a
        # separate subprocess, so block 2 died on NameError and the model spent
        # turns re-deriving state it thought it had. Concatenate them into a single
        # script instead. This deliberately does NOT introduce a persistent REPL
        # across turns (that would change the execution model and leak state
        # between steps); the sharing is scoped to one message, exactly as # item 9 asks.
        merged_note = ""
        if len(code_blocks) > 1:
            n_parts = n_source_blocks
            joined = "\n\n".join(
                f"# --- block {j + 1}/{n_parts} (concatenated by framework) ---\n{c}"
                for j, c in enumerate(code_blocks)
            )
            # Any part salvaged from a truncated fence taints the merged script.
            was_truncated = bool(truncated_block_idxs)
            code_blocks = [joined]
            truncated_block_idxs = {0} if was_truncated else set()
            merged_note = (
                f"\n\n[Framework note] The {n_parts} code blocks in your last reply "
                "were concatenated in order and run as ONE script, so they share a "
                "namespace. Note that state does NOT carry over between turns — "
                "each turn starts a fresh interpreter, so re-load what you need."
            )

        outputs: list[str] = []
        any_failed = False
        for i, code in enumerate(code_blocks):
            log_name = f"block_{self._turn_idx:02d}_{i:02d}"
            # executor faults (OSError, subprocess errors) are
            # infrastructure crashes, not the model's fault — tag them as
            # executor_crash so aggregate.py can drop them from the mean.
            try:
                try:
                    ok, output = self.executor.run(code, log_name=log_name)  # type: ignore[call-arg]
                except TypeError:
                    # Backward compat: executors without log_name kwarg.
                    ok, output = self.executor.run(code)  # type: ignore[union-attr]
                # 记「跑过代码」这个事实即可，不再记时刻。
                self._ran_any_code = True
            except TimeoutError:
                # 墙钟看门狗（runner 里的 signal.alarm handler）抛的就是
                # TimeoutError，而 TimeoutError 是 OSError 的子类，正好被下面那个
                # `except (OSError, subprocess.SubprocessError)` 接住 —— 它 raise 出去
                # 之后被 run() 的 `except Exception` 收成 executor_crash，看门狗那条
                # 「墙钟超时」的信号就此消失。更糟的是 signal.alarm 是一次性的：一旦
                # 这一响被改写成别的语义，就再没有第二次闹钟，run 会一路跑到
                # max_turns（已实测：alarm 设在 1s，实际跑满 12/12 轮）。所以先原样
                # 抛回去，让 runner 的 `except TimeoutError` 认领它，wall_timeout 才
                # 报得出来。执行器自己的超时不走这里 —— SubprocessExecutor 和
                # DockerExecutor 都在内部把 subprocess.TimeoutExpired 接掉并返回
                # (False, "[timeout after ...]")，压根不会传上来。
                raise
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
                "n_blocks": n_source_blocks,  # v4.1 合并（取自 the earlier revision's item 9）
                "truncated": i in truncated_block_idxs,  #"block_idx": i,
                "turn_idx": self._turn_idx,
                "log_name": log_name,
            })
            # 第 9 条把多个块塌成一个脚本之后，原来的
            # 「Block i/N」标号就恒等于「Block 1/1」，跟同一条观察里
            # merged_note 说的「你那 3 个块被拼成了一个」自相矛盾。改成按真实
            # 来源块数写，consistent with the earlier "N block(s) run as one script"；
            # 单块时仍然是 Block 1/1，与基线一致。
            if n_source_blocks > 1:
                outputs.append(
                    f"[{n_source_blocks} blocks run as ONE script, ok={ok}]\n{trunc}"
                )
            else:
                outputs.append(f"[Block {i + 1}/{len(code_blocks)} ok={ok}]\n{trunc}")

        pending = "\n\n".join(outputs) if outputs else ""
        # per-turn accounting for the max_turns review.
        if code_blocks:
            self.turn_stats["turns_with_code"] += 1
            if any_failed:
                self.turn_stats["turns_exec_failed"] += 1
        if capped_note:
            pending += capped_note
        if merged_note:
            pending += merged_note
        if truncated_note:
            pending += truncated_note

        # refresh the required-output fingerprint on EVERY turn
        # so a change is credited once. The result only matters in the
        # "no code block this turn" branch below, but refreshing here keeps the
        # code-block path from leaving stale state that would later hand a free
        # pass to a genuinely idle turn.
        artifact_changed = self._artifact_progress()

        # anchor [FINAL_ANSWER] to its own line, outside fences.
        declared_final = _detect_final_answer(reply)

        if declared_final:
            self.turn_stats["turns_final_answer"] += 1  ## accept [FINAL_ANSWER] whenever outcome.json is already
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
                # advisory schema self-check. Pre-v4 the only
                # gate at submission time was "does outcome.json exist and parse",
                # so a file holding two of twenty sections was accepted as a
                # finished answer with no trace of the fact anywhere. We now record
                # the gaps in the trajectory and tell the model once — but we still
                # accept the submission, exactly as item 11 asks (observable, not
                # enforced). The scorer remains the authority on correctness.
                gaps = self._schema_selfcheck()
                if gaps:
                    shown = gaps[:12]
                    self._record({
                        "role": "system",
                        "event": "schema_selfcheck_warning",
                        "gap_count": len(gaps),
                        "gaps": gaps,
                        "turn_idx": self._turn_idx,
                    })
                    pending += (
                        f"\n\n[Framework note] outcome.json was accepted, but a "
                        f"structure self-check found {len(gaps)} gap(s) against "
                        f"schema.json: " + "; ".join(shown)
                        + ("; ..." if len(gaps) > len(shown) else "")
                        + ". This is a warning only — the answer has been "
                        "submitted."
                    )
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
            # this is the turn class item 16 calls "浪费的
            # turn" — the model spoke but the framework got nothing runnable out of
            # it. Counted whether or not the artifact-progress signal spares the
            # circuit breaker, because for max_turns budgeting what matters is that
            # the turn produced no executable work.
            self.turn_stats["turns_no_code"] += 1
            pending = (
                "No Python code block detected. Please provide runnable Python in a "
                "```python fenced block, or output `[FINAL_ANSWER]` on its own line "
                "after your outcome.json has actually been written."
            )
            if truncated_note:
                pending += truncated_note
            # a turn with no code block still counts as progress
            # when a declared required output appeared or changed — the run is
            # visibly producing deliverables, so killing it as no_progress would
            # throw away a trial that was working. Reset the streak instead, and
            # tell the model what the framework saw so the signal is auditable
            # from the trajectory alone.
            if artifact_changed:
                self._empty_streak = 0
                pending += (
                    "\n\n[Framework note] Required output(s) "
                    f"{artifact_changed} changed on disk since the last turn, so "
                    "this turn counts as progress. Keep going, then send "
                    "[FINAL_ANSWER] once everything is written."
                )
                self._record({
                    "role": "system",
                    "event": "artifact_progress",
                    "changed": artifact_changed,
                    "turn_idx": self._turn_idx,
                })
                self._pending_observation = pending
                return "continue"
            # count consecutive empty (no code, no final) turns.
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

    def _schema_selfcheck(self) -> list[str]:
        """ advisory structure check on outcome.json.

        Returns a list of human-readable gaps ("part3_model.auc is missing").
        An EMPTY list means "nothing to report" — it is not a pass/fail verdict
        and the caller must NOT block submission on it. item 11 asks for
        an observable warning, explicitly not a forced retry loop: turning this
        into a gate would let a schema bug hold a finished run hostage.

        This is a deliberately shallow structural walk, not JSON-Schema
        validation: top-level ``properties`` keys are treated as expected
        sections, and each section's own ``required`` list is checked one level
        down. ``jsonschema`` is not a dependency of this framework and adding one
        for an advisory message is not worth it; types are the scorer's job.

        除了「少了什么」，也报「多了什么」。
        两个方向查的是不同的病：少一个 section 是半成品，多一个顶层键通常是模型
        自己发明了字段名（``part_3`` 写成 ``part3_extra``），判分时那一段直接是 0
        分，但从产物本身看不出来。多余键同样只告警不拦。
        """
        if self._schema is None or self._workspace is None:
            return []
        props = self._schema.get("properties")
        if not isinstance(props, dict):
            return []
        try:
            data = json.loads((self._workspace / "outcome.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []  # the exists/parses check already reported this
        if not isinstance(data, dict):
            return []

        gaps: list[str] = []
        top_required = self._schema.get("required")
        top_required = top_required if isinstance(top_required, list) else list(props)
        for section in props:
            if section not in data:
                if section in top_required:
                    gaps.append(f"{section} (whole section) is missing")
                continue
            sub_schema = props.get(section)
            if not isinstance(sub_schema, dict):
                continue
            sub_required = sub_schema.get("required")
            if not isinstance(sub_required, list):
                continue
            body = data.get(section)
            if not isinstance(body, dict):
                gaps.append(f"{section} should be an object")
                continue
            for key in sub_required:
                if key not in body:
                    gaps.append(f"{section}.{key} is missing")
                elif body[key] is None:
                    gaps.append(f"{section}.{key} is null")
        # v4.1 合并: 反方向——schema 没声明的顶层键。措辞分两档：schema 明确写了
        # additionalProperties=false 时是硬性多余键；缺省（JSON Schema 默认 True）
        # 时只能说「可能是有意扩展」。两档都只告警，不拦交卷。
        extra = sorted(set(data) - set(props))
        if extra:
            strict = self._schema.get("additionalProperties") is False
            head = ("unexpected top-level key(s) not in schema: " if strict
                    else "top-level key(s) not declared in schema (may be intentional): ")
            gaps.append(head + ", ".join(extra))
        return gaps

    def _verify_final_answer(self) -> tuple[bool, str]:
        """ check workspace/outcome.json exists and parses.

        还要判「这份 outcome.json 是不是这次 run 真写出来的」，以免模型
        捡了盘上一份遗留文件当答案交上去。

        判据从「mtime < 上次执行时刻」换成「指纹是否还等于 run
        开始前那一张」。原来的写法结构上恒成立，理由是时序：outcome.json 是**在**
        那次执行期间写的，而 _last_exec_time 记的是那次执行**返回之后**的时刻，所以
        mtime 必然更早。实测同轮写盘 + 交卷的 now - mtime = 0.000055s，判定照样成立。
        后果不是「偶尔误伤」而是「只要模型跑过代码就永远交不上卷」：8 轮 max_turns 的
        回放里，模型每轮都正确写盘并交卷，8 次全被退回，终态 max_turns_exhausted
        （见 /tmp/luo3_severity.py）。真实 trial 里 0 次命中只是因为那批 run 早于这
        棵树，不是因为它不会触发。

        新判据只在一种情形下拒：run 开始前盘上就有 outcome.json，且到交卷时它的
        (mtime_ns, size) 一个字节都没变 —— 那确实不是这次 run 的产物。基线是 setup()
        拍的，所以「模型自己第一轮写、第五轮才交卷」不受影响。
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
        # 遗留文件检查。base 为 None（run 开始时没有这个文件）
        # 时整条检查不适用，直接放行。
        base = getattr(self, "_outcome_fp_base", None)
        if base is not None and self._outcome_fingerprint() == base:
            self._record({
                "role": "system",
                "event": "stale_outcome_rejected",
                "turn_idx": self._turn_idx,
                "ran_any_code": self._ran_any_code,
            })
            return False, (
                "is byte-for-byte the file that was already in the workspace "
                "before this run started — it is not your answer. Write "
                "outcome.json yourself, then send [FINAL_ANSWER] again."
            )
        return True, ""

    def _workspace_listing_hint(self) -> str:
        """ attach a compact listing of the workspace root so the model
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
                # last-turn nudge.
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
                except TimeoutError:
                    # 墙钟看门狗的信号。step() 里只在执行器那一处把它原样
                    # 放行；chat 那处故意不放行 —— runner 的 _watchdog_fired 已经能
                    # 把状态名恢复成 wall_timeout，而 api_error 那条轨迹记录本身是
                    # 真实且有诊断价值的信号（实测轨迹里就有这种形态）。这里是最后
                    # 一道 —— 不能被下面那个 `except Exception` 收成 api_error。抛出
                    # 去让 runner 的 `except TimeoutError` 认领，terminated_by 才会是
                    # wall_timeout。terminated_by 保持 step() 里的原值（通常是 None），
                    # 由 runner 用 _watchdog_fired 覆盖成 wall_timeout。
                    raise
                except Exception:
                    # preserve executor_crash / stuck_loop / no_progress
                    # signals set by step() before falling back to api_error.
                    self.terminated_by = self.terminated_by or "api_error"
                    return self.terminated_by
                observation = self._pending_observation
                self._pending_observation = ""
                if status == "done":
                    # step() also returns "done" when the circuit
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
                # turn budget breakdown, so "was max_turns
                # too tight" can be answered from the data instead of debated.
                "turn_stats": dict(self.turn_stats),
                "max_turns": self.max_turns,
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
#
# fence 标记从「只认反引号」放宽为「反引号或
# 波浪号，任意 ≥3 长度」，并容忍语言标签后的空格与 CRLF。CommonMark 本来就允许
# ~~~ 围栏，模型在代码体自身含反引号时会主动改用波浪号；```python3 也是常见写法。
# 三级回退的顺序不变（strict python → py/python/python3 → 无标签），所以这次放宽
# 只会「多认」，不会把一个已经匹配上的块重新归类。
_PYTHON_STRICT = re.compile(
    r"(?:`{3,}|~{3,})[ \t]*python[ \t]*\r?\n(.*?)(?:`{3,}|~{3,})",
    re.DOTALL | re.IGNORECASE,
)
_PYTHON_LOOSE = re.compile(
    r"(?:`{3,}|~{3,})[ \t]*(?:py|python|python3)[ \t]*\r?\n(.*?)(?:`{3,}|~{3,})",
    re.DOTALL | re.IGNORECASE,
)
_FENCE_ANY_UNLABELED = re.compile(
    r"(?:`{3,}|~{3,})[ \t]*\r?\n(.*?)(?:`{3,}|~{3,})",
    re.DOTALL,
)
_FENCE_LABELED = re.compile(
    r"(?:`{3,}|~{3,})[ \t]*([A-Za-z][A-Za-z0-9+_-]*)[ \t]*\r?\n.*?(?:`{3,}|~{3,})",
    re.DOTALL,
)


def _extract_all_python(text: str) -> list[str]:
    """Extract Python code blocks with three-level fallback (v2 #7).

    Order:
      1. ```python (canonical)
      2. ```py / ```Python (case-insensitive)
      3. Unlabeled ``` fences — but only when the reply has no OTHER labeled
         fence. This blocks ```json / ```bash / ```sql from tunneling into
         the Python executor.

    when all three levels come up empty, retry once on a
    fence-normalized copy of the reply. Some vendors wrap code in an XML-ish
    pseudo-fence instead of backticks (DeepSeek's ``<｜｜DSML｜｜python>`` /
    ``</｜｜DSML｜｜python>`` is the case we hit), and the un-normalized text
    yields zero blocks — the turn then looks idle and the run dies on
    no_progress even though the model did emit code. The normalizer is a
    shape-based heuristic (see ``_normalize_vendor_fences``), not a per-vendor
    hardcoded terminator, and it only runs on the fallback path so a reply the
    standard regexes already understood is never rewritten.
    """
    blocks = _extract_standard_python(text)
    if blocks:
        return blocks
    normalized = _normalize_vendor_fences(text)
    if normalized != text:
        return _extract_standard_python(normalized)
    return []


def _extract_standard_python(text: str) -> list[str]:
    """The three fence levels described in _extract_all_python.

    v4.1 合并: 「standard」现在指反引号与波浪号两种 CommonMark 围栏，见上面的正则。
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


# shape-based pseudo-fence detector. Matches a single-line
# tag that (a) looks like a tag — angle brackets, optional leading slash — and
# (b) names python/py inside. We deliberately require a "｜" / "|" separator or
# the literal DSML marker so ordinary inline HTML (``<a href=...>``) and prose
# about Python cannot be mistaken for a fence. Kept as a shape rule rather than
# a vendor table so the next gateway that invents its own terminator is covered
# without another patch.
_PSEUDO_FENCE = re.compile(
    r"<\s*(/?)\s*[^<>\n]{0,60}?(?:DSML|[｜|])[^<>\n]{0,60}?(?:python|py)\b[^<>\n]{0,20}?>",
    re.IGNORECASE,
)


def _normalize_vendor_fences(text: str) -> str:
    """Rewrite vendor pseudo-fences into standard ```python fences.

    Returns the text unchanged when nothing matched, so callers can cheaply
    tell whether normalization is worth a second extraction pass.
    """
    def _sub(m: re.Match) -> str:
        return "\n```\n" if m.group(1) else "\n```python\n"

    out = _PSEUDO_FENCE.sub(_sub, text)
    if out == text:
        return text
    # A vendor that only emits a CLOSING pseudo-fence (open with ```python,
    # close with the tag) leaves an odd number of backtick markers. Close it so
    # the strict regex can see a complete block instead of an unclosed one.
    if out.count("```") % 2 == 1:
        out += "\n```\n"
    return out


_FINAL_ANSWER_RE = re.compile(r"(?m)^\s*\[FINAL_ANSWER\]\s*$")
# 剥围栏时也认波浪号。原先只剥反引号，
# 于是模型在 ~~~ 围栏里举例写的裸 [FINAL_ANSWER] 会被当成真的交卷标记，
# 一个还没做完的 trial 就被提前判定成已交卷。这是同批回归之一，随波浪号的
# 的正则统一一起修掉。
_ANY_FENCE_RE = re.compile(
    r"(?:`{3,}|~{3,})[A-Za-z0-9+_-]*[ \t]*\r?\n.*?(?:`{3,}|~{3,})",
    re.DOTALL,
)


def _detect_final_answer(reply: str) -> bool:
    """v2 (#9): [FINAL_ANSWER] must be on its own line, outside code fences."""
    outside_code = _ANY_FENCE_RE.sub("", reply)
    return bool(_FINAL_ANSWER_RE.search(outside_code))


def _extract_python(text: str) -> str | None:
    blocks = _extract_all_python(text)
    return blocks[0] if blocks else None


# salvage helper for max_tokens-truncated replies.
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
