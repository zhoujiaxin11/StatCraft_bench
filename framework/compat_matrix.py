"""
framework/compat_matrix.py
==========================
model × gateway compatibility probe ("烟囱用例").

Why this exists
---------------
Every protocol defect in P1 list was discovered the expensive way: in a
real scoring round, one model at a time, after the round's numbers were already
polluted. Item 13 asks for the cheap way — a smoke matrix run BEFORE the first
real round that answers, per model:

  * does a plain streaming call come back with content at all?
  * is ``temperature=0`` accepted, or does the gateway reject it?
  * is ``max_tokens`` honoured (does a tiny budget produce stop_reason=length)?
  * is ``seed`` accepted (item 12 — probe only, we do not start sending it)?
  * does an EMPTY assistant message in the history get the whole request
    rejected (item 7's failure mode, verified instead of assumed)?
  * does the vendor stream ``reasoning_content`` (item 8's accounting gap)?
  * are code fences standard backticks, or a vendor pseudo-fence the extractor
    has to normalize (item 10)?

Design notes
------------
* Every probe is READ-ONLY and costs a handful of tokens: tiny prompts, tiny
  ``max_tokens``. Nothing here writes into a trial or a task.
* No probe can fail the run. Each returns a status string plus a detail, and an
  unexpected exception becomes ``status="error"`` — a probe crashing must never
  be the reason a compatibility sweep produces no report.
* This module builds its OWN OpenAI client from ``ModelConfig`` instead of
  reusing ``ModelClient``. Probes need to pass parameters (``seed``) and craft
  histories (an empty assistant turn) that ``ModelClient.chat()`` deliberately
  does not allow, and probing must not require widening the production client's
  API surface.
* ``python3 -m framework.compat_matrix --models a,b --out matrix.md`` writes both
  a JSON record and a Markdown table; the JSON is the artifact to attach to a
  round, the table is for reading.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from framework.agent_adapter import ModelConfig, _extract_all_python, load_model_config

ROOT = Path(__file__).resolve().parent.parent

# Probe outcome vocabulary. Kept tiny and explicit — same reasoning as
# statuses.py: a closed word list is the only way a matrix stays comparable
# across models and rounds.
OK = "ok"                   # behaves as the harness assumes
UNSUPPORTED = "unsupported"  # gateway explicitly refuses the feature
DEGRADED = "degraded"       # works, but not the way the harness assumes
ERROR = "error"             # probe itself could not reach a verdict


@dataclass
class ProbeResult:
    name: str
    status: str
    detail: str = ""

    def as_dict(self) -> dict:
        return {"probe": self.name, "status": self.status, "detail": self.detail}


def _raw_client(cfg: ModelConfig):
    from openai import OpenAI

    # max_retries=0: a probe must report the gateway's first answer, not the
    # answer after the SDK quietly papered over a flake (that is exactly the
    # habit item 6 removed from the production path).
    return OpenAI(
        base_url=cfg.api_base, api_key=cfg.api_key, timeout=cfg.timeout, max_retries=0
    )


def _collect(stream) -> tuple[str, str, str | None]:
    """Drain a streaming response into (content, reasoning, stop_reason)."""
    chunks: list[str] = []
    reasoning: list[str] = []
    stop_reason: str | None = None
    for event in stream:
        if not getattr(event, "choices", None):
            continue
        choice = event.choices[0]
        delta = getattr(choice, "delta", None)
        if delta is not None:
            if getattr(delta, "content", None):
                chunks.append(delta.content)
            for attr in ("reasoning_content", "reasoning"):
                rc = getattr(delta, attr, None)
                if isinstance(rc, str) and rc:
                    reasoning.append(rc)
                    break
        fr = getattr(choice, "finish_reason", None)
        if fr:
            stop_reason = fr
    return "".join(chunks), "".join(reasoning), stop_reason


def _status_code(exc: Exception) -> int | None:
    return getattr(exc, "status_code", None)


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------
PROBE_MARKER = "__PROBE_PLACEHOLDER__"


def probe_basic_stream(cfg: ModelConfig) -> ProbeResult:
    """A plain streaming call must return non-empty content."""
    client = _raw_client(cfg)
    stream = client.chat.completions.create(
        model=cfg.model_id,
        messages=[{"role": "user", "content": "Reply with the single word: ready"}],
        stream=True,
        max_tokens=32,
    )
    content, _reasoning, stop_reason = _collect(stream)
    if content.strip():
        return ProbeResult("basic_stream", OK, f"stop_reason={stop_reason}")
    # Empty content on a trivial prompt is exactly item 7's precondition.
    return ProbeResult(
        "basic_stream", DEGRADED,
        f"empty content on a trivial prompt (stop_reason={stop_reason}) — "
        "the empty-reply placeholder path will be exercised on every turn",
    )


def probe_temperature_zero(cfg: ModelConfig) -> ProbeResult:
    """Some gateways reject temperature=0, which the harness sends by default."""
    client = _raw_client(cfg)
    try:
        stream = client.chat.completions.create(
            model=cfg.model_id,
            messages=[{"role": "user", "content": "Say: ok"}],
            stream=True,
            temperature=0.0,
            max_tokens=16,
        )
        _collect(stream)
        return ProbeResult("temperature_0", OK, "accepted")
    except Exception as exc:  # noqa: BLE001 — a refusal is data, not a failure
        code = _status_code(exc)
        if code == 400:
            return ProbeResult(
                "temperature_0", UNSUPPORTED,
                f"400 on temperature=0 — this model needs --temperature "
                f"({type(exc).__name__})",
            )
        raise


def probe_max_tokens(cfg: ModelConfig) -> ProbeResult:
    """A tiny max_tokens must actually truncate and report stop_reason=length.

    A gateway that ignores max_tokens (or reports stop_reason="stop" on a cut
    reply) breaks the truncated-fence salvage path: the framework only tries to
    salvage when it sees stop_reason == "length".
    """
    client = _raw_client(cfg)
    stream = client.chat.completions.create(
        model=cfg.model_id,
        messages=[{"role": "user", "content": "Count from 1 to 200, one number per line."}],
        stream=True,
        max_tokens=16,
    )
    content, _reasoning, stop_reason = _collect(stream)
    if stop_reason == "length":
        return ProbeResult("max_tokens", OK, "truncation reported as stop_reason=length")
    return ProbeResult(
        "max_tokens", DEGRADED,
        f"stop_reason={stop_reason!r} on a reply cut at 16 tokens "
        f"({len(content)} chars) — the truncated-fence salvage path keys off "
        "'length' and will not fire",
    )


def probe_seed(cfg: ModelConfig) -> ProbeResult:
    """Item 12 probe ONLY: is `seed` accepted? We do not start sending it."""
    client = _raw_client(cfg)
    try:
        stream = client.chat.completions.create(
            model=cfg.model_id,
            messages=[{"role": "user", "content": "Say: ok"}],
            stream=True,
            max_tokens=16,
            seed=7,
        )
        _collect(stream)
        return ProbeResult(
            "seed_param", OK,
            "accepted (acceptance is NOT proof it changes sampling — that needs "
            "a repeated-sampling check, item 12)",
        )
    except Exception as exc:  # noqa: BLE001
        code = _status_code(exc)
        if code in (400, 404, 422):
            return ProbeResult("seed_param", UNSUPPORTED, f"{code} on seed=7")
        raise


def probe_empty_assistant_history(cfg: ModelConfig) -> ProbeResult:
    """Verify item 7's premise instead of assuming it.

    Sends a history containing ``{"role":"assistant","content":""}``. A 400 here
    is the exact failure the placeholder fix prevents; an OK means this gateway
    tolerates it (the placeholder is then harmless but unnecessary).
    """
    client = _raw_client(cfg)
    try:
        stream = client.chat.completions.create(
            model=cfg.model_id,
            messages=[
                {"role": "user", "content": "First question."},
                {"role": "assistant", "content": ""},
                {"role": "user", "content": "Say: ok"},
            ],
            stream=True,
            max_tokens=16,
        )
        _collect(stream)
        return ProbeResult(
            "empty_assistant_in_history", OK,
            "gateway tolerates an empty assistant turn",
        )
    except Exception as exc:  # noqa: BLE001
        code = _status_code(exc)
        if code == 400:
            return ProbeResult(
                "empty_assistant_in_history", UNSUPPORTED,
                "400 — an empty assistant turn kills the whole conversation; the "
                "item 7 placeholder is load-bearing for this model",
            )
        raise


def probe_reasoning_channel(cfg: ModelConfig) -> ProbeResult:
    """Does this vendor stream reasoning_content (billed but invisible)?"""
    client = _raw_client(cfg)
    stream = client.chat.completions.create(
        model=cfg.model_id,
        messages=[{"role": "user", "content": "What is 17 * 23? Think it through."}],
        stream=True,
        max_tokens=256,
    )
    content, reasoning, _stop = _collect(stream)
    if reasoning:
        return ProbeResult(
            "reasoning_channel", DEGRADED,
            f"{len(reasoning)} chars of reasoning_content vs {len(content)} chars "
            "of content — completion_tokens will exceed what the visible reply "
            "explains; budget max_tokens accordingly",
        )
    return ProbeResult("reasoning_channel", OK, "no separate reasoning channel")


def probe_code_fence_shape(cfg: ModelConfig) -> ProbeResult:
    """Does the model fence code the way the extractor expects?"""
    client = _raw_client(cfg)
    stream = client.chat.completions.create(
        model=cfg.model_id,
        messages=[{
            "role": "user",
            "content": "Give me a Python code block that prints hello. Code only.",
        }],
        stream=True,
        max_tokens=128,
    )
    content, _reasoning, _stop = _collect(stream)
    if _extract_all_python(content):
        shape = "standard backtick fence" if "```" in content else "pseudo-fence (normalized)"
        return ProbeResult("code_fence", OK, f"extractable — {shape}")
    return ProbeResult(
        "code_fence", DEGRADED,
        "no code block could be extracted from a code-only request — every turn "
        f"would look like a stalled turn. Raw head: {content[:120]!r}",
    )


PROBES: list[Callable[[ModelConfig], ProbeResult]] = [
    probe_basic_stream,
    probe_temperature_zero,
    probe_max_tokens,
    probe_seed,
    probe_empty_assistant_history,
    probe_reasoning_channel,
    probe_code_fence_shape,
]


# ---------------------------------------------------------------------------
# Sweep + rendering
# ---------------------------------------------------------------------------
def probe_model(cfg: ModelConfig, probes=None) -> list[ProbeResult]:
    """Run every probe against one model. A probe that blows up is recorded.

    No probe may abort the sweep: a matrix listing six answers and one ``error``
    is useful, a traceback with zero answers is not.
    """
    results: list[ProbeResult] = []
    for probe in (probes if probes is not None else PROBES):
        name = getattr(probe, "__name__", "probe").replace("probe_", "")
        try:
            results.append(probe(cfg))
        except Exception as exc:  # noqa: BLE001
            code = _status_code(exc)
            detail = f"{type(exc).__name__}: {exc}"
            if code is not None:
                detail = f"HTTP {code} — {detail}"
            results.append(ProbeResult(name, ERROR, detail))
    return results


def render_markdown(matrix: dict) -> str:
    """One row per model, one column per probe, plus a detail list below."""
    models = matrix.get("models", {})
    probe_names: list[str] = []
    for results in models.values():
        for r in results:
            if r["probe"] not in probe_names:
                probe_names.append(r["probe"])

    lines = [
        "# 模型-网关兼容性矩阵",
        "",
        f"生成时间：{matrix.get('generated_at', '')}",
        "",
        "状态：`ok` 符合框架假设 / `degraded` 能跑但和框架假设不一致 / "
        "`unsupported` 网关明确拒绝 / `error` 探针本身没跑出结论。",
        "",
        "| model | " + " | ".join(probe_names) + " |",
        "|" + "---|" * (len(probe_names) + 1),
    ]
    for model, results in models.items():
        by_name = {r["probe"]: r["status"] for r in results}
        cells = [by_name.get(p, "—") for p in probe_names]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")

    lines += ["", "## 明细", ""]
    for model, results in models.items():
        lines.append(f"### {model}")
        for r in results:
            detail = f" — {r['detail']}" if r.get("detail") else ""
            lines.append(f"- `{r['probe']}` **{r['status']}**{detail}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe model × gateway protocol compatibility before a real round."
    )
    parser.add_argument("--models", required=True,
                        help="Comma-separated model names from models.yaml.")
    parser.add_argument("--models-yaml", default=str(ROOT / "configs" / "models.yaml"))
    parser.add_argument("--out", default=None,
                        help="Write the Markdown table here (JSON goes to <out>.json).")
    args = parser.parse_args()

    names = [m.strip() for m in args.models.split(",") if m.strip()]
    if not names:
        raise SystemExit("No models provided.")

    matrix: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "models": {},
    }
    for name in names:
        print(f"[compat] probing {name} ...")
        try:
            cfg = load_model_config(Path(args.models_yaml), name)
        except (KeyError, OSError) as exc:
            matrix["models"][name] = [
                ProbeResult("config", ERROR, f"{type(exc).__name__}: {exc}").as_dict()
            ]
            continue
        results = probe_model(cfg)
        matrix["models"][name] = [r.as_dict() for r in results]
        for r in results:
            print(f"  {r.name:28s} {r.status:12s} {r.detail}")

    text = render_markdown(matrix)
    if args.out:
        out = Path(args.out)
        out.write_text(text, encoding="utf-8")
        out.with_suffix(out.suffix + ".json").write_text(
            json.dumps(matrix, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[compat] matrix → {out}")
    else:
        print("")
        print(text)

    # Exit code is informational only: a degraded/unsupported cell is a fact to
    # record before the round, not a reason to stop the sweep.
    n_bad = sum(
        1 for results in matrix["models"].values()
        for r in results if r["status"] in (UNSUPPORTED, ERROR)
    )
    if n_bad:
        print(f"[compat] {n_bad} unsupported/error cell(s) — read 明细 before the round.",
              file=sys.stderr)


if __name__ == "__main__":
    main()


