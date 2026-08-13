"""P1 回归：revision checklist items 6/7/8/9/10/11/12。

第 5 条按组内约定不动（网关侧问题）。

第 6 条 SDK 层与 app 层双重重试相乘 → 收敛成单层、退避感知剩余预算。
第 7 条 空 assistant 消息回灌触发 400 → 历史里换占位符。
第 8 条 reasoning_content 只记账、不进 content、不进历史。
第 9 条 同一条消息内多代码块拼成一个脚本执行（共享命名空间）。
第 10 条 非标准 fence（厂商伪代码块标签）也要能抽出代码。
第 11 条 交卷时按 schema 自检，缺项告警但仍允许交卷。
第 12 条 seed 逐次下发给模型侧 + system_fingerprint 落轨迹（v4.1 并入那版）。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from framework.agent_adapter import (
    ChatResponse,
    HttpApiAgent,
    ModelConfig,
    _extract_all_python,
)


# --- 公共假件 -----------------------------------------------------------------

class _RecordingExecutor:
    """记下每次拿到的代码，永远成功。"""

    timeout = 5

    def __init__(self):
        self.calls: list[str] = []

    def setup(self):
        pass

    def teardown(self):
        pass

    def run(self, code, log_name=None):
        self.calls.append(code)
        return True, "(ok)"


def _make_agent(tmp_path, reply, *, executor=None, reasoning="",
                stop_reason="stop", required_outputs=None):
    class _FakeClient:
        def __init__(self):
            self.seen_messages: list[list[dict]] = []

        def chat(self, messages, *, deadline=None):
            self.seen_messages.append([dict(m) for m in messages])
            # 同上：reasoning 是本部分（第 8 条）给 ChatResponse 新加的字段，
            # 只在用例真的要测思维链时才传，别让其他用例在基线上因 TypeError 挂。
            extra = {"reasoning": reasoning} if reasoning else {}
            return ChatResponse(content=reply, stop_reason=stop_reason, **extra)

    cfg = ModelConfig(name="fake", api_base="http://x", api_key="k")
    # required_outputs 是第二部分（P0 第 2 条）加的形参，本组用例不依赖它；
    # 只在显式传入时才带上，这样这些用例放到未改的基线上也能跑（用来证明
    # 它们打在 P1 的缺陷上，而不是被上一部分的签名变化连带弄挂）。
    kwargs = {} if required_outputs is None else {"required_outputs": required_outputs}
    agent = HttpApiAgent(
        cfg, max_turns=5, executor=executor or _RecordingExecutor(), **kwargs,
    )
    agent.model = _FakeClient()
    agent._workspace = tmp_path
    return agent


# --- 第 10 条：厂商伪 fence ---------------------------------------------------

def test_vendor_pseudo_fence_is_extracted():
    """DeepSeek 那种 <｜｜DSML｜｜python> 结束符也要能抽出代码。"""
    reply = (
        "我来算一下。\n"
        "<｜｜DSML｜｜python>\n"
        "import pandas as pd\n"
        "print(pd.__version__)\n"
        "</｜｜DSML｜｜python>\n"
        "算完了。"
    )
    blocks = _extract_all_python(reply)
    assert blocks, (
        "厂商伪 fence 一个代码块都没抽到：这一轮会被当成空转，"
        "连续几轮之后 run 被 no_progress 杀掉，而模型其实一直在给代码"
    )
    assert "import pandas as pd" in blocks[0]


def test_pseudo_fence_only_closing_tag_still_works():
    """只有结束符是伪 fence、开头是正常 ```python 的混合形态。"""
    reply = "```python\nprint(1)\n</｜｜DSML｜｜python>\n"
    blocks = _extract_all_python(reply)
    assert blocks and "print(1)" in blocks[0]


def test_standard_fence_is_not_rewritten():
    """反向锚点：标准 fence 走原路径，不许被归一化改写。"""
    assert _extract_all_python("```python\nx = 1\n```") == ["x = 1"]


def test_prose_and_html_are_not_mistaken_for_fences():
    """反向锚点：提到 python 的散文和普通 HTML 标签不算代码块。"""
    assert _extract_all_python('I like python. <a href="x">py</a> docs.') == []


def test_other_labeled_fences_still_blocked():
    """反向锚点：```json / ```bash 不许钻进 Python 执行器。"""
    assert _extract_all_python('```json\n{"a": 1}\n```') == []


# --- 第 9 条：多代码块拼接 ---------------------------------------------------

def test_multiple_blocks_are_concatenated_into_one_script(tmp_path):
    """同一条消息里的两个代码块要合成一个脚本跑，否则第二块 NameError。"""
    ex = _RecordingExecutor()
    reply = (
        "先读数据：\n```python\nimport pandas as pd\ndf = pd.DataFrame()\n```\n"
        "再看一眼：\n```python\nprint(len(df))\n```\n"
    )
    agent = _make_agent(tmp_path, reply, executor=ex)
    assert agent.step("") == "continue"
    assert len(ex.calls) == 1, (
        f"两个代码块被分成 {len(ex.calls)} 次执行：它们不共享命名空间，"
        "第二块引用第一块的变量必然 NameError"
    )
    assert "df = pd.DataFrame()" in ex.calls[0]
    assert "print(len(df))" in ex.calls[0]
    assert ex.calls[0].index("df = pd.DataFrame()") < ex.calls[0].index("print(len(df))"), (
        "拼接顺序颠倒了"
    )
    assert "concatenated" in agent._pending_observation


def test_single_block_is_untouched(tmp_path):
    """反向锚点：只有一个代码块时不加任何包装。"""
    ex = _RecordingExecutor()
    agent = _make_agent(tmp_path, "```python\nprint(1)\n```", executor=ex)
    assert agent.step("") == "continue"
    assert ex.calls == ["print(1)"]
    assert "concatenated" not in agent._pending_observation


def test_block_cap_still_applies_before_merge(tmp_path):
    """max_blocks_per_turn 仍生效：先截断再拼接，不许被合并绕过。"""
    ex = _RecordingExecutor()
    reply = "".join(f"```python\nprint({i})\n```\n" for i in range(3))
    cfg = ModelConfig(name="fake", api_base="http://x", api_key="k")
    agent = HttpApiAgent(cfg, max_turns=5, executor=ex, max_blocks_per_turn=2)

    class _FakeClient:
        def chat(self, messages, *, deadline=None):
            return ChatResponse(content=reply, stop_reason="stop")

    agent.model = _FakeClient()
    agent._workspace = tmp_path
    assert agent.step("") == "continue"
    assert len(ex.calls) == 1
    assert "print(2)" not in ex.calls[0], "被 cap 掉的第三块又从拼接路径漏进来了"


# --- 第 7 条：空 assistant 消息 ---------------------------------------------

def test_empty_reply_is_not_replayed_into_history(tmp_path):
    """空 content 不许原样回灌——回灌会让下一次请求整体 400。"""
    agent = _make_agent(tmp_path, "", stop_reason="tool_calls")
    agent.step("")
    assistant_msgs = [m for m in agent.messages if m["role"] == "assistant"]
    assert assistant_msgs, "assistant 轮次丢了"
    assert assistant_msgs[-1]["content"].strip(), (
        "历史里存了一条空 assistant 消息：下一轮请求会被网关整体拒掉（400）"
    )
    assert "empty" in assistant_msgs[-1]["content"].lower()
    # 轨迹里仍要看得见原始的空回复，别把证据擦掉
    last = [e for e in agent.trajectory if e.get("role") == "assistant"][-1]
    assert last["content"] == ""
    assert last.get("empty_reply") is True
    assert last.get("stop_reason") == "tool_calls"


def test_normal_reply_goes_into_history_verbatim(tmp_path):
    """反向锚点：正常回复原样进历史，不许被占位符替换。"""
    agent = _make_agent(tmp_path, "```python\nprint(1)\n```")
    agent.step("")
    assistant_msgs = [m for m in agent.messages if m["role"] == "assistant"]
    assert assistant_msgs[-1]["content"] == "```python\nprint(1)\n```"


# --- 第 8 条：reasoning_content 只记账 --------------------------------------

def test_reasoning_content_is_accounted_not_replayed(tmp_path):
    """思维链只记账：不进 content、不进历史，但轨迹里查得到。"""
    agent = _make_agent(
        tmp_path, "```python\nprint(1)\n```", reasoning="我先想想……很长的推理",
    )
    agent.step("")
    assistant_msgs = [m for m in agent.messages if m["role"] == "assistant"]
    assert "我先想想" not in assistant_msgs[-1]["content"], (
        "reasoning_content 被并进了对话历史：等于把模型自己的思维链回灌给它"
    )
    last = [e for e in agent.trajectory if e.get("role") == "assistant"][-1]
    assert last.get("reasoning_content") == "我先想想……很长的推理"
    assert last.get("reasoning_chars") == len("我先想想……很长的推理")


def test_no_reasoning_field_when_absent(tmp_path):
    """反向锚点：没有思维链就别在轨迹里凭空加字段。"""
    agent = _make_agent(tmp_path, "```python\nprint(1)\n```")
    agent.step("")
    last = [e for e in agent.trajectory if e.get("role") == "assistant"][-1]
    assert "reasoning_content" not in last


# --- 第 6 条：单层重试 + 退避感知预算 ---------------------------------------

def test_sdk_retry_layer_is_disabled():
    """SDK 层重试必须钉成 0，否则和 app 层相乘。"""
    src = Path("framework/agent_adapter.py").resolve().read_text(encoding="utf-8")
    idx = src.find("self._client = OpenAI(")
    assert idx > 0
    block = src[idx:idx + 900]
    assert "max_retries=0" in block, (
        "SDK 层仍在用 cfg.max_retries：同一个配置值喂两层重试，"
        "max_retries=4 会变成 5×5=25 次请求，且 SDK 内部的 sleep 逃出 deadline"
    )


def test_app_layer_retries_exactly_max_retries_times(monkeypatch):
    """app 层是唯一一层：max_retries=2 → 一共 3 次请求。"""
    from openai import APIConnectionError

    from framework import agent_adapter as A
    from framework.agent_adapter import ModelClient

    cfg = ModelConfig(name="fake", api_base="http://x", api_key="k", max_retries=2)
    mc = ModelClient.__new__(ModelClient)   # 绕开 __init__ 里真建 OpenAI 客户端
    mc.cfg = cfg
    calls = {"n": 0}
    slept: list[float] = []
    monkeypatch.setattr(A.time, "sleep", slept.append)

    def _boom(messages):
        calls["n"] += 1
        raise APIConnectionError(request=None)  # type: ignore[arg-type]

    mc._do_chat = _boom  # type: ignore[method-assign]
    with pytest.raises(APIConnectionError):
        mc.chat([{"role": "user", "content": "x"}], deadline=time.time() + 3600)
    assert calls["n"] == 3, f"重试次数不对：{calls['n']}，应为 1 次首发 + 2 次重试"
    assert len(slept) == 2, f"退避次数不对：{slept}"


def test_retry_gives_up_when_budget_cannot_hold_next_attempt():
    """退避感知剩余 wall 预算：装不下下一次尝试就别睡，直接抛原始异常。"""
    from openai import APIConnectionError

    from framework.agent_adapter import ModelClient

    cfg = ModelConfig(name="fake", api_base="http://x", api_key="k", max_retries=5)
    mc = ModelClient.__new__(ModelClient)
    mc.cfg = cfg
    calls = {"n": 0}

    def _boom(messages):
        calls["n"] += 1
        raise APIConnectionError(request=None)  # type: ignore[arg-type]

    mc._do_chat = _boom  # type: ignore[method-assign]
    t0 = time.time()
    with pytest.raises(APIConnectionError):
        mc.chat([{"role": "user", "content": "x"}], deadline=time.time() + 0.5)
    elapsed = time.time() - t0
    assert calls["n"] == 1, (
        f"预算只剩 0.5s 还发了 {calls['n']} 次请求：重试把 api_error 拖成超时就是这么来的"
    )
    assert elapsed < 2.0, f"睡过了头：{elapsed:.2f}s"


# --- 第 11 条：交卷结构自检（告警不拦） -------------------------------------

_SCHEMA = {
    "type": "object",
    "properties": {
        "part1": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        "part2": {
            "type": "object",
            "properties": {"c": {"type": "number"}},
            "required": ["c"],
        },
    },
    "required": ["part1", "part2"],
}


def _agent_with_schema(tmp_path, outcome: dict):
    (tmp_path / "outcome.json").write_text(
        json.dumps(outcome, ensure_ascii=False), encoding="utf-8"
    )
    agent = _make_agent(tmp_path, "[FINAL_ANSWER]\n")
    agent._schema = _SCHEMA
    return agent


def test_schema_selfcheck_warns_but_still_submits(tmp_path):
    """半成品交卷要留下告警，但不许因此拦住交卷。"""
    agent = _agent_with_schema(tmp_path, {"part1": {"a": 1}})
    assert agent.step("") == "done", (
        "结构自检把交卷拦住了：第 11 条明确要求只告警、不做强制循环"
    )
    events = [e.get("event") for e in agent.trajectory]
    assert "schema_selfcheck_warning" in events, f"告警没进轨迹：{events}"
    warn = [e for e in agent.trajectory if e.get("event") == "schema_selfcheck_warning"][0]
    gaps = warn["gaps"]
    assert any("part1.b" in g for g in gaps), gaps
    assert any("part2" in g for g in gaps), gaps


def test_schema_selfcheck_silent_when_complete(tmp_path):
    """反向锚点：结构齐全时不许刷无意义告警。"""
    agent = _agent_with_schema(
        tmp_path, {"part1": {"a": 1, "b": 2}, "part2": {"c": 0.5}}
    )
    assert agent.step("") == "done"
    events = [e.get("event") for e in agent.trajectory]
    assert "schema_selfcheck_warning" not in events


def test_schema_selfcheck_flags_null_values(tmp_path):
    """必填项填 None 也算缺项——半成品最常见的形状。"""
    agent = _agent_with_schema(
        tmp_path, {"part1": {"a": 1, "b": None}, "part2": {"c": 0.5}}
    )
    assert agent.step("") == "done"
    warn = [e for e in agent.trajectory if e.get("event") == "schema_selfcheck_warning"][0]
    assert any("part1.b" in g and "null" in g for g in warn["gaps"]), warn["gaps"]


def test_schema_selfcheck_disabled_without_schema(tmp_path):
    """schema 缺失/解析失败时自检整体关掉，不许影响交卷。"""
    (tmp_path / "outcome.json").write_text('{"anything": 1}', encoding="utf-8")
    agent = _make_agent(tmp_path, "[FINAL_ANSWER]\n")
    agent._schema = None
    assert agent.step("") == "done"
    assert agent._schema_selfcheck() == []


# --- 第 12 条：seed 逐次下发 + 采样指纹 --------------------------------------

class _FakeStream:
    """最小可迭代流：一个 content chunk + 一个带 usage/指纹的收尾 chunk。"""

    def __init__(self, fingerprint=None):
        self.fingerprint = fingerprint

    def __iter__(self):
        import types
        yield types.SimpleNamespace(
            choices=[types.SimpleNamespace(
                delta=types.SimpleNamespace(content="hi", reasoning_content=None),
                finish_reason=None)],
            usage=None, model="m", system_fingerprint=self.fingerprint,
        )
        yield types.SimpleNamespace(
            choices=[types.SimpleNamespace(
                delta=types.SimpleNamespace(content=None, reasoning_content=None),
                finish_reason="stop")],
            usage=types.SimpleNamespace(
                prompt_tokens=1, completion_tokens=2, total_tokens=3),
            model="m", system_fingerprint=self.fingerprint,
        )


def _client_recording_kwargs(*, seed, fingerprint=None, reject_once=False):
    """造一个只记请求体的 ModelClient，返回 (client, calls)。"""
    import types

    from framework.agent_adapter import ModelClient

    cfg = ModelConfig(name="fake", api_base="http://x", api_key="k", seed=seed)
    client = ModelClient.__new__(ModelClient)   # 绕开 __init__ 里的真 SDK 构造
    client.cfg = cfg
    calls: list[dict] = []
    state = {"rejected": False}

    def create(**kwargs):
        calls.append(dict(kwargs))
        if reject_once and not state["rejected"]:
            state["rejected"] = True
            raise TypeError("got an unexpected keyword argument")
        return _FakeStream(fingerprint)

    client._client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
    )
    return client, calls


def test_seed_is_sent_to_the_model_when_configured():
    """配了 seed 就必须真发到请求体里。

    之前 seed 只用来给 trial 目录起名 + 播种本地 random/numpy，模型侧压根不知道，
    于是「同 seed 复跑应当同分」这个前提不成立，replication_check 那层失去依据。
    """
    client, calls = _client_recording_kwargs(seed=7)
    client._do_chat([{"role": "user", "content": "hi"}])
    assert calls[0].get("seed") == 7, f"seed 没进请求体：keys={sorted(calls[0])}"


def test_seed_absent_means_no_seed_key_at_all():
    """反向锚点：没配 seed 时一个 seed 字段都不许加（有网关见到生键直接 400）。"""
    client, calls = _client_recording_kwargs(seed=None)
    client._do_chat([{"role": "user", "content": "hi"}])
    assert "seed" not in calls[0], f"多发了 seed：{calls[0].get('seed')!r}"


def test_typeerror_fallback_drops_both_stream_options_and_seed():
    """老 shim 抛 TypeError 时，stream_options 和 seed 必须一起摘掉。

    只摘 stream_options 的话第二次调用还是同一个 TypeError，等于没退化。
    """
    client, calls = _client_recording_kwargs(seed=7, reject_once=True)
    client._do_chat([{"role": "user", "content": "hi"}])
    assert len(calls) == 2, f"没走到退化重试，calls={len(calls)}"
    assert "seed" in calls[0] and "stream_options" in calls[0]
    assert "seed" not in calls[1] and "stream_options" not in calls[1], (
        f"退化重试没摘干净：keys={sorted(calls[1])}"
    )


def test_system_fingerprint_is_captured_from_stream():
    """采样指纹要从流里捞出来带回 ChatResponse。指纹变了说明后端换了。"""
    client, _ = _client_recording_kwargs(seed=7, fingerprint="fp_abc")
    resp = client._do_chat([{"role": "user", "content": "hi"}])
    assert resp.system_fingerprint == "fp_abc"


def _agent_with_seeded_client(tmp_path, *, seed, fingerprint, with_cfg=True):
    agent = _make_agent(tmp_path, "先想一下，这轮不写代码。")

    class _SeededClient:
        def __init__(self):
            if with_cfg:
                self.cfg = ModelConfig(
                    name="f", api_base="b", api_key="k", seed=seed)

        def chat(self, messages, *, deadline=None):
            return ChatResponse(
                content="先想一下，这轮不写代码。", stop_reason="stop",
                system_fingerprint=fingerprint,
            )

    agent.model = _SeededClient()
    return agent


@pytest.mark.parametrize("seed", [7, 0])
def test_seed_sent_and_fingerprint_land_in_trajectory(tmp_path, seed):
    """轨迹里要能事后反查「发了哪个 seed / 对方回的什么指纹」。

    seed=0 单独跑一遍：0 是 falsy，用 `if seed:` 写会被吞掉。
    """
    agent = _agent_with_seeded_client(tmp_path, seed=seed, fingerprint="fp_abc")
    agent.step("")
    asst = [e for e in agent.trajectory if e.get("role") == "assistant"]
    assert asst, "没有 assistant 轨迹条目"
    assert asst[0].get("seed_sent") == seed, f"seed_sent={asst[0].get('seed_sent')!r}"
    assert asst[0].get("system_fingerprint") == "fp_abc"


def test_no_seed_no_fingerprint_keys_when_unset(tmp_path):
    """反向锚点：都没有时不许写空键进轨迹。"""
    agent = _agent_with_seeded_client(tmp_path, seed=None, fingerprint=None)
    agent.step("")
    asst = [e for e in agent.trajectory if e.get("role") == "assistant"][0]
    assert "seed_sent" not in asst and "system_fingerprint" not in asst


def test_trajectory_entry_survives_client_without_cfg(tmp_path):
    """兼容锚点：假 client 没有 .cfg 时不许抛 AttributeError。

    本文件和 test_p0 里的替身都是这形状，取 seed 必须用 getattr 兜。
    """
    agent = _agent_with_seeded_client(
        tmp_path, seed=None, fingerprint=None, with_cfg=False)
    agent.step("")   # 不抛异常即算过
    asst = [e for e in agent.trajectory if e.get("role") == "assistant"][0]
    assert "seed_sent" not in asst


def test_runner_threads_seed_into_run_agent():
    """runner 侧：签名带 seed、调用点真传、并有 BENCH_SEED 兜底。"""
    src = Path("framework/runner.py").resolve().read_text(encoding="utf-8")
    assert "seed: int | None = None," in src, "run_agent 签名里没有 seed 形参"
    assert "seed=args.seed," in src, "调用点没把 CLI 的 seed 传下去"
    assert 'os.environ.get("BENCH_SEED")' in src, (
        "没有 BENCH_SEED 兜底：调用方忘了显式传时同进程里的 seed 会丢"
    )
