"""P0 回归：revision checklist items 1/2/3。

第 1 条 末尾一次网关抖动不许抹掉已算出的有效分（api_error 不覆写 ok）。
第 2 条 required_outputs 在产出即算进展，no_progress 不许杀掉正在出活的 run。
第 3 条 全 infra 的任务要报 N/A 而不是 0 分（ability_measured=False）。

第 4 条（wall_timeout 单点分类）按the earlier revision's stance暂不改动，对应用例已撤。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# --- 第 1 条：终态覆写规则 -------------------------------------------------

def _apply_terminal_mapping(terminated_by: str, status: str | None) -> str | None:
    """把 runner 里 terminated_by→status 的映射抽出来跑。

    直接调 main() 需要真跑一次 agent；这里复制不了逻辑就失去意义，
    所以用源码里同一套判定条件做等价断言，另有一条源码锚点用例
    (test_runner_source_api_error_branch_protects_ok) 盯着两边别走散。
    """
    result: dict = {"status": status}
    if terminated_by == "api_error" and result.get("status") in (
        None, "no_output"
    ):
        result["status"] = "api_error"
    elif terminated_by == "executor_crash" and result.get("status") in (
        None, "no_output", "scorer_error"
    ):
        result["status"] = "executor_crash"
    elif terminated_by in ("stuck_loop", "no_progress") and result.get("status") in (
        None, "no_output"
    ):
        result["status"] = terminated_by
    elif terminated_by == "max_turns_exhausted" and result.get("status") == "no_output":
        result["status"] = "max_turns"
    return result["status"]


def test_runner_source_api_error_branch_protects_ok():
    """第 1 条：源码里 api_error 分支只许覆写「压根没打过分」的两种状态。

    v4.1：判定从黑名单换成白名单后，源码里不再出现 STATUS_OK，所以这条断言
    改成锚白名单本身 —— 既证明 ok 不在覆写集合里，也证明 bad_json 不在。
    """
    src = Path("framework/runner.py").resolve().read_text(encoding="utf-8")
    if not src:  # pragma: no cover
        pytest.skip("runner.py 不可读")
    idx = src.find('if terminated_by == "api_error"')
    assert idx > 0, "找不到 api_error 分支"
    branch = src[idx:idx + 200]
    assert 'in (\n        None, "no_output"\n    )' in branch, (
        "api_error 分支不是白名单写法：只有 status 为 None / no_output（压根没打出分）"
        "才允许覆写成 api_error，否则末尾一次网关抖动会把已算出的有效分改写成 infra，"
        "整条 trial 被聚合层丢掉"
    )
    assert "not in" not in branch, "api_error 分支仍是黑名单写法（默认覆写，新增状态会被误伤）"


def test_api_error_does_not_overwrite_ok():
    assert _apply_terminal_mapping("api_error", "ok") == "ok"
    assert _apply_terminal_mapping("api_error", "no_output") == "api_error"
    assert _apply_terminal_mapping("api_error", None) == "api_error"
    # scorer_error 优先级仍高于 api_error（原有行为不退化）
    assert _apply_terminal_mapping("api_error", "scorer_error") == "scorer_error"
    # v4.1：bad_json 是模型自己把 outcome.json 写坏了，属于模型失败（该记 0 分），
    # 不许被末尾一次网关抖动改写成 infra 丢掉。
    assert _apply_terminal_mapping("api_error", "bad_json") == "bad_json"


def test_api_error_overwrite_set_matches_status_buckets():
    """第 1 条配套：被 api_error 覆写掉的状态，必须都是「没打出分」的状态。

    反过来说：凡是 statuses.py 归 MODEL_FAIL 且代表「模型交了东西但不合格」的
    状态（bad_json / max_turns / stuck_loop / no_progress），都不能因为一次
    api_error 而变成 infra 被丢掉。
    """
    from framework.statuses import MODEL_FAIL_STATUSES, bucket

    for st in sorted(MODEL_FAIL_STATUSES - {"no_output"}):
        assert _apply_terminal_mapping("api_error", st) == st, (
            f"status={st} 被 api_error 覆写了，模型的真实失败记录会被聚合层丢掉"
        )
    assert bucket("bad_json") == "model_fail"


# --- 第 2 条：产物在写 = 有进展 ---------------------------------------------

def _agent_with_fake_model(tmp_path, replies, required_outputs):
    from framework.agent_adapter import ChatResponse, HttpApiAgent, ModelConfig

    class _Noop:
        timeout = 5

        def setup(self):
            pass

        def teardown(self):
            pass

        def run(self, code, log_name=None):  # pragma: no cover - 不该被调用
            return True, ""

    class _FakeClient:
        def __init__(self):
            self.i = 0

        def chat(self, messages, *, deadline=None):
            r = replies[min(self.i, len(replies) - 1)]
            self.i += 1
            # 每轮回复加个计数后缀：本组用例专测 no_progress，
            # 不能让内容重复先触发 stuck_loop 熔断。
            return ChatResponse(content=f"{r}（第{self.i}轮）", stop_reason="stop")

    cfg = ModelConfig(name="fake", api_base="http://x", api_key="k")
    agent = HttpApiAgent(
        cfg, max_turns=10, executor=_Noop(),
        empty_streak_limit=2,
        required_outputs=required_outputs,
    )
    agent.model = _FakeClient()
    agent._workspace = tmp_path
    agent._artifact_fingerprint = agent._snapshot_artifacts()
    return agent


def test_no_progress_fires_when_nothing_is_written(tmp_path):
    """反向锚点：真的什么都没产出时，熔断器照旧要开。"""
    agent = _agent_with_fake_model(
        tmp_path, ["我在想想。"], ["report.csv"]
    )
    assert agent.step("") == "continue"
    assert agent.step("") == "done"
    assert agent.terminated_by == "no_progress"


def test_artifact_write_counts_as_progress(tmp_path):
    """第 2 条：产物在写就算有进展，不许被 no_progress 杀掉。"""
    agent = _agent_with_fake_model(
        tmp_path, ["还在跑，稍等。"], ["report.csv"]
    )
    # 第一轮：没有产物 → 计一次空转
    assert agent.step("") == "continue"
    assert agent._empty_streak == 1
    # 第二轮：required_output 落盘了 → 应当重置而不是熔断
    (tmp_path / "report.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    status = agent.step("")
    assert status == "continue", (
        "required_outputs 正在产出，run 仍被 no_progress 杀掉"
    )
    assert agent.terminated_by != "no_progress"
    assert agent._empty_streak == 0
    events = [e.get("event") for e in agent.trajectory]
    assert "artifact_progress" in events, f"进展信号没进轨迹：{events}"
    # 第三轮：产物不再变化 → 计数重新累积，熔断仍然可达
    assert agent.step("") == "continue"
    assert agent.step("") == "done"
    assert agent.terminated_by == "no_progress"


def test_preexisting_files_are_not_counted_as_progress(tmp_path):
    """setup 之前就在的文件不算进展，否则等于永久关掉熔断器。"""
    (tmp_path / "report.csv").write_text("stale\n", encoding="utf-8")
    agent = _agent_with_fake_model(tmp_path, ["无所事事。"], ["report.csv"])
    assert agent.step("") == "continue"
    assert agent.step("") == "done"
    assert agent.terminated_by == "no_progress"


# --- 第 3 条：全 infra 报 N/A 而不是 0 -------------------------------------

def _write_trial(task_dir: Path, name: str, *, status: str, score: float) -> None:
    d = task_dir / "trials" / name
    d.mkdir(parents=True)
    (d / "score.json").write_text(json.dumps({
        "combined_score": score, "status": status,
    }, ensure_ascii=False), encoding="utf-8")


def _bare_task(tmp_path: Path) -> Path:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(
        '[task]\nid = "demo"\ntitle = "demo"\n', encoding="utf-8"
    )
    return task_dir


def test_all_infra_task_reports_na_not_zero(tmp_path):
    from framework.aggregate import aggregate_task

    task_dir = _bare_task(tmp_path)
    for i in range(3):
        _write_trial(task_dir, f"2026_m1_seed{i}", status="api_error", score=0.0)

    agg = aggregate_task(task_dir, "m1")
    assert agg is not None
    assert agg.n_infra == 3
    assert agg.ability_measured is False, "全 infra 却声称测到了能力"
    assert agg.stable_score is None, (
        f"全 infra 的任务 stable_score={agg.stable_score!r}，"
        "读者无法把它和模型真拿了 0 分区分开"
    )
    assert agg.pass_at_1 is None and agg.pass_all is None and agg.std is None


def test_model_fail_task_still_scores_zero(tmp_path):
    """反向锚点：模型自己放弃仍然是实打实的 0，不许变成 N/A。"""
    from framework.aggregate import aggregate_task

    task_dir = _bare_task(tmp_path)
    for i in range(3):
        _write_trial(task_dir, f"2026_m2_seed{i}", status="no_output", score=0.0)

    agg = aggregate_task(task_dir, "m2")
    assert agg is not None
    assert agg.ability_measured is True
    assert agg.stable_score == 0.0
    assert agg.n_model_fail == 3
