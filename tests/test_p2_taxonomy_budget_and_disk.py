"""P2 回归：revision checklist items 13/14/15/16/17。

第 13 条 首测前的模型-网关兼容性矩阵（烟囱用例）→ framework/compat_matrix.py。
第 14 条 scenario 白名单不校验、改造导致评分对象漂移 → framework/taxonomy.py，
         告警而非硬失败。
第 15 条 磁盘紧张下并发跑题 → framework/batch.py 的 --serial / --min-free-gb。
第 16 条 max_turns 偏紧、成功 run 被浪费的 turn 拖垮 → turn_stats 记账，
         先把「浪费了几轮」变成可读的数，不盲调 task.toml。
第 17 条 scenario 白名单尚未更新 → 词表收到 taxonomy.py 一处，改一行即可。

注意：为了让这一组用例能在**未改的发布基线上逐条跑**（而不是整文件 import
就崩、26 条被算成 1 个 collection error），taxonomy / compat_matrix 这两个新模块
的 import 放在各用例内部。基线上这些用例会各自因 ModuleNotFoundError 失败——
「模块根本不存在」本身就是第 13/14/17 条缺陷的证明。
"""
from __future__ import annotations

import json

import pytest


def _taxonomy():
    from framework import taxonomy

    return taxonomy


def _compat():
    from framework import compat_matrix

    return compat_matrix


# --- 公共假件 -----------------------------------------------------------------

def _write_task(root, scenario_dir, task_id, *, scenario, difficulty):
    """在 tmp 里搭出 tasks/<scenario_dir>/<task_id>/task.toml 的最小骨架。"""
    d = root / "tasks" / scenario_dir / task_id
    d.mkdir(parents=True, exist_ok=True)
    lines = [f'[task]\nid = "{task_id}"\n', "[taxonomy]\n"]
    if scenario is not None:
        lines.append(f'scenario = "{scenario}"\n')
    if difficulty is not None:
        lines.append(f'difficulty = "{difficulty}"\n')
    (d / "task.toml").write_text("".join(lines), encoding="utf-8")
    return d


def _write_registry(root, task_id, *, scenario, difficulty, path="registry/v1.json"):
    p = root / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "tasks": [{
            "task_id": task_id,
            "scenario": scenario,
            "difficulty": difficulty,
        }],
    }), encoding="utf-8")
    return p


# --- 第 14/17 条：taxonomy 白名单 ---------------------------------------------

def test_clean_task_produces_no_warning(tmp_path):
    """反向锚点：声明、目录、registry 三处一致时不许刷告警。"""
    d = _write_task(tmp_path, "education_academia", "159_x",
                    scenario="education_academia", difficulty="extreme")
    _write_registry(tmp_path, "159_x", scenario="education_academia", difficulty="extreme")
    assert _taxonomy().validate_task_taxonomy(d) == []


def test_scenario_typo_is_caught(tmp_path):
    """白名单外的 scenario 必须报出来——否则 per-scenario 聚合被拆成两个桶。"""
    d = _write_task(tmp_path, "educationn_academia", "160_x",
                    scenario="educationn_academia", difficulty="hard")
    problems = _taxonomy().validate_task_taxonomy(d)
    assert any("educationn_academia" in p and "whitelist" in p for p in problems), problems


def test_difficulty_typo_is_caught(tmp_path):
    """difficulty 词表同样是闭集合：Extreme（大写）不算 extreme。"""
    d = _write_task(tmp_path, "education_academia", "161_x",
                    scenario="education_academia", difficulty="Extreme")
    problems = _taxonomy().validate_task_taxonomy(d)
    assert any("difficulty" in p and "whitelist" in p for p in problems), problems


def test_scenario_disagrees_with_directory(tmp_path):
    """两边都在白名单里、但目录和声明对不上：改造时挪了目录没改元数据。"""
    d = _write_task(tmp_path, "society_policy", "162_x",
                    scenario="education_academia", difficulty="hard")
    problems = _taxonomy().validate_task_taxonomy(d)
    assert any("tasks/society_policy/" in p for p in problems), problems


def test_registry_scenario_drift_is_caught(tmp_path):
    """第 14 条的评分对象漂移：选进轮次用一个标签、被评分时是另一个。"""
    d = _write_task(tmp_path, "education_academia", "163_x",
                    scenario="education_academia", difficulty="hard")
    _write_registry(tmp_path, "163_x", scenario="society_policy", difficulty="hard")
    problems = _taxonomy().validate_task_taxonomy(d)
    assert any("registry" in p or "v1.json" in p for p in problems), problems
    assert any("'society_policy'" in p or '"society_policy"' in p for p in problems), problems


def test_registry_difficulty_drift_is_caught(tmp_path):
    """难度必须在改造前后保持不变（第 14 条后半句）。"""
    d = _write_task(tmp_path, "education_academia", "164_x",
                    scenario="education_academia", difficulty="hard")
    _write_registry(tmp_path, "164_x", scenario="education_academia", difficulty="extreme")
    problems = _taxonomy().validate_task_taxonomy(d)
    assert any("difficulty" in p for p in problems), problems


def test_missing_taxonomy_fields_are_reported(tmp_path):
    """整段 [taxonomy] 只写了一半也要报，不能静默通过。"""
    d = _write_task(tmp_path, "education_academia", "165_x",
                    scenario=None, difficulty=None)
    problems = _taxonomy().validate_task_taxonomy(d)
    assert len(problems) >= 2, problems


def test_validate_never_raises_on_garbage(tmp_path):
    """告警而非硬失败：task.toml 读不出来也只能返回一条字符串，不许抛。"""
    d = tmp_path / "tasks" / "education_academia" / "166_x"
    d.mkdir(parents=True)
    (d / "task.toml").write_text("this is not toml [[[", encoding="utf-8")
    problems = _taxonomy().validate_task_taxonomy(d)
    assert problems and isinstance(problems[0], str)


def test_broken_registry_json_is_skipped(tmp_path):
    """registry 本身坏了不是这个检查的事：不许因此报 scenario 漂移。"""
    d = _write_task(tmp_path, "education_academia", "167_x",
                    scenario="education_academia", difficulty="hard")
    p = tmp_path / "registry" / "v1.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not json", encoding="utf-8")
    assert _taxonomy().validate_task_taxonomy(d) == []


def test_real_repo_task_159_is_clean():
    """仓库里现存的 159 必须是干净的——否则这个检查一上线就在刷噪音。"""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    task_dir = root / "tasks" / "education_academia" / "159_gaokao_reform"
    if not task_dir.is_dir():
        pytest.skip("159 not present in this checkout")
    assert _taxonomy().validate_task_taxonomy(task_dir) == []


def test_vocabulary_matches_task_authoring_doc():
    """第 17 条：词表只有一处。这里钉住它的内容，改词表必须过这道。"""
    assert _taxonomy().SCENARIOS == {
        "medical_health", "finance_economics", "retail_ecommerce",
        "industrial_energy", "transport_logistics", "environment_climate",
        "geo_hazards", "education_academia", "society_policy",
        "tech_internet", "sports_entertainment", "bio_chem_materials",
    }
    assert _taxonomy().DIFFICULTIES == {"easy", "hard", "extreme"}


def test_runner_taxonomy_gate_is_warning_only():
    """源码层：runner 里的 taxonomy 检查不许是 SystemExit / raise。"""
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent
           / "framework" / "runner.py").read_text(encoding="utf-8")
    assert "validate_task_taxonomy" in src, "runner 根本没接上这个检查"
    idx = src.index("_tax_problems = validate_task_taxonomy")
    window = src[idx: idx + 600]
    assert "stderr" in window, "告警必须走 stderr"
    assert "SystemExit" not in window and "raise" not in window, (
        "第 14 条要求告警而非硬失败：一个 bookkeeping 笔误不该浪费模型时间"
    )


# --- 第 16 条：turn_stats 记账 ------------------------------------------------

def _agent(tmp_path, reply, executor=None):
    from framework.agent_adapter import ChatResponse, HttpApiAgent, ModelConfig

    class _FakeClient:
        def chat(self, messages, *, deadline=None):
            return ChatResponse(content=reply, stop_reason="stop")

    class _Exec:
        timeout = 5

        def __init__(self, ok=True):
            self.ok = ok
            self.calls: list[str] = []

        def setup(self):
            pass

        def teardown(self):
            pass

        def run(self, code, log_name=None):
            self.calls.append(code)
            return self.ok, "(out)" if self.ok else "Traceback: boom"

    agent = HttpApiAgent(ModelConfig(name="fake", api_base="http://x", api_key="k"),
                         max_turns=5, executor=executor or _Exec())
    agent.model = _FakeClient()
    agent._workspace = tmp_path
    return agent


def test_turn_stats_counts_a_code_turn(tmp_path):
    agent = _agent(tmp_path, "```python\nprint(1)\n```")
    agent.step("")
    assert agent.turn_stats["turns_with_code"] == 1
    assert agent.turn_stats["turns_no_code"] == 0
    assert agent.turn_stats["turns_exec_failed"] == 0


def test_turn_stats_counts_a_wasted_turn(tmp_path):
    """第 16 条的核心：没有可执行产出的那一轮必须能被数出来。"""
    agent = _agent(tmp_path, "我先想一想这道题的思路，稍后给代码。")
    agent.step("")
    assert agent.turn_stats["turns_no_code"] == 1
    assert agent.turn_stats["turns_with_code"] == 0


def test_turn_stats_counts_failed_execution(tmp_path):
    class _Bad:
        timeout = 5

        def setup(self):
            pass

        def teardown(self):
            pass

        def run(self, code, log_name=None):
            return False, "Traceback: boom"

    agent = _agent(tmp_path, "```python\nboom\n```", executor=_Bad())
    agent.step("")
    assert agent.turn_stats["turns_with_code"] == 1
    assert agent.turn_stats["turns_exec_failed"] == 1


def test_turn_stats_counts_empty_reply(tmp_path):
    agent = _agent(tmp_path, "")
    agent.step("")
    assert agent.turn_stats["turns_empty_reply"] == 1


def test_turn_stats_counts_final_answer(tmp_path):
    """[FINAL_ANSWER] 必须独占一行（框架原有约定），交卷被拒也照样记一次。"""
    agent = _agent(tmp_path, "写完了。\n[FINAL_ANSWER]\n")
    agent.step("")
    assert agent.turn_stats["turns_final_answer"] == 1


def test_turn_stats_is_pure_bookkeeping(tmp_path):
    """反向锚点：记账不许影响控制流——熔断计数只看进展，不看 turn_stats。"""
    agent = _agent(tmp_path, "还在想。")
    before = agent._empty_streak
    agent.step("")
    assert agent._empty_streak == before + 1


def test_runner_carries_turn_stats_into_meta():
    """源码层：turn_stats 要真的流到 score.json 的 _meta 里才叫可复核。"""
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent
           / "framework" / "runner.py").read_text(encoding="utf-8")
    assert "last_turn_stats" in src
    assert '"turn_stats"' in src
    assert '"max_turns"' in src


# --- 第 15 条：磁盘门 / 串行约束 ----------------------------------------------

def test_serial_and_disk_flags_exist():
    from framework import batch

    src = (batch.__file__)
    text = open(src, encoding="utf-8").read()
    assert "--serial" in text and "--min-free-gb" in text


def test_disk_floor_skips_trial_without_spawning(tmp_path):
    """磁盘不够时不许启动子进程：跳过要 rc=-2 且日志说明原因。"""
    from framework import batch

    cmd = ["python3", "-c", "raise SystemExit(0)"]
    tag, rc, log_path = batch._run_one(
        cmd, tmp_path / "logs", "t1", 60, 10 ** 9,  # 1e9 GiB 的地板必然触发
    )
    assert rc == -2, "磁盘门没生效，子进程被照常启动了"
    body = open(log_path, encoding="utf-8").read()
    assert "SKIPPED before launch" in body
    assert "no model tokens" in body


def test_disk_floor_zero_is_baseline_behaviour(tmp_path):
    """反向锚点：默认 min_free_gb=0 时行为与基线一致，照常起子进程。"""
    from framework import batch

    cmd = ["python3", "-c", "print('hello from child')"]
    tag, rc, log_path = batch._run_one(cmd, tmp_path / "logs", "t2", 60, 0.0)
    assert rc == 0
    assert "hello from child" in open(log_path, encoding="utf-8").read()


def test_free_gb_never_raises(tmp_path):
    from framework import batch

    assert batch._free_gb(tmp_path) > 0
    # 不存在的路径不许把 batch 打挂，只能退化成"测不出来就别拦"
    assert batch._free_gb(tmp_path / "no" / "such" / "dir") == float("inf")


# --- 第 13 条：兼容性矩阵 ----------------------------------------------------

def test_probe_crash_becomes_error_cell_not_a_traceback():
    """一个探针炸了不许让整张矩阵跑不出来。"""
    cm = _compat()

    def probe_boom(cfg):
        raise RuntimeError("gateway ate the socket")

    def probe_fine(cfg):
        return cm.ProbeResult("fine", cm.OK, "")

    results = cm.probe_model(object(), probes=[probe_boom, probe_fine])
    assert [r.status for r in results] == [cm.ERROR, cm.OK]
    assert "gateway ate the socket" in results[0].detail


def test_render_markdown_has_a_row_per_model_and_details():
    cm = _compat()
    matrix = {
        "generated_at": "2026-08-12T10:00:00",
        "models": {
            "m1": [cm.ProbeResult("basic_stream", cm.OK, "stop_reason=stop").as_dict(),
                   cm.ProbeResult("seed_param", cm.UNSUPPORTED, "400 on seed=7").as_dict()],
            "m2": [cm.ProbeResult("basic_stream", cm.DEGRADED, "empty").as_dict()],
        },
    }
    text = cm.render_markdown(matrix)
    assert "| m1 |" in text and "| m2 |" in text
    assert "basic_stream" in text and "seed_param" in text
    assert "400 on seed=7" in text
    # m2 没跑 seed 探针，那一格要留空位而不是错位
    m2_row = [ln for ln in text.splitlines() if ln.startswith("| m2 |")][0]
    assert m2_row.count("|") == 4


def test_probe_statuses_are_a_closed_vocabulary():
    """和 statuses.py 一个道理：词表开口了矩阵就不可比。"""
    cm = _compat()
    assert {cm.OK, cm.UNSUPPORTED, cm.DEGRADED, cm.ERROR} == {
        "ok", "unsupported", "degraded", "error"
    }
