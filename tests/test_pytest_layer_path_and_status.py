"""复现 the pytest-collection-path regression：相对 test_dir 零收集 + broken 仍上报 ok。

覆盖两处：
  - run_pytest: test_dir 已是绝对路径时不得再拼 Path.cwd()，
    否则 cwd 不在仓库根时路径畸形 → pytest 零收集。
  - score_run: pytest_status 落在 pytest_broken / pytest_broken_no_collection
    时不得打 STATUS_OK —— 那是框架在说自己的测试层坏了，
    上报成 ok+0 分会被聚合层当成模型能力算进均值。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


# --- 第一处：相对/绝对 test_dir 的路径拼装 ---------------------------------

def test_run_pytest_absolute_test_dir_not_joined_to_cwd(tmp_path, monkeypatch):
    """test_dir 是绝对路径时，收集结果不能受调用方 cwd 影响。"""
    from framework.verifier_wrapper import run_pytest

    task_tests = tmp_path / "task" / "tests"
    task_tests.mkdir(parents=True)
    (task_tests / "test_probe.py").write_text(
        "def test_probe():\n    assert True\n", encoding="utf-8"
    )
    workspace = tmp_path / "ws"
    workspace.mkdir()

    # 故意把 cwd 换到一个与 task 无关的目录：老实现 Path.cwd() / <绝对路径>
    # 在 pathlib 下会丢弃左侧，所以绝对路径这一支老实现其实是过的；
    # 真正的回归点是下面那个相对路径用例。此处作为不退化的锚点。
    monkeypatch.chdir(tmp_path / "ws")
    res_abs = run_pytest(task_tests.resolve(), workspace)
    assert res_abs.per_test, "绝对路径应能正常收集到测试"
    assert not res_abs.no_executable_tests


def test_run_pytest_relative_test_dir_resolved_against_task_not_cwd(tmp_path, monkeypatch):
    """相对 test_dir 在 cwd 不是仓库根时，老实现会零收集。"""
    from framework.verifier_wrapper import run_pytest

    root = tmp_path / "repo"
    task_tests = root / "tasks" / "demo" / "tests"
    task_tests.mkdir(parents=True)
    (task_tests / "test_probe.py").write_text(
        "def test_probe():\n    assert True\n", encoding="utf-8"
    )
    workspace = tmp_path / "ws"
    workspace.mkdir()

    # 调用方在仓库根下发起，但传的是相对路径；cwd 此刻不是 root。
    monkeypatch.chdir(root)
    rel = Path("tasks/demo/tests")
    res = run_pytest(rel, workspace)
    assert res.per_test, (
        "相对 test_dir 零收集：Path.cwd() / test_dir 拼出的路径在 "
        "cwd 不是仓库根时不存在，pytest 一个测试都收不到"
    )
    assert not res.no_executable_tests


# --- 第二处：broken 的测试层不得上报 ok ------------------------------------

def _make_task(tmp_path: Path, *, weight_map: dict, broken: bool) -> tuple[Path, Path]:
    """造一个 pytest-only 任务：broken=True 时 conftest 直接炸掉收集。"""
    task_dir = tmp_path / "task"
    tests = task_dir / "tests"
    tests.mkdir(parents=True)
    if broken:
        (tests / "conftest.py").write_text(
            "raise RuntimeError('collection blown up on purpose')\n", encoding="utf-8"
        )
    (tests / "test_demo.py").write_text(
        "def test_alpha():\n    assert True\n", encoding="utf-8"
    )

    gt_dir = tmp_path / "gt" / "public"
    gt_dir.mkdir(parents=True)
    (gt_dir / "demo.json").write_text(json.dumps({
        "task_id": "demo",
        "factoid_alpha": 0,
        "pytest_weight_map": weight_map,
        "values": {},
    }, ensure_ascii=False), encoding="utf-8")

    (task_dir / "task.toml").write_text(
        '[task]\nid = "demo"\ntitle = "demo"\n\n[scoring]\nfactoid_alpha = 0.0\n',
        encoding="utf-8",
    )
    return task_dir, tmp_path / "gt"


def test_broken_pytest_layer_must_not_report_ok(tmp_path):
    """conftest 炸掉 → pytest_status=pytest_broken* → status 不许是 ok。"""
    from framework.runner import score_run
    from framework.statuses import INFRA_STATUSES

    task_dir, gt_dir = _make_task(
        tmp_path, weight_map={"test_alpha": 100.0}, broken=True
    )
    workspace = tmp_path / "ws"
    workspace.mkdir()

    result = score_run(
        task_id="demo",
        workspace=workspace,
        task_dir=task_dir,
        groundtruth_dir=gt_dir,
    )

    pytest_status = result.get("pytest_status")
    assert pytest_status in {"pytest_broken", "pytest_broken_no_collection"}, (
        f"用例没造出 broken 场景，实际 pytest_status={pytest_status!r}"
    )
    assert result["status"] != "ok", (
        "框架已经把 pytest_status 写成 " f"{pytest_status!r}（自己知道测试层坏了），"
        "却仍然上报 status=ok + 0 分，聚合层会把这个假 0 分算进模型能力均值"
    )
    assert result["status"] in INFRA_STATUSES, (
        f"broken 的测试层应归 infra 类以便聚合剔除，实际 status={result['status']!r}"
    )
    # 除了通用的 error 位，还要落一个专用的
    # scorer_error_reason。前者 aggregate.py 还拿来给没有 status 的老 score.json
    # 反推终态，语义是「出错了」；后者只有「判分层自己坏了」这一种来源，机器可读。
    assert isinstance(result.get("scorer_error_reason"), str) and result["scorer_error_reason"], (
        "缺 scorer_error_reason：排查时只能去 grep 一句自然语言，"
        "还会跟 scorer 抛异常那条 error 混在一起"
    )
    assert pytest_status in result["scorer_error_reason"], (
        "scorer_error_reason 里没带上 pytest_status 原值：pytest_broken（跑崩）和 "
        "pytest_broken_no_collection（压根没收集到）排查方向不一样，不能糊成一句"
    )
    assert isinstance(result.get("error"), str) and result["error"], (
        "通用 error 位被去掉了：aggregate.py:264 靠它给老 score.json 反推终态"
    )


def test_healthy_pytest_layer_still_reports_ok(tmp_path):
    """反向锚点：测试层正常时不能被误判成 infra 失败。"""
    from framework.runner import score_run

    task_dir, gt_dir = _make_task(
        tmp_path, weight_map={"test_alpha": 100.0}, broken=False
    )
    workspace = tmp_path / "ws"
    workspace.mkdir()

    result = score_run(
        task_id="demo",
        workspace=workspace,
        task_dir=task_dir,
        groundtruth_dir=gt_dir,
    )
    pytest_status = result.get("pytest_status")
    assert pytest_status == "scored", f"正常场景 pytest_status={pytest_status!r}"
    assert result["status"] == "ok"
    assert result["combined_score"] == pytest.approx(1.0)
    # 反向锚点：测试层正常时不许留 scorer_error_reason，否则这个字段就没法当
    # 「判分层坏了」的判据用了。
    assert "scorer_error_reason" not in result, (
        f"正常场景也写了 scorer_error_reason={result.get('scorer_error_reason')!r}"
    )
