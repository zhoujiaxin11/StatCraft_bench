"""v3.3 守门单测：状态词表一致性 & 新 infra 状态登记防退化。

覆盖点：
  - 新引入的 infra 状态
    (image_missing/docker_unavailable/bad_io/wall_timeout) 必须留在 INFRA_STATUSES，
    否则 aggregate._bucket 会把基础设施失败当成模型真实能力计入均值。
  - bad_json 应归 model_fail（agent 写坏 JSON 是模型过错），与 OSError→bad_io(infra) 区分。
  - 未登记的 status 不能再静默走 ok 兜底，必须 warn 并按 unknown(≈infra) 处理。
"""
from __future__ import annotations

import warnings

import pytest


def _statuses():
    from framework.statuses import (
        ALL_KNOWN_STATUSES,
        CRASH_STATUSES,
        INFRA_STATUSES,
        MODEL_FAIL_STATUSES,
        bucket,
    )
    return dict(
        ALL=ALL_KNOWN_STATUSES, CRASH=CRASH_STATUSES, INFRA=INFRA_STATUSES,
        FAIL=MODEL_FAIL_STATUSES, bucket=bucket,
    )


def test_new_infra_statuses_registered():
    S = _statuses()
    expected_infra = {
        "api_error", "executor_crash", "scorer_error",
        "docker_unavailable", "image_missing", "bad_io", "wall_timeout",
    }
    missing = [s for s in expected_infra if s not in S["INFRA"]]
    assert not missing, f"infra 状态被从 statuses.py 漏掉: {missing}"


def test_bad_json_is_model_fail_not_infra():
    # 李昊 路由一致性：JSONDecodeError→bad_json 计责；OSError→bad_io 不计责
    b = _statuses()["bucket"]
    assert b("bad_json") == "model_fail"
    assert b("bad_io") == "infra"


def test_buckets_disjoint():
    S = _statuses()
    overlap = S["INFRA"] & S["FAIL"]
    assert not overlap, f"同一状态同时属于 infra 与 model_fail，路由会矛盾: {overlap}"


def test_ok_and_none_bucket_as_ok():
    b = _statuses()["bucket"]
    assert b(None) == "ok"
    assert b("ok") == "ok"


def test_unknown_status_warns_and_does_not_silently_pass_as_ok():
    # 核心诉求:未知 status 必须显式告警并保守丢弃，绝不再静默走 ok
    b = _statuses()["bucket"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = b("some_brand_new_status_xyz")
        warned = any(
            issubclass(w.category, UserWarning) and "status" in str(w.message).lower()
            for w in caught
        )
        assert not (result == "ok" or not warned), (
            f"unknown status 静默走 ok 了！bucket={result!r} warned={warned}"
        )
