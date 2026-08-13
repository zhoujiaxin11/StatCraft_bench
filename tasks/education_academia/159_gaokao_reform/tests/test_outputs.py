"""
tests/test_outputs.py — Artifact-level checks for 159_gaokao_reform

Design principle
----------------
Main scoring is done by framework/scorer.py against outcome.json (factoid
protocol). This pytest module handles ONLY things that don't fit factoids:
- File existence
- File format sanity (CSV columns, row count)
- No-leakage keyword scan on the trajectory / feature list
- Basic structural checks on artifacts

Weights for these tests live in groundtruth JSON under `pytest_weight_map`.
Blended into the final score by framework/verifier_wrapper.py at
factoid_alpha = 0.9 (i.e. these tests together weigh ~10%).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


WORKSPACE = Path(".")
OUTCOME = WORKSPACE / "outcome.json"
PREDICTIONS = WORKSPACE / "output" / "predictions.csv"

# Keywords banned in Part 3 features (target leakage)
FORBIDDEN_LEAKAGE = ["人数", "累计", "rank", "位次", "high_score", "high_count"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def outcome() -> dict:
    if not OUTCOME.exists():
        pytest.skip("outcome.json not produced")
    with open(OUTCOME, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def predictions_df():
    if not PREDICTIONS.exists():
        pytest.skip("predictions.csv not produced")
    import pandas as pd

    return pd.read_csv(PREDICTIONS)


# ---------------------------------------------------------------------------
# File existence
# ---------------------------------------------------------------------------
def test_outcome_json_exists():
    assert OUTCOME.exists(), "outcome.json missing at workspace root"


def test_predictions_csv_exists():
    assert PREDICTIONS.exists(), "output/predictions.csv missing"


# ---------------------------------------------------------------------------
# outcome.json structural sanity
# ---------------------------------------------------------------------------
def test_outcome_has_all_parts(outcome):
    for k in ("part1_overview", "part2_reform_impact", "part3_model", "part4_causal", "part5_advanced"):
        assert k in outcome, f"missing top-level key: {k}"


def test_outcome_is_json_serializable(outcome):
    """Values must be plain JSON types — no NaN, no numpy scalars.

    allow_nan=False makes the rejection real: Python's json.dumps otherwise
    silently emits bare NaN/Infinity, so the docstring's promise was not
    enforced before.
    """
    for k, v in outcome.items():
        json.dumps(v, allow_nan=False)  # will raise on non-serializable values


# ---------------------------------------------------------------------------
# predictions.csv sanity
# ---------------------------------------------------------------------------
def test_predictions_has_required_columns(predictions_df):
    required = {"省份", "科类", "年份", "y_true", "y_pred"}
    assert required.issubset(predictions_df.columns), f"missing columns: {required - set(predictions_df.columns)}"


def test_predictions_has_conformal_bounds(predictions_df):
    assert "y_pred_lo" in predictions_df.columns and "y_pred_hi" in predictions_df.columns, \
        "predictions.csv must include y_pred_lo and y_pred_hi (Part 5 conformal)"


def test_predictions_include_2024_2025(predictions_df):
    years = set(predictions_df["年份"].unique())
    assert {2024, 2025}.issubset(years), f"test set must include 2024 and 2025; got {sorted(years)}"


def test_predictions_no_extreme_negative(predictions_df):
    # Predictions can theoretically be negative for regression, but domain-wise
    # high-score counts should be non-negative. Allow slight negatives from
    # RF extrapolation but flag if >5% of test rows are strongly negative.
    n_bad = (predictions_df["y_pred"] < -100).sum()
    n = len(predictions_df)
    assert n_bad / max(n, 1) < 0.05, f"too many strongly-negative predictions: {n_bad}/{n}"


# ---------------------------------------------------------------------------
# Leakage detection
# ---------------------------------------------------------------------------
def test_no_leakage_in_feature_names(outcome):
    """Part 3 feature names must not contain forbidden target-leakage terms."""
    features = outcome.get("part3_model", {}).get("feature_names", [])
    if not features:
        pytest.skip("feature_names not reported")
    joined = "|".join(str(f).lower() for f in features)
    for kw in FORBIDDEN_LEAKAGE:
        assert kw.lower() not in joined, f"leakage keyword {kw!r} in feature names: {features}"


def test_feature_count_at_least_5(outcome):
    fc = outcome.get("part3_model", {}).get("feature_count", 0)
    assert fc >= 5, f"feature_count must be ≥5; got {fc}"


# ---------------------------------------------------------------------------
# Part 4 structural
# ---------------------------------------------------------------------------
def test_part4_has_two_methods(outcome):
    """Task requires ≥2 independent causal methods — at minimum DID + SCM."""
    p4 = outcome.get("part4_causal", {})
    assert "did_effect" in p4 and "scm_effect" in p4, \
        "part4_causal must report both did_effect and scm_effect"


def test_part4_placebo_reported(outcome):
    p4 = outcome.get("part4_causal", {})
    assert "placebo_time_effect" in p4 or "placebo_treat_effect" in p4, \
        "at least one placebo effect must be reported"


# ---------------------------------------------------------------------------
# Part 5 structural
# ---------------------------------------------------------------------------
def test_part5_panel_ols_has_min_coefs(outcome):
    coefs = outcome.get("part5_advanced", {}).get("panel_ols_coefs", {})
    assert isinstance(coefs, dict), "panel_ols_coefs must be a dict"
    assert len(coefs) >= 3, "panel_ols_coefs must have at least 3 coefficients"
    # Each coef must have coef / se / t / p
    for name, obj in coefs.items():
        for f in ("coef", "se", "t", "p"):
            assert f in obj, f"coef {name!r} missing field {f}"
