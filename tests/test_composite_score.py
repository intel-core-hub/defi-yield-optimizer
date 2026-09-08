import numpy as np
import pandas as pd
import pytest

from scoring import (
    MAX_TRUSTED_APY_CV,
    MIN_TRUSTED_LISTING_AGE_DAYS,
    OLD_HACK_PENALTY,
    composite_score,
)


def _base_row(**overrides) -> pd.Series:
    row = {
        "apy_median": 10.0,
        "apy_cv": 0.0,
        "tvl_trend_pct": 0.0,
        "tvl_max_drawdown_pct": 0.0,
        "audits_count": 2,
        "days_since_defillama_listing": MIN_TRUSTED_LISTING_AGE_DAYS + 10,
        "category": "Yield",
        "was_hacked": False,
        "hack_recent": False,
    }
    row.update(overrides)
    return pd.Series(row)


def test_no_penalties_returns_apy_median():
    assert composite_score(_base_row()) == pytest.approx(10.0)


def test_high_volatility_excluded_is_nan():
    row = _base_row(apy_cv=MAX_TRUSTED_APY_CV + 0.01)
    assert np.isnan(composite_score(row))


def test_recent_hack_is_nan_even_with_good_stability():
    row = _base_row(was_hacked=True, hack_recent=True)
    assert np.isnan(composite_score(row))


def test_excluded_category_is_nan():
    row = _base_row(category="Uncollateralized Lending")
    assert np.isnan(composite_score(row))


def test_old_hack_penalizes_but_does_not_exclude():
    row = _base_row(was_hacked=True, hack_recent=False)
    score = composite_score(row)
    assert score == pytest.approx(10.0 * OLD_HACK_PENALTY)


def test_no_audits_and_unknown_age_penalized():
    row = _base_row(audits_count=0, days_since_defillama_listing=np.nan)
    score = composite_score(row)
    # audit_penalty=0.5, age_penalty=0.5 (不明) の掛け算
    assert score == pytest.approx(10.0 * 0.5 * 0.5)


def test_tvl_drawdown_and_old_hack_penalties_both_apply():
    # 1つのflagしか反映されない実装だと、この2つが両方効いていることを見落とす回帰を検知する。
    row = _base_row(tvl_max_drawdown_pct=-0.6, was_hacked=True, hack_recent=False)
    score = composite_score(row)
    expected = 10.0 * (1 + (-0.6)) * OLD_HACK_PENALTY
    assert score == pytest.approx(expected)


def test_missing_metrics_is_nan():
    row = _base_row(apy_cv=np.nan)
    assert np.isnan(composite_score(row))
