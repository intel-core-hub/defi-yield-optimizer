import numpy as np
import pandas as pd
import pytest

from backtest import MIN_FOLD_HISTORY_DAYS, _realized_metrics, _walk_forward_folds


def _history(apy: list[float], tvl: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=len(apy), freq="D")
    return pd.DataFrame({"apy": apy, "tvlUsd": tvl}, index=idx)


def test_realized_metrics_constant_series_has_zero_cv_and_no_drawdown():
    window = _history([10.0] * 10, [1_000_000.0] * 10)

    metrics = _realized_metrics(window)

    assert metrics["realized_apy_cv"] == 0.0
    assert metrics["realized_tvl_max_drawdown_pct"] == 0.0


def test_realized_metrics_detects_mid_window_tvl_drawdown():
    # peaks at 1_000_000, drops 50%, then recovers - trend alone would miss this
    tvl = [1_000_000.0, 1_000_000.0, 500_000.0, 1_000_000.0]
    window = _history([10.0] * 4, tvl)

    metrics = _realized_metrics(window)

    assert metrics["realized_tvl_max_drawdown_pct"] == pytest.approx(-0.5)


def test_realized_metrics_nan_when_fewer_than_two_valid_points():
    window = _history([10.0], [1_000_000.0])

    metrics = _realized_metrics(window)

    assert np.isnan(metrics["realized_apy_cv"])
    assert np.isnan(metrics["realized_tvl_max_drawdown_pct"])


def _pool_row() -> pd.Series:
    return pd.Series({"chain": "Ethereum", "project": "test-project", "symbol": "USDC"})


def test_walk_forward_folds_splits_history_at_step_intervals():
    # 90 days = MIN_FOLD_HISTORY_DAYS (60) + 2*STEP_DAYS (30), so fold starts
    # land exactly on 0, 15, 30 with no partial fold left over
    history = _history([10.0] * 90, [1_000_000.0] * 90)

    folds = _walk_forward_folds(_pool_row(), history)

    assert len(folds) == 3
    assert [f["fold_start_date"] for f in folds] == list(history.index[[0, 15, 30]])
    for fold in folds:
        assert fold["project"] == "test-project"
        assert fold["chain"] == "Ethereum"


def test_walk_forward_folds_empty_when_history_shorter_than_one_fold():
    history = _history([10.0] * (MIN_FOLD_HISTORY_DAYS - 10), [1_000_000.0] * (MIN_FOLD_HISTORY_DAYS - 10))

    folds = _walk_forward_folds(_pool_row(), history)

    assert folds == []
