import numpy as np
import pandas as pd

from scoring import MAX_TRUSTED_TVL_DRAWDOWN, stability_metrics


def _history(apy: list[float], tvl: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=len(apy), freq="D")
    return pd.DataFrame({"apy": apy, "tvlUsd": tvl}, index=idx)


def test_insufficient_history_flagged():
    history = _history([10.0] * 5, [1_000_000.0] * 5)
    metrics = stability_metrics(history)
    assert metrics["flag"] == "insufficient_history"
    assert np.isnan(metrics["apy_median"])
    assert metrics["history_days"] == 5


def test_stable_history_is_ok():
    history = _history([10.0] * 30, [1_000_000.0] * 30)
    metrics = stability_metrics(history)
    assert metrics["flag"] == "ok"
    assert metrics["apy_median"] == 10.0
    assert metrics["apy_cv"] == 0.0
    assert metrics["tvl_max_drawdown_pct"] == 0.0


def test_high_volatility_excluded():
    # 変動係数(std/mean)は対称な交互パターンだと1.0を超えにくいので、少数の
    # 極端なスパイクで平均から大きく偏らせた歪んだ分布にする(変動係数が閾値を超える)。
    apy = [1.0] * 25 + [1000.0] * 5
    history = _history(apy, [1_000_000.0] * 30)
    metrics = stability_metrics(history)
    assert metrics["apy_cv"] > 1.0
    assert metrics["flag"] == "high_volatility_excluded"


def test_mid_period_tvl_drawdown_detected_even_if_recovered():
    # 開始・終了は同じTVLだが、途中でピークから50%超急落する(取り付け騒ぎ→回復)。
    # tvl_trend_pctだけでは0(変化なし)に見えるはずだが、drawdownは検知されるべき。
    tvl = [1_000_000.0] * 10 + [400_000.0] * 10 + [1_000_000.0] * 10
    history = _history([10.0] * 30, tvl)
    metrics = stability_metrics(history)
    assert metrics["tvl_trend_pct"] == 0.0
    assert metrics["tvl_max_drawdown_pct"] < MAX_TRUSTED_TVL_DRAWDOWN
    assert metrics["flag"] == "tvl_drawdown_risk"
