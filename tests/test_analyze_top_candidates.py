import logging

import pandas as pd
import requests

from scoring import RANKED_COLUMNS, analyze_top_candidates


def _fake_pools() -> pd.DataFrame:
    # 1件だけ存在するが、テストではmin_tvl_usdでこれを弾いて候補0件を作る。
    return pd.DataFrame({
        "tvlUsd": [100.0],
        "chain": ["Ethereum"],
        "stablecoin": [True],
        "project": ["p1"],
        "symbol": ["USDC"],
        "pool": ["11111111-1111-1111-1111-111111111111"],
        "apy": [5.0],
    })


def test_zero_candidates_returns_empty_dataframe_with_expected_columns(monkeypatch):
    monkeypatch.setattr("scoring.fetch_pools", _fake_pools)

    result = analyze_top_candidates(min_tvl_usd=10**9)

    assert list(result.columns) == RANKED_COLUMNS
    assert len(result) == 0


def test_pool_history_fetch_failure_is_logged_not_silent(monkeypatch, caplog):
    monkeypatch.setattr("scoring.fetch_pools", _fake_pools)

    def _raise(pool_id):
        raise requests.RequestException("boom")

    monkeypatch.setattr("scoring.fetch_pool_history", _raise)

    with caplog.at_level(logging.WARNING, logger="scoring"):
        result = analyze_top_candidates(min_tvl_usd=0)

    assert len(result) == 0
    assert any(
        "11111111-1111-1111-1111-111111111111" in record.message
        for record in caplog.records
    )
