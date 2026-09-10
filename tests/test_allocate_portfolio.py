import pandas as pd
import pytest

from scoring import DEFAULT_UNKNOWN_CHAIN_GAS_JPY, allocate_portfolio


def _ranked(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_caps_are_respected_with_concentrated_candidates():
    # 同じプロトコル・同じチェーンに偏った候補でも、どの上限も超えてはいけない。
    rows = [
        {"chain": "Ethereum", "project": "p1", "symbol": f"USDC{i}", "pool_id": f"pool-{i}", "score": 10.0 - i}
        for i in range(6)
    ] + [
        {"chain": "Arbitrum", "project": "p2", "symbol": "USDC", "pool_id": "pool-p2", "score": 3.0},
        {"chain": "Base", "project": "p3", "symbol": "USDC", "pool_id": "pool-p3", "score": 2.5},
    ]
    result = allocate_portfolio(_ranked(rows), top_n=10)

    assert (result["weight"] <= 0.20 + 1e-9).all()
    assert result.groupby("project")["weight"].sum().le(0.35 + 1e-9).all()
    assert result.groupby("chain")["weight"].sum().le(0.40 + 1e-9).all()
    assert result["weight"].sum() <= 1.0 + 1e-9


def test_infeasible_case_leaves_remainder_unallocated_instead_of_breaking_caps():
    # 4プールしかなく全部同一チェーン。20%/35%/40%上限では100%を配分しきれない
    # (チェーン上限40% ÷ プール上限20% = 2プール分までしか埋まらない)。
    rows = [
        {"chain": "Ethereum", "project": f"p{i}", "symbol": "USDC", "pool_id": f"pool-{i}", "score": 10.0 - i}
        for i in range(4)
    ]
    result = allocate_portfolio(_ranked(rows), top_n=10)

    assert (result["weight"] <= 0.20 + 1e-9).all()
    assert result["weight"].sum() == 0.40  # 2プール分(各20%)で頭打ち
    assert result["weight"].sum() < 1.0  # 上限を守るため意図的に100%を配分しない


def test_pool_id_is_preserved_for_disambiguation():
    rows = [
        {"chain": "Monad", "project": "accountable", "symbol": "USDC", "pool_id": "uuid-1", "score": 10.0},
        {"chain": "Monad", "project": "accountable", "symbol": "USDC", "pool_id": "uuid-2", "score": 9.0},
    ]
    result = allocate_portfolio(_ranked(rows), top_n=10)
    assert set(result["pool_id"]) == {"uuid-1", "uuid-2"}


def test_empty_ranked_returns_empty():
    result = allocate_portfolio(pd.DataFrame(columns=["chain", "project", "symbol", "score"]))
    assert result.empty


def test_no_capital_jpy_omits_gas_columns():
    rows = [{"chain": "Ethereum", "project": "p1", "symbol": "USDC", "pool_id": "pool-1", "score": 10.0}]
    result = allocate_portfolio(_ranked(rows), top_n=10)
    assert "est_gas_jpy" not in result.columns
    assert "high_gas_cost_warning" not in result.columns


def test_capital_jpy_adds_gas_columns_and_flags_expensive_chain():
    rows = [
        {"chain": "Ethereum", "project": "p1", "symbol": "USDC", "pool_id": "pool-1", "score": 10.0},
        {"chain": "Arbitrum", "project": "p2", "symbol": "USDC", "pool_id": "pool-2", "score": 9.0},
    ]
    # 1万円規模。各プール20%上限なので1プールあたり2000円程度のポジションになる。
    result = allocate_portfolio(_ranked(rows), top_n=10, capital_jpy=10_000.0)

    eth_row = result[result["chain"] == "Ethereum"].iloc[0]
    arb_row = result[result["chain"] == "Arbitrum"].iloc[0]

    assert eth_row["est_gas_jpy"] == 459
    assert arb_row["est_gas_jpy"] == 15
    assert eth_row["position_jpy"] == pytest.approx(eth_row["weight"] * 10_000.0)
    # イーサリアムはポジションの小ささに対してガス代の比率が高く、警告が立つはず。
    assert bool(eth_row["high_gas_cost_warning"]) is True
    # Arbitrumは比率が十分低く、警告は立たないはず。
    assert bool(arb_row["high_gas_cost_warning"]) is False


def test_unknown_chain_uses_conservative_default_gas():
    rows = [{"chain": "SomeNewChain", "project": "p1", "symbol": "USDC", "pool_id": "pool-1", "score": 10.0}]
    result = allocate_portfolio(_ranked(rows), top_n=10, capital_jpy=10_000.0)
    assert result.iloc[0]["est_gas_jpy"] == DEFAULT_UNKNOWN_CHAIN_GAS_JPY
