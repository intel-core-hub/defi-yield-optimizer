import pandas as pd

from scoring import allocate_portfolio


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
