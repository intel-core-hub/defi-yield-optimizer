"""安定性スコアの予測力を簡易的に検証するバックテスト。

本来のバックテストには複数時点で保存したスナップショットの蓄積が要るが、まだ
フェーズ2の途中でそれが無い。代わりに、各プールの取得済み履歴データ(前半)だけで
scoring.pyと同じ安定性指標を計算し、それが後半(未来側)の実際の値動きを予測できて
いるかを検証する「疑似バックテスト」を行う。

具体的には:
1. 候補プールごとに全履歴を取得し、前半/後半に2分割する。
2. 前半だけを使ってstability_metrics()と同じ指標(apy_cv, tvl_max_drawdown_pct)を計算し、
   前半時点で"ok"だったか"risk"だったかを判定する(=もし前半時点でスコアリングしていたら
   どう判定されていたか)。
3. 後半の実際のapy_cv・tvl_max_drawdown_pctを計算し、前半の判定が悪かったプールほど
   後半の実現値も悪いか(=判定に予測力があるか)を比較する。
"""

import sys

import numpy as np
import pandas as pd
import requests

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from scoring import fetch_pool_history, stability_metrics
from yields import fetch_pools, DATA_DIR

MIN_HALF_HISTORY_DAYS = 14


def _realized_metrics(half: pd.DataFrame) -> dict:
    apy = half["apy"].dropna()
    tvl = half["tvlUsd"].dropna()
    apy_cv = float(apy.std() / apy.mean()) if len(apy) >= 2 and apy.mean() != 0 else np.nan
    if len(tvl) >= 2:
        running_max = tvl.cummax()
        tvl_max_drawdown_pct = float(((tvl - running_max) / running_max).min())
    else:
        tvl_max_drawdown_pct = np.nan
    return {"realized_apy_cv": apy_cv, "realized_tvl_max_drawdown_pct": tvl_max_drawdown_pct}


def backtest_stability_flag(
    stablecoin_only: bool = True,
    min_tvl_usd: float = 5_000_000,
    candidate_pool: int = 40,
) -> pd.DataFrame:
    pools = fetch_pools()
    filtered = pools[pools["tvlUsd"] >= min_tvl_usd]
    if stablecoin_only:
        filtered = filtered[filtered["stablecoin"]]
    candidates = filtered.sort_values("apy", ascending=False).head(candidate_pool)

    rows = []
    for _, pool in candidates.iterrows():
        try:
            history = fetch_pool_history(pool["pool"])
        except requests.RequestException:
            continue

        mid = len(history) // 2
        first_half, second_half = history.iloc[:mid], history.iloc[mid:]
        if len(first_half) < MIN_HALF_HISTORY_DAYS or len(second_half) < MIN_HALF_HISTORY_DAYS:
            continue

        predicted = stability_metrics(first_half)
        realized = _realized_metrics(second_half)

        rows.append({
            "chain": pool["chain"],
            "project": pool["project"],
            "symbol": pool["symbol"],
            "predicted_flag": predicted["flag"],
            "predicted_apy_cv": predicted["apy_cv"],
            "predicted_tvl_max_drawdown_pct": predicted["tvl_max_drawdown_pct"],
            **realized,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    result = backtest_stability_flag()
    DATA_DIR.mkdir(exist_ok=True)
    result.to_csv(DATA_DIR / "backtest_stability_flag.csv", index=False)

    result["predicted_risky"] = result["predicted_flag"] != "ok"
    summary = result.groupby("predicted_risky")[
        ["realized_apy_cv", "realized_tvl_max_drawdown_pct"]
    ].mean()

    print(f"検証対象プール数: {len(result)}件")
    print()
    print("前半データの判定(predicted_risky)別に見た、後半期間の実現値の平均:")
    print(summary.to_string())
    print()
    print(
        "predicted_risky=Trueの方がrealized_apy_cvが高い/"
        "realized_tvl_max_drawdown_pctがより大きくマイナスであれば、"
        "前半だけの判定が後半のリスクをある程度予測できていたことになる。"
    )
