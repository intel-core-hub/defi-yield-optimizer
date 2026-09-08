"""安定性スコアの予測力を、複数の時点に分けたウォークフォワード検証で確認する。

本来のバックテストには複数時点で保存したスナップショットの蓄積が要るが、まだ
フェーズ2の途中でそれが無い。代わりに、DeFiLlamaが返す各プールの既存の履歴データを
使い、一度きりの前半/後半2分割ではなく、時間をスライドさせながら複数回
「直近LOOKBACK_DAYS日で判定 → 次のHORIZON_DAYS日の実現値を見る」を繰り返す
ウォークフォワード検証を行う。

これにより、例えば200日分の履歴があるプールなら1つではなく十数個の(判定時点,
検証期間)ペアが得られ、そのプールの歴史の中の複数の時期(=ある程度異なる相場状況)
にわたって判定の予測力を確認できる。ただし取得できる期間はDeFiLlama側の各プールの
公開履歴に限られるため、複数の強気/弱気相場をまたいだ厳密な検証ではないことに注意。
"""

import sys

import numpy as np
import pandas as pd
import requests

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from scoring import fetch_pool_history, stability_metrics
from yields import fetch_pools, DATA_DIR

LOOKBACK_DAYS = 30  # 判定に使う直近日数
HORIZON_DAYS = 30  # 判定後に実現値を見る日数
STEP_DAYS = 15  # ウォークフォワードで判定時点をずらす刻み幅
MIN_FOLD_HISTORY_DAYS = LOOKBACK_DAYS + HORIZON_DAYS


def _realized_metrics(window: pd.DataFrame) -> dict:
    apy = window["apy"].dropna()
    tvl = window["tvlUsd"].dropna()
    apy_cv = float(apy.std() / apy.mean()) if len(apy) >= 2 and apy.mean() != 0 else np.nan
    if len(tvl) >= 2:
        running_max = tvl.cummax()
        tvl_max_drawdown_pct = float(((tvl - running_max) / running_max).min())
    else:
        tvl_max_drawdown_pct = np.nan
    return {"realized_apy_cv": apy_cv, "realized_tvl_max_drawdown_pct": tvl_max_drawdown_pct}


def _walk_forward_folds(pool_row: pd.Series, history: pd.DataFrame) -> list[dict]:
    folds = []
    last_start = len(history) - MIN_FOLD_HISTORY_DAYS
    for start in range(0, max(last_start, 0) + 1, STEP_DAYS):
        train = history.iloc[start : start + LOOKBACK_DAYS]
        test = history.iloc[start + LOOKBACK_DAYS : start + LOOKBACK_DAYS + HORIZON_DAYS]
        if len(train) < LOOKBACK_DAYS or len(test) < HORIZON_DAYS:
            continue

        predicted = stability_metrics(train)
        realized = _realized_metrics(test)
        folds.append({
            "chain": pool_row["chain"],
            "project": pool_row["project"],
            "symbol": pool_row["symbol"],
            "fold_start_date": train.index[0],
            "predicted_flag": predicted["flag"],
            "predicted_apy_cv": predicted["apy_cv"],
            "predicted_tvl_max_drawdown_pct": predicted["tvl_max_drawdown_pct"],
            **realized,
        })
    return folds


def walk_forward_backtest(
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
        rows.extend(_walk_forward_folds(pool, history))

    return pd.DataFrame(rows)


if __name__ == "__main__":
    result = walk_forward_backtest()
    DATA_DIR.mkdir(exist_ok=True)
    result.to_csv(DATA_DIR / "backtest_walk_forward.csv", index=False)

    result["predicted_risky"] = result["predicted_flag"] != "ok"
    result["fold_quarter"] = pd.to_datetime(result["fold_start_date"]).dt.tz_localize(None).dt.to_period("Q")

    overall = result.groupby("predicted_risky")[
        ["realized_apy_cv", "realized_tvl_max_drawdown_pct"]
    ].mean()

    print(f"検証フォールド数: {len(result)}件 (プール数: {result['project'].nunique()}, "
          f"判定時点の四半期数: {result['fold_quarter'].nunique()})")
    print()
    print("全期間通算: 前半判定(predicted_risky)別に見た、後半HORIZON_DAYS日間の実現値の平均")
    print(overall.to_string())
    print()

    by_quarter = result.groupby(["fold_quarter", "predicted_risky"])[
        ["realized_apy_cv", "realized_tvl_max_drawdown_pct"]
    ].mean()
    print("四半期別の内訳(判定時点がいつだったかで層別。サンプル数が少ない四半期は参考値):")
    print(by_quarter.to_string())
    print()
    print(
        "predicted_risky=Trueの方がrealized_apy_cvが高い/"
        "realized_tvl_max_drawdown_pctがより大きくマイナスである状態が、"
        "特定の四半期だけでなく通算・各四半期で概ね一貫していれば、"
        "判定の予測力が特定の相場局面に依存した偶然ではないと言える。"
    )
