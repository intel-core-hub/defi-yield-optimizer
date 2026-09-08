"""プールごとのAPY履歴を取得し、安定性を加味したスコアで再ランキングする。

高APYでもボラティリティが高い/TVLが急減しているプールは危険信号として減点する。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from yields import fetch_pools, DATA_DIR

CHART_ENDPOINT = "https://yields.llama.fi/chart/{pool_id}"


def fetch_pool_history(pool_id: str) -> pd.DataFrame:
    resp = requests.get(CHART_ENDPOINT.format(pool_id=pool_id), timeout=30)
    resp.raise_for_status()
    data = resp.json()["data"]
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.set_index("timestamp")


MAX_TRUSTED_APY_CV = 1.0  # これを超える変動係数は「異常値/信頼できないデータ」として除外する


def stability_metrics(history: pd.DataFrame) -> dict:
    apy = history["apy"].dropna()
    tvl = history["tvlUsd"].dropna()
    if len(apy) < 7 or len(tvl) < 7:
        return {
            "apy_median": np.nan, "apy_cv": np.nan, "tvl_trend_pct": np.nan,
            "history_days": len(apy), "flag": "insufficient_history",
        }

    apy_median = float(apy.median())
    apy_cv = float(apy.std() / apy.mean()) if apy.mean() != 0 else np.nan
    tvl_trend_pct = float((tvl.iloc[-1] - tvl.iloc[0]) / tvl.iloc[0]) if tvl.iloc[0] != 0 else np.nan

    flag = "ok"
    if apy_cv is not None and not np.isnan(apy_cv) and apy_cv > MAX_TRUSTED_APY_CV:
        flag = "high_volatility_excluded"

    return {
        "apy_median": apy_median,
        "apy_cv": apy_cv,
        "tvl_trend_pct": tvl_trend_pct,
        "history_days": len(apy),
        "flag": flag,
    }


def composite_score(row: pd.Series) -> float:
    """中央値APYを、変動係数(小さいほど良い)とTVL減少(マイナスなら減点)で調整する。

    apy_cvが閾値を超える(=データが異常/リスクが極端)プールはスコア対象外(NaN)にする。
    """
    if pd.isna(row["apy_cv"]) or pd.isna(row["tvl_trend_pct"]):
        return np.nan
    if row["flag"] == "high_volatility_excluded":
        return np.nan
    stability_penalty = 1 / (1 + row["apy_cv"])
    tvl_penalty = min(1.0, max(0.0, 1 + row["tvl_trend_pct"])) if row["tvl_trend_pct"] < 0 else 1.0
    return row["apy_median"] * stability_penalty * tvl_penalty


def analyze_top_candidates(
    chain: str | None = None,
    stablecoin_only: bool = True,
    min_tvl_usd: float = 5_000_000,
    candidate_pool: int = 40,
) -> pd.DataFrame:
    pools = fetch_pools()
    filtered = pools[pools["tvlUsd"] >= min_tvl_usd]
    if chain:
        filtered = filtered[filtered["chain"].str.lower() == chain.lower()]
    if stablecoin_only:
        filtered = filtered[filtered["stablecoin"]]
    candidates = filtered.sort_values("apy", ascending=False).head(candidate_pool)

    rows = []
    for _, pool in candidates.iterrows():
        try:
            history = fetch_pool_history(pool["pool"])
        except requests.RequestException:
            continue
        metrics = stability_metrics(history)
        rows.append({
            "chain": pool["chain"],
            "project": pool["project"],
            "symbol": pool["symbol"],
            "current_apy": pool["apy"],
            "tvlUsd": pool["tvlUsd"],
            **metrics,
        })

    result = pd.DataFrame(rows)
    result["score"] = result.apply(composite_score, axis=1)
    return result.sort_values("score", ascending=False)


if __name__ == "__main__":
    ranked = analyze_top_candidates()
    DATA_DIR.mkdir(exist_ok=True)
    ranked.to_csv(Path(DATA_DIR) / "ranked_pools.csv", index=False)

    excluded = ranked[ranked["flag"] == "high_volatility_excluded"]
    print(f"高ボラティリティで除外: {len(excluded)}件 (apy_cv > {1.0})")
    print()
    print(ranked.dropna(subset=["score"]).head(15).to_string(index=False))
