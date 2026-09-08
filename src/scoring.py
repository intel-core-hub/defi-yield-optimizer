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

from risk import fetch_protocol_risk
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
MAX_TRUSTED_TVL_DRAWDOWN = -0.5  # ピークから50%超のTVL下落があった場合は取り付け騒ぎリスクとして減点
MIN_TRUSTED_PROTOCOL_AGE_DAYS = 90  # これ未満(または不明)の稼働歴は実績が浅いとして減点


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

    # 期間中に一時的にTVLが急減した(取り付け騒ぎ的な動き)が最終日には戻っている場合、
    # tvl_trend_pctだけでは検知できない。ピークからの最大下落率も見ておく。
    running_max = tvl.cummax()
    tvl_max_drawdown_pct = float(((tvl - running_max) / running_max).min())

    flag = "ok"
    if apy_cv is not None and not np.isnan(apy_cv) and apy_cv > MAX_TRUSTED_APY_CV:
        flag = "high_volatility_excluded"
    elif tvl_max_drawdown_pct < MAX_TRUSTED_TVL_DRAWDOWN:
        flag = "tvl_drawdown_risk"

    return {
        "apy_median": apy_median,
        "apy_cv": apy_cv,
        "tvl_trend_pct": tvl_trend_pct,
        "tvl_max_drawdown_pct": tvl_max_drawdown_pct,
        "history_days": len(apy),
        "flag": flag,
    }


def composite_score(row: pd.Series) -> float:
    """中央値APYを、安定性(APY変動・TVL推移)とプロトコルリスク(監査数・稼働歴・
    過去のハック有無)で調整する。

    apy_cvが閾値を超える、または過去にハック被害があるプールはスコア対象外(NaN)にする。
    """
    if pd.isna(row["apy_cv"]) or pd.isna(row["tvl_trend_pct"]) or pd.isna(row["tvl_max_drawdown_pct"]):
        return np.nan
    if row["flag"] in ("high_volatility_excluded", "hacked_protocol_excluded"):
        return np.nan

    stability_penalty = 1 / (1 + row["apy_cv"])
    # 期間終了時点の増減(tvl_trend_pct)と期間中の最大下落(tvl_max_drawdown_pct)の
    # うち悪い方でTVLペナルティを計算する。終値だけ見ると途中の急落からの回復を見逃すため。
    worst_tvl_move = min(row["tvl_trend_pct"], row["tvl_max_drawdown_pct"])
    tvl_penalty = min(1.0, max(0.0, 1 + worst_tvl_move)) if worst_tvl_move < 0 else 1.0

    # 監査0件は0.5倍、1件は0.75倍、2件以上は満額。DeFiLlamaの自己申告データなので
    # 「監査ありと申告が無い」ことは即危険を意味しないが、保守的に減点しておく。
    audits_count = row.get("audits_count", 0)
    audits_count = 0 if pd.isna(audits_count) else audits_count
    audit_penalty = min(1.0, 0.5 + 0.25 * min(audits_count, 2))

    # 稼働歴がMIN_TRUSTED_PROTOCOL_AGE_DAYS未満、またはデータ不明の場合は減点する。
    age_days = row.get("protocol_age_days", np.nan)
    age_penalty = 0.5 if pd.isna(age_days) else min(1.0, age_days / MIN_TRUSTED_PROTOCOL_AGE_DAYS)

    return row["apy_median"] * stability_penalty * tvl_penalty * audit_penalty * age_penalty


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

    risk = fetch_protocol_risk()
    result = result.merge(risk, on="project", how="left")
    result["audits_count"] = result["audits_count"].fillna(0)
    result["was_hacked"] = result["was_hacked"].fillna(False)
    result.loc[result["was_hacked"], "flag"] = "hacked_protocol_excluded"

    result["score"] = result.apply(composite_score, axis=1)
    return result.sort_values("score", ascending=False)


def allocate_portfolio(
    ranked: pd.DataFrame,
    top_n: int = 10,
    max_pool_weight: float = 0.20,
    max_protocol_weight: float = 0.35,
) -> pd.DataFrame:
    """スコア上位プールへの資金配分「案」を作る(実際の送金・スワップは行わない)。

    スコアに比例した重みから始め、単一プール/単一プロトコルへの集中を避けるため
    上限(max_pool_weight/max_protocol_weight)でキャップし、超過分を他へ比例配分し直す。
    """
    candidates = ranked.dropna(subset=["score"]).sort_values("score", ascending=False).head(top_n).copy()
    if candidates.empty:
        return candidates.assign(weight=pd.Series(dtype=float))

    weight = candidates["score"] / candidates["score"].sum()

    for _ in range(50):
        changed = False

        over_pool = weight > max_pool_weight + 1e-9
        if over_pool.any():
            excess = (weight[over_pool] - max_pool_weight).sum()
            weight[over_pool] = max_pool_weight
            under_pool = ~over_pool
            if under_pool.any() and weight[under_pool].sum() > 0:
                weight[under_pool] += excess * (weight[under_pool] / weight[under_pool].sum())
            changed = True

        protocol_totals = weight.groupby(candidates["project"]).transform("sum")
        over_protocol = protocol_totals > max_protocol_weight + 1e-9
        if over_protocol.any():
            scale = max_protocol_weight / protocol_totals[over_protocol]
            excess = (weight[over_protocol] * (1 - scale)).sum()
            weight[over_protocol] *= scale
            under_protocol = ~over_protocol
            if under_protocol.any() and weight[under_protocol].sum() > 0:
                weight[under_protocol] += excess * (weight[under_protocol] / weight[under_protocol].sum())
            changed = True

        if not changed:
            break

    candidates["weight"] = weight / weight.sum()
    return candidates[["chain", "project", "symbol", "score", "weight"]].sort_values("weight", ascending=False)


if __name__ == "__main__":
    ranked = analyze_top_candidates()
    DATA_DIR.mkdir(exist_ok=True)
    ranked.to_csv(Path(DATA_DIR) / "ranked_pools.csv", index=False)

    excluded = ranked[ranked["flag"] == "high_volatility_excluded"]
    drawdown_risk = ranked[ranked["flag"] == "tvl_drawdown_risk"]
    hacked = ranked[ranked["flag"] == "hacked_protocol_excluded"]
    print(f"高ボラティリティで除外: {len(excluded)}件 (apy_cv > {MAX_TRUSTED_APY_CV})")
    print(f"TVL急減リスクで減点: {len(drawdown_risk)}件 (max_drawdown < {MAX_TRUSTED_TVL_DRAWDOWN})")
    print(f"過去のハック歴で除外: {len(hacked)}件")
    print()
    print(ranked.dropna(subset=["score"]).head(15).to_string(index=False))

    allocation = allocate_portfolio(ranked)
    print()
    print("--- 配分案(スコア比例、プール上限20%/プロトコル上限35%。実際の送金は行わない) ---")
    print(allocation.to_string(index=False))
