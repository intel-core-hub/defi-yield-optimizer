"""プールごとのAPY履歴を取得し、安定性を加味したスコアで再ランキングする。

高APYでもボラティリティが高い/TVLが急減しているプールは危険信号として減点する。
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from risk import fetch_hack_events, fetch_protocol_static_risk, RECENT_HACK_WINDOW_DAYS
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
MIN_TRUSTED_LISTING_AGE_DAYS = 90  # DeFiLlama掲載からこれ未満(または不明)なら実績が浅いとして減点
OLD_HACK_PENALTY = 0.3  # 直近ではない過去のハック歴は除外まではしないが大きく減点する

# analyze_top_candidates()が返すDataFrameの列。候補が0件の場合でもこの列を持つ空の
# DataFrameを返すことで、呼び出し側(allocate_portfolio()など)が列の有無で落ちないようにする。
RANKED_COLUMNS = [
    "chain", "project", "symbol", "pool_id", "current_apy", "tvlUsd",
    "apy_median", "apy_cv", "tvl_trend_pct", "tvl_max_drawdown_pct", "history_days",
    "audits_count", "days_since_defillama_listing", "category",
    "most_recent_hack_date", "hack_days_ago", "was_hacked", "hack_recent",
    "flag", "score",
]

# 借り手・保険引受先など実世界のカウンターパーティに対する信用リスクを内包するカテゴリ。
# スマートコントラクトが無事でも元本が返ってこない可能性があり、TVL/APYの安定性・
# 監査数・稼働歴のどれを見ても検知できないため、機械的に候補から除外する。
EXCLUDED_CATEGORIES = {"Uncollateralized Lending", "RWA Lending", "RWA"}


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

    除外/減点の判定はすべて個々の生カラム(apy_cv, hack_recent, was_hackedなど)から
    独立に行う。「flag」列は表示用の要約ラベルにすぎず、ここでの計算には使わない
    (そうしないと、例えばTVL急減リスクと過去ハックの両方に該当するプールで
    片方のペナルティしか掛からない、というような取りこぼしが起きる)。

    apy_cvが閾値を超える、または直近(RECENT_HACK_WINDOW_DAYS日以内)にハック被害が
    あるプールはスコア対象外(NaN)にする。
    """
    if pd.isna(row["apy_cv"]) or pd.isna(row["tvl_trend_pct"]) or pd.isna(row["tvl_max_drawdown_pct"]):
        return np.nan
    if row["apy_cv"] > MAX_TRUSTED_APY_CV:
        return np.nan
    if row.get("hack_recent", False):
        return np.nan
    if row.get("category") in EXCLUDED_CATEGORIES:
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

    # DeFiLlama掲載からの経過日数がMIN_TRUSTED_LISTING_AGE_DAYS未満、またはデータ不明の
    # 場合は減点する(実際のコントラクト稼働開始日ではなく、あくまでDeFiLlama掲載日からの
    # 経過日数であることに注意。掲載前から稼働していたプロトコルはここで過小評価されうる)。
    listing_age_days = row.get("days_since_defillama_listing", np.nan)
    age_penalty = 0.5 if pd.isna(listing_age_days) else min(1.0, listing_age_days / MIN_TRUSTED_LISTING_AGE_DAYS)

    # 直近ではないが過去にハック歴がある(同一チェーン、またはチェーン不明で全チェーン
    # 適用)場合は、除外はしないが大きく減点する。
    hack_penalty = OLD_HACK_PENALTY if row.get("was_hacked", False) else 1.0

    return row["apy_median"] * stability_penalty * tvl_penalty * audit_penalty * age_penalty * hack_penalty


def _final_flag(row: pd.Series) -> str:
    """表示用の要約フラグ(優先度順)。実際の除外/減点判定はcomposite_scoreが
    個々のカラムから独立に行うので、ここでの優先順位は表示上の見せ方の問題でしかない。
    """
    if row["flag"] == "insufficient_history":
        return "insufficient_history"
    if row.get("category") in EXCLUDED_CATEGORIES:
        return "category_excluded"
    if row.get("hack_recent", False):
        return "hacked_protocol_excluded"
    if row["flag"] == "high_volatility_excluded":
        return "high_volatility_excluded"
    if row.get("was_hacked", False):
        return "past_hack_risk"
    if row["flag"] == "tvl_drawdown_risk":
        return "tvl_drawdown_risk"
    return "ok"


def apply_hack_risk(result: pd.DataFrame, hack_events: pd.DataFrame, now: float | None = None) -> pd.DataFrame:
    """(project, chain)ごとのハック履歴を、候補プールの行に(hack_recent/was_hackedとして)
    突き合わせる。チェーンが分かっているハックは同じチェーンの行にのみ、チェーンが不明な
    ハックは同じプロジェクトの全チェーンの行に適用する(保守的に倒す)。

    ネットワーク呼び出しを含まない純粋関数にしてあるのは、単体テストで実際のAPIを
    叩かずに検証できるようにするため。
    """
    if now is None:
        now = time.time()

    result = result.copy()
    chain_specific_hacks = (
        hack_events.dropna(subset=["chain"])
        .groupby(["project", "chain"])["hack_date"].max()
        .rename("chain_hack_date")
        .reset_index()
    )
    chain_agnostic_hacks = (
        hack_events[hack_events["chain"].isna()]
        .groupby("project")["hack_date"].max()
        .rename("global_hack_date")
        .reset_index()
    )
    result = result.merge(chain_specific_hacks, on=["project", "chain"], how="left")
    result = result.merge(chain_agnostic_hacks, on="project", how="left")

    result["most_recent_hack_date"] = result[["chain_hack_date", "global_hack_date"]].max(axis=1, skipna=True)
    result["hack_days_ago"] = (now - result["most_recent_hack_date"]) / 86400
    result["was_hacked"] = result["most_recent_hack_date"].notna()
    result["hack_recent"] = result["was_hacked"] & (result["hack_days_ago"] <= RECENT_HACK_WINDOW_DAYS)
    return result.drop(columns=["chain_hack_date", "global_hack_date"])


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
            "pool_id": pool["pool"],
            "current_apy": pool["apy"],
            "tvlUsd": pool["tvlUsd"],
            **metrics,
        })

    if not rows:
        return pd.DataFrame(columns=RANKED_COLUMNS)

    result = pd.DataFrame(rows)

    static_risk = fetch_protocol_static_risk()
    result = result.merge(static_risk, on="project", how="left")
    result["audits_count"] = result["audits_count"].fillna(0)

    hack_events = fetch_hack_events()
    result = apply_hack_risk(result, hack_events)

    result["flag"] = result.apply(_final_flag, axis=1)
    result["score"] = result.apply(composite_score, axis=1)
    return result.sort_values("score", ascending=False)


def allocate_portfolio(
    ranked: pd.DataFrame,
    top_n: int = 10,
    max_pool_weight: float = 0.20,
    max_protocol_weight: float = 0.35,
    max_chain_weight: float = 0.40,
) -> pd.DataFrame:
    """スコア上位プールへの資金配分「案」を作る(実際の送金・スワップは行わない)。

    スコアの高い順に、単一プール/単一プロトコル(同じコントラクト・チームへの集中)/
    単一チェーン(ブリッジ障害・チェーン停止などチェーン固有障害への集中)の上限に
    収まる範囲で、貪欲(greedy)に重みを割り当てる。チェーンの上限をプロトコルより
    緩め(既定40% vs 35%)にしているのは、単一プロトコルのコントラクトバグの方が
    単一チェーンの障害より発生確率・被害範囲ともに大きい(プロトコル固有のリスクの
    方がありふれている)という判断による。厳密な数値根拠はなく目安。

    以前は「スコア比例の重みから始め、上限超過分を他へ比例配分し直す」を複数の
    上限(プール/プロトコル/チェーン)について繰り返す実装だったが、1つの行が
    複数の上限に同時に抵触するケースで再配分が振動し、実データでスコアが並程度の
    プールの重みがほぼ0に潰れる不具合を確認した。貪欲法は1回のパスで完結するため
    そうした振動が原理的に起きない。トレードオフとして、上位に採用できるプールの
    多様性がtop_n件で足りない場合、100%を配分しきれず端数が残ることがある
    (その分は、あえて配分先を作らない=未配分として扱う)。
    """
    candidates = ranked.dropna(subset=["score"]).sort_values("score", ascending=False).head(top_n).copy()
    if candidates.empty:
        return candidates.assign(weight=pd.Series(dtype=float))

    remaining = 1.0
    protocol_used: dict[str, float] = {}
    chain_used: dict[str, float] = {}
    weights = []
    for _, row in candidates.iterrows():
        protocol_room = max_protocol_weight - protocol_used.get(row["project"], 0.0)
        chain_room = max_chain_weight - chain_used.get(row["chain"], 0.0)
        w = max(0.0, min(max_pool_weight, protocol_room, chain_room, remaining))
        weights.append(w)
        remaining -= w
        protocol_used[row["project"]] = protocol_used.get(row["project"], 0.0) + w
        chain_used[row["chain"]] = chain_used.get(row["chain"], 0.0) + w

    candidates["weight"] = weights
    return candidates[["chain", "project", "symbol", "pool_id", "score", "weight"]].sort_values(
        "weight", ascending=False
    )


if __name__ == "__main__":
    ranked = analyze_top_candidates()
    DATA_DIR.mkdir(exist_ok=True)
    ranked.to_csv(Path(DATA_DIR) / "ranked_pools.csv", index=False)

    excluded = ranked[ranked["flag"] == "high_volatility_excluded"]
    drawdown_risk = ranked[ranked["flag"] == "tvl_drawdown_risk"]
    hacked = ranked[ranked["flag"] == "hacked_protocol_excluded"]
    past_hack = ranked[ranked["flag"] == "past_hack_risk"]
    category_excluded = ranked[ranked["flag"] == "category_excluded"]
    print(f"高ボラティリティで除外: {len(excluded)}件 (apy_cv > {MAX_TRUSTED_APY_CV})")
    print(f"TVL急減リスクで減点: {len(drawdown_risk)}件 (max_drawdown < {MAX_TRUSTED_TVL_DRAWDOWN})")
    print(f"直近{RECENT_HACK_WINDOW_DAYS}日以内のハック歴で除外: {len(hacked)}件")
    print(f"それより前のハック歴で減点: {len(past_hack)}件")
    print(f"カウンターパーティリスク系カテゴリ({', '.join(sorted(EXCLUDED_CATEGORIES))})で除外: {len(category_excluded)}件")
    print()
    print(ranked.dropna(subset=["score"]).head(15).to_string(index=False))

    allocation = allocate_portfolio(ranked)
    unallocated = 1.0 - allocation["weight"].sum()
    print()
    print(
        "--- 配分案(スコア順の貪欲割り当て、プール上限20%/プロトコル上限35%/"
        "チェーン上限40%。実際の送金は行わない) ---"
    )
    print(allocation.to_string(index=False))
    print(f"未配分(上限に収まる候補が足りなかった分。増額前提ではなく単に据え置き): {unallocated:.1%}")
