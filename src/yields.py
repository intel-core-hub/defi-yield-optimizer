"""DeFiLlama公開APIから利回り(APY)データを取得するユーティリティ(APIキー不要)。"""

from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
POOLS_ENDPOINT = "https://yields.llama.fi/pools"

# 環境によっては(特にWindows + brotlicffiの組み合わせ)DeFiLlamaが返すbrotli圧縮
# レスポンスのストリーミングデコードでrequests側が失敗することがある
# (urllib3.exceptions.DecodeError: content-encoding: br)。gzip/deflateのみを
# 許可することでbrotliネゴシエーション自体を避ける。
NO_BROTLI_HEADERS = {"Accept-Encoding": "gzip, deflate"}


def fetch_pools() -> pd.DataFrame:
    resp = requests.get(POOLS_ENDPOINT, timeout=30, headers=NO_BROTLI_HEADERS)
    resp.raise_for_status()
    data = resp.json()["data"]
    return pd.DataFrame(data)


def top_pools_by_apy(
    df: pd.DataFrame,
    chain: str | None = None,
    stablecoin_only: bool = False,
    min_tvl_usd: float = 1_000_000,
    top_n: int = 20,
) -> pd.DataFrame:
    filtered = df[df["tvlUsd"] >= min_tvl_usd]
    if chain:
        filtered = filtered[filtered["chain"].str.lower() == chain.lower()]
    if stablecoin_only:
        filtered = filtered[filtered["stablecoin"]]
    cols = ["chain", "project", "symbol", "apy", "apyBase", "apyReward", "tvlUsd"]
    return filtered.sort_values("apy", ascending=False)[cols].head(top_n)


if __name__ == "__main__":
    pools = fetch_pools()
    DATA_DIR.mkdir(exist_ok=True)
    pools.to_csv(DATA_DIR / "pools_snapshot.csv", index=False)
    print(top_pools_by_apy(pools, stablecoin_only=True))
