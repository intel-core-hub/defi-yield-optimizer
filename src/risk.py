"""プロトコルの監査状況・過去のハック履歴からリスクシグナルを取得するユーティリティ(APIキー不要)。

DeFiLlamaの`/protocols`と`/hacks`エンドポイントは、yields.llama.fiの`pools`が返す
`project`スラッグ(例: "aave-v3")とそのまま一致するので、それをキーに突き合わせる。

`/protocols`の`id`はバージョンごとに別々に振られている(例: aave-v2とaave-v3のidは別、
複数slugが同じidを共有することはない)ので、`/hacks`の`defillamaId`で突き合わせる限り
「別バージョンのハックが誤って巻き込まれる」ことは起きない。

ただし以下2点はより粗い/誤検出の余地があるため、この2点を精緻化する:
- ハックは特定チェーンでのみ起きることが多い(`/hacks`の`chain`フィールド)。プロジェクト単位
  でしか見ないと、無関係な別チェーンのプールまで巻き込んで除外してしまう。
  → `fetch_hack_events()`は(project, chain)単位でハック履歴を返し、chainが不明な記録
  だけプロジェクト全体に適用する(保守的に倒す)。
- 発生時期を考慮しないと、何年も前に一度ハックされて以降ずっと安全に稼働しているプロトコル
  と、直近ハックされたばかりのプロトコルが同列(永久除外)になってしまう。
  → `RECENT_HACK_WINDOW_DAYS`より新しいハックは除外対象、それより古いものは減点に留める
  判断をscoring.py側に委ねられるよう、ハックの日付も返す。
"""

import sys
import time

import pandas as pd
import requests

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

PROTOCOLS_ENDPOINT = "https://api.llama.fi/protocols"
HACKS_ENDPOINT = "https://api.llama.fi/hacks"

RECENT_HACK_WINDOW_DAYS = 730  # これより新しいハックのみ「除外」対象。古いものは減点に留める


def fetch_protocol_static_risk() -> pd.DataFrame:
    """プロジェクト単位(チェーンを問わない)の監査数・稼働歴。"""
    protocols = requests.get(PROTOCOLS_ENDPOINT, timeout=30).json()

    now = time.time()
    rows = []
    for p in protocols:
        try:
            audits_count = int(p.get("audits"))
        except (TypeError, ValueError):
            audits_count = 0

        listed_at = p.get("listedAt")
        protocol_age_days = (now - listed_at) / 86400 if listed_at else float("nan")

        rows.append({
            "project": p["slug"],
            "audits_count": audits_count,
            "protocol_age_days": protocol_age_days,
        })

    # 同じslugが複数エントリに出ることがあるので、より監査数が多い方の情報を残す。
    return (
        pd.DataFrame(rows)
        .sort_values("audits_count", ascending=False)
        .drop_duplicates(subset="project", keep="first")
    )


def fetch_hack_events() -> pd.DataFrame:
    """(project, chain)単位のハック履歴。

    `chain`がNaNの行は、そのハックがどのチェーンを対象にしたか`/hacks`側に記録が
    無かったことを意味し、呼び出し側ではプロジェクトの全チェーンに適用すべき
    (保守的に倒す)。
    """
    protocols = requests.get(PROTOCOLS_ENDPOINT, timeout=30).json()
    hacks = requests.get(HACKS_ENDPOINT, timeout=30).json()
    id_to_slug = {p["id"]: p["slug"] for p in protocols}

    rows = []
    for h in hacks:
        slug = id_to_slug.get(h.get("defillamaId"))
        if not slug:
            continue
        chains = h.get("chain") or [None]
        for chain in chains:
            rows.append({
                "project": slug,
                "chain": chain,
                "hack_date": h.get("date"),
                "hack_name": h.get("name"),
            })

    if not rows:
        return pd.DataFrame(columns=["project", "chain", "hack_date", "hack_name"])
    return pd.DataFrame(rows)


if __name__ == "__main__":
    static_risk = fetch_protocol_static_risk()
    hack_events = fetch_hack_events()
    now = time.time()
    hack_events["days_ago"] = (now - hack_events["hack_date"]) / 86400
    recent = hack_events[hack_events["days_ago"] <= RECENT_HACK_WINDOW_DAYS]

    print(f"プロトコル数: {len(static_risk)}")
    print(f"監査0件: {int((static_risk['audits_count'] == 0).sum())}件")
    print(f"ハック記録(プロジェクト×チェーン単位、重複含む): {len(hack_events)}件")
    print(f"うち直近{RECENT_HACK_WINDOW_DAYS}日以内: {len(recent)}件")
    print()
    print(
        hack_events.sort_values("hack_date", ascending=False)
        .head(15)[["project", "chain", "hack_name", "days_ago"]]
        .to_string(index=False)
    )
