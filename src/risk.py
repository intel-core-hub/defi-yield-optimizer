"""プロトコルの監査状況・過去のハック履歴からリスクシグナルを取得するユーティリティ(APIキー不要)。

DeFiLlamaの`/protocols`と`/hacks`エンドポイントは、yields.llama.fiの`pools`が返す
`project`スラッグ(例: "aave-v3")とそのまま一致するので、それをキーに突き合わせる。

注意: `/hacks`は`defillamaId`単位で紐づいており、同じdefillamaIdを共有する旧バージョン
(例: aaveのv1/v2/v3)がまとめて「ハック被害あり」扱いになることがある。過剰検出(false
positive)の方向に倒れるが、安全側に振る設計として許容している。
"""

import sys
import time

import pandas as pd
import requests

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

PROTOCOLS_ENDPOINT = "https://api.llama.fi/protocols"
HACKS_ENDPOINT = "https://api.llama.fi/hacks"


def fetch_protocol_risk() -> pd.DataFrame:
    protocols = requests.get(PROTOCOLS_ENDPOINT, timeout=30).json()
    hacks = requests.get(HACKS_ENDPOINT, timeout=30).json()
    hacked_ids = {h["defillamaId"] for h in hacks if h.get("defillamaId")}

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
            "was_hacked": p.get("id") in hacked_ids,
        })

    # 同じslugが複数エントリに出ることがあるので、最もリスクが低い(監査数が多い/
    # ハック無し)方の情報を優先して残す。
    return (
        pd.DataFrame(rows)
        .sort_values(["was_hacked", "audits_count"], ascending=[True, False])
        .drop_duplicates(subset="project", keep="first")
    )


if __name__ == "__main__":
    risk = fetch_protocol_risk()
    print(f"プロトコル数: {len(risk)}")
    print(f"過去にハック被害あり: {int(risk['was_hacked'].sum())}件")
    print(f"監査0件: {int((risk['audits_count'] == 0).sum())}件")
    print()
    print(risk[risk["was_hacked"]].head(15).to_string(index=False))
