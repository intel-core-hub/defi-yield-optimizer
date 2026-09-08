"""定期実行して、ranked_poolsの判定結果を日付付きで`data/snapshots/`に蓄積する。

`src/backtest.py`はDeFiLlamaが提供する各プールの履歴データ(最大でも2年半程度、
プールごとにバラバラの期間)に頼った疑似バックテストしかできない。ここで自分たちの
判定結果そのものを定期的に記録しておけば、数ヶ月〜数年蓄積した時点で「実際にその時点で
ランキング上位だったプールのその後」を追える、本来の意味でのバックテストの材料になる。

実行方法の目安: 1日1回程度(頻繁すぎてもAPYの動きは大きく変わらない)。スケジューリング
方法(Windows タスクスケジューラ等)はREADME参照。
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from scoring import analyze_top_candidates
from yields import DATA_DIR

SNAPSHOT_DIR = DATA_DIR / "snapshots"


def save_snapshot() -> Path:
    ranked = analyze_top_candidates()
    now = datetime.now(timezone.utc)
    ranked.insert(0, "snapshot_at", now.isoformat())

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SNAPSHOT_DIR / f"{now:%Y-%m-%d}.csv"
    ranked.to_csv(out_path, index=False)
    return out_path


def load_snapshots() -> pd.DataFrame:
    """蓄積済みの全スナップショットを1つのDataFrameに結合する(無ければ空)。"""
    files = sorted(SNAPSHOT_DIR.glob("*.csv"))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_csv(f, parse_dates=["snapshot_at"]) for f in files]
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    path = save_snapshot()
    all_snapshots = load_snapshots()
    print(f"スナップショットを保存: {path}")
    print(f"累計スナップショット日数: {all_snapshots['snapshot_at'].dt.date.nunique()}日")
