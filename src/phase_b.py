"""Phase B(監査・実コントラクト確認)の承認状態をスコアリング結果に反映する。

`docs/phase-b/*.yaml` のmanifestは、DeFiLlamaのpool_id単位で監査状況を記録した
Git管理下のドキュメントであり、`phase_b_status`が"approved"のプールのみが
Phase C(実資金投入)の対象候補になる。承認の可否そのものは監査原本を読んだ
ユーザー本人が判断するものであり、このモジュールはmanifestの内容をそのまま
スコアリング結果に反映するだけで、承認の妥当性を判断しない。
"""

from pathlib import Path

import pandas as pd
import yaml

APPROVED_STATUS = "approved"
VALID_STATUSES = {"pending", "approved", "rejected"}


def load_manifest(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    for pool in manifest.get("pools", []):
        status = pool.get("phase_b_status")
        if status not in VALID_STATUSES:
            raise ValueError(
                f"pool_id={pool.get('pool_id')}のphase_b_statusが不正です: {status!r}"
                f"({sorted(VALID_STATUSES)}のいずれかである必要があります)"
            )
    return manifest


def phase_b_status_map(manifest: dict) -> dict:
    return {
        pool["pool_id"]: pool["phase_b_status"]
        for pool in manifest.get("pools", [])
        if pool.get("pool_id")
    }


def apply_phase_b_gate(ranked: pd.DataFrame, manifest: dict) -> pd.DataFrame:
    """manifestで"approved"以外のプールを配分対象(score)から除外する。

    manifestに記載の無いpool_idは"not_reviewed"として扱い、同様に除外する
    (レビュー対象に入っていない=まだ安全と確認されていない、と保守的に倒す)。
    """
    result = ranked.copy()
    status_map = phase_b_status_map(manifest)
    result["phase_b_status"] = result["pool_id"].map(status_map).fillna("not_reviewed")
    result.loc[result["phase_b_status"] != APPROVED_STATUS, "score"] = float("nan")
    return result
