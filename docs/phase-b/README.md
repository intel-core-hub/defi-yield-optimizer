# Phase B manifest — 監査・実コントラクト確認の記録

Phase B(実資金投入前の監査・コントラクト確認)の結果を、日付ごとのYAMLファイル
として **Git管理下に固定** するためのディレクトリ。CSV等のスコアリング出力
(`data/`配下、gitignore対象)とは違い、ここに置くファイルは常にコミットする。

## なぜ必要か

`python src/scoring.py` のランキングは日々変動する。「Pendleは監査済みだから
このプールも安全」ではなく、「この`pool_id`が指すこの市場の、この実装コード・
このコミットを対象にした監査を確認した」という対応関係を、ランキングの変動とは
独立に固定しておく必要がある。これが無いと、ある日承認したプールと翌日ランキング
上位に出てくる同名プロトコルの別プールを取り違えるリスクがある。

## ファイル

* `README.md` — このファイル。
* `YYYY-MM-DD.yaml` — その日時点でのPhase Bレビュー結果。`selection_snapshot`
  でレビュー対象のpool_idを固定し、`pools`配下に各プールの調査結果を記録する。

## `phase_b_status` の意味と権限

```
pending   — レビュー未完了、または一部項目が未確認
approved  — 全項目確認済み、Phase C(実資金投入)の対象候補にしてよい
rejected  — 確認の結果、対象から外すと判断した
```

**`approved`への格上げは必ずユーザー本人が行う。** Claude Codeがこのmanifestを
生成・更新する際は、DeFiLlama/Pendle等の公開APIから機械的に取得できる客観データ
(アドレス・監査リンクの存在など)までしか埋めず、監査原本の読解・Critical/High
指摘の解消判断・管理権限の妥当性判断は行わない。それらの項目は空欄
(`null`)のまま残し、`reviewer_notes`に「要人手確認」の内容を明記する。

## スコアリングへの反映

```bash
python src/scoring.py <投入予定額(円)> --phase-b docs/phase-b/2026-09-11.yaml
```

manifestを指定すると、`phase_b_status`が`approved`のプール以外はスコアが
`NaN`になり、配分案(`allocate_portfolio`)の対象から自動的に除外される。
manifestに載っていない`pool_id`は`not_reviewed`として扱われ、同様に除外される
(未レビュー=安全未確認、として保守的に倒す)。

## 運用ルール

* レビュー対象のプールを変更したい場合(ランキングが変わった等)は、既存ファイルを
  上書きせず新しい日付のYAMLを追加する(過去のレビュー記録を消さない)。
* `approved`にした後で以下が発生した場合、承認は失効したものとして扱い、
  新しいレビューファイルで`phase_b_status`を`pending`に戻す。
  * 実装(implementation)の変更・proxy upgrade
  * 対象プロトコルでのexploit発生
  * 未解決のCritical/High脆弱性の公表
  * underlying protocol側の重大な変更
