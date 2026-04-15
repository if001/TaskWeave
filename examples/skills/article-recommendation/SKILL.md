---
name: article-recommendation
description: 保存済み artifact と長期記憶をもとに、ユーザーが次に読むと良い web 記事を 1 件推薦し、/tmp に履歴を保存して重複推薦を避ける。
---
# Skill: Next Article Recommendation

## 目的
保存済み artifact の記事や tags、LangMem から注入された長期記憶を使って、
ユーザーが次に読むと良い web page 記事を 1 件推薦する。
出力は以下の 3 つに絞る。

- タイトル
- 要約
- URL

同時に、推薦履歴を `/tmp` に保存し、同じ記事や似た記事を繰り返し推薦しにくくする。

## 利用可能ツール
- artifact_search
- read_file
- write_file
- web_list
- web_page

## 入力元
1) 実行前に LangMem から注入された長期記憶
2) artifact_search で取得した article / web_page / url_digest 系 artifact の meta
3) raw_path から読んだ artifact JSON
4) `/tmp/article_recommendations/history.jsonl`

## 推薦手順
1) 長期記憶から、興味がありそうなテーマ・キーワード・好みを 3 つ以内に絞る。
2) `/tmp/article_recommendations/history.jsonl` があれば読み、過去に推薦した URL / title / tag を確認する。
3) artifact_search を 2-3 回使い、興味キーワードや tags に近い記事候補を集める。
4) raw_path を辿って artifact JSON を読み、少なくとも以下を確認する。
   - URL
   - タイトル
   - 要約または本文の要点
   - tags
5) 候補を絞る。以下を優先する。
   - ユーザーの長期記憶や最近の話題に合う
   - 直近に推薦していない
   - 既読候補と完全一致しない
   - タイトルだけでなく内容にも新しさがある
6) 候補が不足する場合のみ、web_list / web_page を使って追加候補を探す。
7) 最終的に 1 件だけ選び、タイトル・要約・URL を出力する。
8) 選んだ結果を `/tmp/article_recommendations/history.jsonl` に追記する。

## 重複回避ルール
- 同じ URL は推薦しない。
- 同じ title は推薦しない。
- tags がほぼ同じ記事が直近 3 件に含まれる場合は強く避ける。
- 同一ドメインかつ同一テーマの記事が連続しないようにする。
- 候補が重複しかない場合は、その旨を短く明記しつつ最も新規性が高いものを 1 件選ぶ。

## 履歴ファイル形式
履歴は `/tmp/article_recommendations/history.jsonl` に 1 行 1 recommendation で保存する。
各行は以下の JSON とする。

```json
{"recommended_at":"2026-03-28T12:34:56Z","title":"...","url":"...","summary":"...","tags":["..."],"artifact_id":"..."}
```

append ができない場合は、既存内容を read_file で読み、末尾に 1 行追加して write_file で全体を書き戻す。

## 出力形式
最終出力は次だけにする。

タイトル: ...
要約: ...
URL: ...

## 注意
- 回答前に必ず履歴を更新する。
- raw artifact をそのまま大量に貼らず、推薦理由に必要な部分だけ使う。
- 候補が artifact だけで十分なら web 検索は追加しない。
- 推薦は 1 件に固定する。
