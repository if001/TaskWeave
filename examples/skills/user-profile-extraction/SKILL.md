---
name: user-profile-extraction
description: 会話や保存済みアーティファクトからユーザーの長期的に有用な属性や傾向を抽出し、/memories/profile/ を更新する。
---
# Skill: Interest Trends From Files

## 目的
ファイルシステム上の /memories/ と、PGVectorStore に保存された /artifacts のメタ情報を利用して、ユーザーの関心傾向を整理・更新する。

## 利用可能ツール
- ls / glob / grep / read_file

## 入力元（優先順）
1) /memories/profile/
2) /memories/topics/
4) /artifacts/（meta は artifact_search で検索し、raw_path を辿る）

## 基本フロー
1) /memories/profile/ を glob で列挙する。
2) /memories/topics/ を glob で列挙する。
3) 必要に応じて grep で頻出語や関心語を抽出する。
4) 関連ファイルだけ read_file で内容を読む。
5) 安定的な傾向（profile）と一時的な傾向（topics）を分けて要約する。
6) 更新が必要なら /memories/profile/ や /memories/topics/ に追記する。

## ルール
- 「安定的な傾向」は profile に追記。
- 「最近の関心や継続中の話題」は topics に追記。
- 未完了事項や継続タスクは tasks に追記。
- 既存内容がある場合は上書きではなく追記。


## 保存対象
以下のような情報は `/memories/profile/` に保存する価値がある。
例:
- ユーザーの呼び方やニックネーム
- ユーザーの属性/住んでいるエリア
- 技術領域（例: Python / Rust / LLM）
- 興味分野
- 作業スタイル
- よく扱うツール
- 好み（例: CLI / Emacs / Linux）
- 長期目標
- 作業環境
- 学習対象

## 要約して保存
長文ではなく、
簡潔な要約として保存する。

例:
悪い例
「ユーザーはPythonをよく使っているらしい」

良い例
技術: Python

## 既存情報の更新
既存プロフィールがある場合:
1. 新しい情報を比較
2. 必要なら更新
3. 変更理由を短く残す

## 更新タイミング
以下のタイミングでプロフィール更新を検討する。
- 新しい技術領域が判明
- ユーザーの好みが明確化
- 長期的な目標が判明
- 同じテーマが何度も出てきた

## 重要原則
- 推測よりも観察された事実を優先する
- 一時的な情報をプロフィールに保存しない
- 長期的に役立つ情報だけ保存する

## 出力
- 安定的な傾向: ...
- 一時的な傾向: ...
- 更新内容: ...
