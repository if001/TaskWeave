---
name: user-profile-extraction
description: 会話や保存済みアーティファクトからユーザーの長期的に有用な属性や傾向を抽出し、LangMem の長期記憶を更新する。
---
# Skill: Interest Trends From Memory And Artifacts

## 目的
LangMem の長期記憶と、PGVectorStore に保存された /artifacts のメタ情報を利用して、ユーザーの関心傾向を整理する。

## 利用可能ツール
- artifact_search
- read_file

## 入力元（優先順）
1) 実行前に LangMem から注入された長期記憶
2) /artifacts/（meta は artifact_search で検索し、raw_path を辿る）

## 基本フロー
1) 実行前に LangMem から注入された長期記憶を確認する。
2) artifact_search で関連アーティファクトの meta を検索する。
3) raw_path を辿って必要な raw.json だけ読む。
4) 安定的な傾向と一時的な傾向を分けて要約する。
5) 更新が必要な情報は応答に反映し、保存は after_invoke の ReflectionExecutor に委ねる。

## ルール
- 安定的な傾向は LangMem の profile memory に残す。
- 最近の関心や継続中の話題は LangMem の topics memory に残す。
- 保存はファイル追記ではなく ReflectionExecutor による非同期更新で行う。

## 保存対象
以下のような情報は LangMem の profile memory に保存する価値がある。
- ユーザーの呼び方やニックネーム
- ユーザーの属性/住んでいるエリア
- 技術領域
- 興味分野
- 作業スタイル
- よく扱うツール
- 好み
- 長期目標
- 作業環境
- 学習対象

## 重要原則
- 推測よりも観察された事実を優先する
- 一時的な情報をプロフィールとして固定しない
- 長期的に役立つ情報だけ残す
