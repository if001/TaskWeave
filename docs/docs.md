# Agent Runtime PRD

## 文書の位置づけ

この文書は、非同期 worker を持つエージェントシステム向けの Runtime ライブラリ TaskWeave の初期 PRD である。
本ドキュメントは完成版ではなく、実装を進めながら更新していくたたき台とする。

---

# 1. PRD

## 1.1 背景

一般的な agent フレームワークは、ユーザー入力に対してその場で応答を返すことを主目的としている。
LangChain / LangGraph / Deep Agents は、agent 実装、tool 利用、subagent、durable execution、artifact 的な filesystem 活用などを強力に支援する。

一方で、今回作りたい仕組みはそれより一段上にある。

ユーザーとの会話だけでなく、

* user_request を task としてキューに積む
* main agent が task を拾って処理する
* 必要であれば worker を起動する
* worker は main が待たずに裏で動く
* worker は終了時に notification task を生成する
* runtime が notification task を拾ってユーザーへ配送する
* 定期実行・遅延実行・再実行を行う
* 実行結果を artifact として保存し、後で利用する

という **task runtime / scheduler / async execution** が中核になる。

このプロジェクトでは、agent 自体は LangChain / LangGraph / Deep Agents の仕組みをできるだけ流用し、プロダクト資産としては **runtime 側** を中心に整備する。

## 1.2 プロダクトの目的

### 目的

任意の agent 実装を後ろに差し込み可能な、軽量で責務分離された **task runtime ライブラリ** を提供する。

### 目指すもの

* main / worker という役割を持つ agent システムを支える共通 runtime
* 非同期実行、定期実行、通知、再試行、状態遷移を扱える execution 基盤
* LangChain / LangGraph / Deep Agents を自然に利用できる integration のしやすさ
* agent や artifact の詳細を runtime-core に持ち込まない、疎結合な構成
* 将来的に CLI / Web / Discord など複数チャネルで利用できる基盤

### 目指さないもの

* 独自の agent framework を一から作ること
* LLM orchestration をすべて runtime-core に埋め込むこと
* workflow / graph / tool 実装の内部構造まで runtime が理解すること
* artifact store や vector DB の完全な標準化を最初から行うこと
* すべてのエージェント実装を同一インターフェースに完全統一すること

## 1.3 解決したい課題

1. main agent が worker の完了待ちをしてしまうと、対話レイテンシが悪化する。
2. 定期実行や遅延実行を agent の prompt / memory に任せると不安定になる。
3. artifact や notification を agent 内部状態に混ぜると責務が曖昧になる。
4. LangChain / LangGraph の利点を活かしつつ、プロダクト固有の runtime を持ちたい。
5. 再試行・キャンセル・スケジュール・重複抑止などを deterministic に扱いたい。

## 1.4 対象ユーザー

### 直接の利用者

* Python で agent システムを構築する開発者
* LangChain / LangGraph / Deep Agents を使って main / worker 構成を作りたい開発者
* 非同期タスク実行を伴う personal assistant / research agent / background worker を作りたい開発者

### 間接的な利用者

* 上記 runtime を利用したアプリケーションのエンドユーザー

## 1.5 コアユースケース

### ユースケース 1: 通常の user_request 処理

1. ユーザー入力が user_request task として enqueue される
2. runtime が main handler を選ぶ
3. main agent が応答する
4. 必要であれば worker_run task を生成する
5. main は worker 完了を待たず即時返答する

### ユースケース 2: deep research の後送

1. main が「今すぐ答えられること」は回答する
2. 追加調査が必要なら worker_run task を生成する
3. worker が Web 取得、要約、artifact 保存を行う
4. worker が notification task を作る
5. runtime が notification を配送する

### ユースケース 3: 定期リサーチ

1. scheduler が定期的に worker_run task を生成する
2. worker がユーザーの過去履歴や interest artifact を参照する
3. 新しい調査結果を artifact として保存する
4. 条件を満たす場合のみ notification を生成する

### ユースケース 4: artifact の再利用

1. 過去 worker が作成した artifact が存在する
2. main または別 worker が artifact を読んで利用する
3. 必要に応じて再要約や整理 artifact を生成する

## 1.6 スコープ

### In Scope

* task queue
* scheduler / delayed task / periodic task
* task state machine
* handler dispatch
* retry / timeout / cancellation の基礎
* main / worker / notification の代表的 task フロー
* LangChain / LangGraph / Deep Agents を包む integration 層
* artifact / notification を runtime-core 外の service として接続するための境界

### Out of Scope

* 独自 LLM API client の実装
* 独自 agent planner / tool executor の実装
* artifact 保存形式の最終標準化
* 分散キューの本格運用機能
* UI 実装そのもの
* vector store / embeddings / memory の細かい設計

## 1.7 成功指標

初期段階では以下を成功条件とする。

* user_request → main → worker_run → notification の一連フローが動作する
* runtime-core が LangChain 非依存である
* create_agent / create_deep_agent / compiled graph を adapter 経由で実行できる
* task state の遷移が追跡できる
* retry / timeout / delayed execution が最低限動作する
* artifact service を後付けできる

将来的な指標:

* 実装側が独自 agent を少ないコードで登録できる
* runtime 側の修正なしに複数の agent handler を追加できる
* scheduler や notification channel を差し替えられる

## 1.8 制約

* Python ベース
* LangChain / LangGraph / Deep Agents に寄せる
* 過度な抽象化を避ける
* runtime-core は agent や file を知らない
* 実装は段階的に進める
