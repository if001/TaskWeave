# Agent Runtime PRD

## 1. 文書の位置づけ

この文書は、TaskWeave の現行実装に基づく PRD である。
TaskWeave は、main / worker を持つ agent システム向けに、task queue・scheduler・async execution・notification を提供する runtime ライブラリである。

## 2. 背景

一般的な agent フレームワークは、ユーザー入力に対するその場の応答を得意とする。
一方で、実アプリケーションでは以下が必要になる。

- ユーザー入力を task として扱う
- main が必要に応じて worker を起動する
- worker を裏で非同期実行する
- 定期実行や遅延実行を扱う
- 完了時に notification を後送する
- artifact を保存し再利用する
- retry / timeout / cancel を deterministic に扱う

TaskWeave はこの runtime 部分をライブラリとして切り出す。

## 3. プロダクトの目的

### 3.1 目的

LangChain / LangGraph / Deep Agents を後段に差し込める、軽量で責務分離された task runtime を提供する。

### 3.2 目指すもの

- main / worker 構成を支える共通 runtime
- queue / retry / timeout / cancel / scheduling を持つ execution 基盤
- LangGraph / Deep Agents を自然に組み込める integration
- runtime-core と examples の責務分離
- CLI / Discord など複数 entrypoint で再利用できる構成

### 3.3 目指さないもの

- 独自 agent framework を一から作ること
- runtime-core が agent の内部構造を理解すること
- memory / artifact / vector store の完全標準化
- UI 実装そのもの

## 4. 現行ユースケース

### 4.1 main の即時応答

1. ユーザー入力を `main_research` task として enqueue する
2. runtime が main handler を実行する
3. main graph が即時応答を返す

### 4.2 background worker

1. main graph が worker request tool を呼ぶ
2. `worker_research` task が生成される
3. worker が background で実行される
4. worker 完了後に notification task が生成される

### 4.3 delayed / periodic 実行

1. main graph が delayed / periodic worker request tool を呼ぶ
2. runtime が `run_after` と periodic payload に従って task を管理する
3. 条件を満たすと worker が実行される

### 4.4 artifact の再利用

1. worker が raw artifact を filesystem に保存する
2. meta を PGVectorStore に保存する
3. main / worker が `artifact_search` で meta を検索し、必要な raw だけ読む

### 4.5 長期記憶の利用

1. main 実行前に LangMem を検索する
2. 検索結果を prompt に注入する
3. main 実行後に ReflectionExecutor へ submit する
4. profile / topics memory を非同期更新する

## 5. In Scope

- task queue
- handler dispatch
- retry / timeout / cancellation
- delayed / periodic task
- main / worker / notification フロー
- LangGraph / LangChain / Deep Agents integration
- worker request tools
- artifact の保存と検索の接続点
- notification sender の接続点
- LangMem hook を examples から差し込める仕組み

## 6. Out of Scope

- 独自 LLM client
- 独自 planner / workflow engine
- distributed queue の本格運用
- vector store の抽象標準化
- memory の cross-backend abstraction
- UI 実装

## 7. 現行アーキテクチャ

### 7.1 runtime_core

- task モデル
- runtime loop
- repository
- scheduler
- registry
- notifications

### 7.2 runtime_langchain

- RunnableTaskHandler
- research handler
- worker request tools
- runtime builder
- TaskContext 由来の config helper

### 7.3 examples

- main / worker graph
- tools
- LangMem memory hook
- artifact tools
- CLI / Discord bootstrap

## 8. 現行の重要仕様

### 8.1 worker の起動条件

worker は main が tool を呼んだときだけ起動する。
`needs_worker=True` のような payload フラグでは起動しない。

### 8.2 Runnable hook

examples は `before_invoke` / `after_invoke` を追加できる。
既定の main / worker フローは維持され、hook は上書きではなく合成される。

### 8.3 長期記憶

- `/memories/` は filesystem として扱わない
- LangMem が検索と保存を担う
- `before_invoke` で検索、`after_invoke` で保存する

### 8.4 artifact

- raw は filesystem
- meta は PGVectorStore
- 検索は `artifact_search` 経由

### 8.5 notification

- main / worker の `TaskResult` から notification task を生成する
- 配送先は `NotificationSender` 実装へ委譲する

## 9. 成功条件

現行段階での成功条件は次の通り。

- `main_research -> worker_research -> notification` フローが通る
- runtime-core が LangChain 非依存である
- LangGraph / Deep Agents を adapter 経由で実行できる
- delayed / periodic / retry / timeout / cancel が機能する
- artifact を後から再利用できる
- LangMem の検索と保存を main 実行前後に差し込める

## 10. 今後の拡張余地

- NotificationSender のチャネル追加
- RetryPolicy / TaskScheduler の差し替え
- worker graph の種類追加
- tracing / metrics の整理
- artifact / memory 運用の改善
