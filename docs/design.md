# Design Doc

## 1. 文書の位置づけ

この文書は、TaskWeave の現行実装に基づく設計書である。
TaskWeave は main / worker / notification を task として扱い、非同期実行、遅延実行、定期実行、retry、timeout、cancel を支える runtime ライブラリを提供する。

## 2. 設計方針

### 2.1 runtime-core を主役にする

TaskWeave の独自価値は agent 自体ではなく、以下を deterministic に扱う runtime にある。

- task 化
- queue / lease / state 管理
- delayed / periodic 実行
- retry / timeout / cancellation
- notification の後送
- artifact の再利用前提

### 2.2 agent 実装は既存フレームワークを利用する

agent 本体は LangChain / LangGraph / Deep Agents を使う。
runtime-core はこれらを知らず、integration 層が `TaskHandler` に包む。

### 2.3 runtime-core は TaskHandler だけを知る

core は `Task` / `TaskContext` / `TaskResult` / `TaskHandler` を中心に構成する。
LangGraph の `CompiledStateGraph` や LangChain Runnable は `runtime_langchain` で吸収する。

### 2.4 examples は agent の組み立てに集中する

examples 側の責務は以下に絞る。

- main / worker graph の作成
- tools の作成
- prompt / skills の定義
- artifact 保存・検索の具体実装
- runtime の composition root

## 3. レイヤ構成

### 3.1 runtime_core

責務:

- task enqueue / lease / commit
- task status 管理
- retry / timeout / cancel
- delayed / periodic task の生成
- handler dispatch
- notification payload 整形

知らないもの:

- LangChain
- LangGraph
- Deep Agents
- tool の具体内容
- artifact の保存形式
- memory の保存形式

### 3.2 runtime_langchain

責務:

- LangGraph / LangChain runnable を `RunnableTaskHandler` として実行する
- `TaskContext` を graph input に変換する
- graph output を `TaskResult` に変換する
- worker 起動 tool を提供する
- main / worker 共通の前処理・後処理を持つ
- examples から渡された `before_invoke` / `after_invoke` を既定フローに合成する

### 3.3 examples

責務:

- main / worker graph の具体実装
- artifact tools の提供
- notification sender の具体実装
- LangMem hook の提供
- CLI / Discord など run loop の差し替え

## 4. コアモデル

### 4.1 Task

`Task` は runtime が扱う最小単位で、`kind` と `payload` を持つ。
主な属性:

- `id`
- `kind`
- `payload`
- `status`
- `run_after`
- `parent_task_id`
- `dedupe_key`
- `metadata`

### 4.2 TaskContext

handler 実行時に task と実行文脈を渡す。

- `task`
- `attempt`
- `deadline_unix`
- `cancellation_requested`

### 4.3 TaskResult

handler の結果を表す。

- `status`
- `output`
- `next_tasks`
- `error`

### 4.4 TaskHandler

runtime-core が知る唯一の実行契約は以下である。

```python
class TaskHandler(Protocol):
    async def run(self, ctx: TaskContext) -> TaskResult: ...
```

## 5. 現行タスクフロー

### 5.1 main_research

1. ユーザー入力が `main_research` task として enqueue される
2. runtime が main handler を実行する
3. main graph は即時応答を返す
4. main graph が worker request tool を呼んだ場合のみ `worker_research` task が生成される
5. `TaskResult.next_tasks` により worker / notification が enqueue される

### 5.2 worker_research

1. worker task が ready になる
2. runtime が worker handler を実行する
3. worker は artifact を保存し、必要なら結果通知を `notification` task に変換する
4. periodic worker の場合は次回 task を enqueue する

### 5.3 notification

1. main / worker の `TaskResult` から notification task が生成される
2. runtime は notification handler を呼ぶ
3. 実際の送信先は `NotificationSender` 実装に委譲する

## 6. Runnable 実行フロー

`RunnableTaskHandler.run` の流れは次の通りである。

1. `input_mapper(ctx)`
2. `before_invoke(ctx, input)`
3. `runnable.ainvoke(input, config=...)`
4. `after_invoke(ctx, raw_output)`
5. `output_mapper(ctx, output)`

この構造により、runtime 側の既定フローを保ったまま examples 側から hook を追加できる。

## 7. Main / Worker の前後処理

### 7.1 main の既定前処理

main handler は invoke 前に recorder を drain し、task payload に含まれる delayed / periodic plan を worker request として取り込む。

### 7.2 examples からの追加 hook

examples は `register_main(..., before_invoke=..., after_invoke=...)` を通じて追加処理を注入できる。
既定の main / worker 処理は維持され、受け取った hook は合成される。

## 8. 長期記憶の扱い

現行実装では `/memories/` を filesystem として exposed しない。
長期記憶は LangMem + ReflectionExecutor で扱う。

### 8.1 before_invoke

main の `before_invoke` で LangMem を検索し、以下を prompt に注入する。

- 長期的なユーザープロファイル
- 継続的な関心トピック

### 8.2 after_invoke

main の `after_invoke` で user / assistant の turn を `ReflectionExecutor.submit(...)` に渡し、非同期で profile / topics memory を更新する。

### 8.3 保存先

LangMem の保存先は `langgraph_store.sqlite` を使う。
main agent の store と同系統の永続 store を共有し、memory の正本を filesystem に持たない。

## 9. Artifact の扱い

artifact の raw は filesystem に保存する。
meta は PGVectorStore に保存する。

### 9.1 保存

- `raw.json` を artifact directory に保存
- title / summary / tags を Ollama で生成
- meta を PGVectorStore に保存

### 9.2 検索

- `artifact_search` は PGVectorStore を検索する
- vector 検索と rerank を組み合わせる
- 必要な raw だけ `raw_path` から読む

## 10. 主要な公開境界

### 10.1 runtime_core

公開する中心:

- task モデル
- runtime
- repository
- scheduler
- registry
- notifications

### 10.2 runtime_langchain

公開する中心:

- `RunnableTaskHandler`
- `ResearchRuntimeBuilder`
- worker request tools

内部 helper:

- `research_handlers`
- `task_context_config`
- `task_orchestrator`

## 11. 現行の設計上の判断

- worker は main の tool 呼び出しでのみ起動する
- `needs_worker` のような payload フラグによる即時起動は行わない
- main / worker の business flow は runtime 側が持つ
- examples は graph / tools / prompt / hooks の差し替えに集中する
- notification と artifact の配送責務は runtime-core の外に置く
