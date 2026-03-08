# Design Doc v0.1
## 文書の位置づけ

この文書は、非同期 worker を持つエージェントシステム向けのRuntime ライブラリであるTaskWeaveの 設計書である。
本ドキュメントは完成版ではなく、実装を進めながら更新していくたたき台とする。

# 2. 設計書

## 2.1 設計方針

### 方針 1: runtime-core を主役にする

このシステムの独自価値は、agent の賢さそのものよりも、

* task 化
* 非同期実行
* 定期実行
* retry / timeout
* notification 後送
* artifact の再利用前提

を支える runtime にある。

### 方針 2: agent は既存フレームワークを流用する

agent 実装は以下を想定する。

* LangChain `create_agent()`
* Deep Agents `create_deep_agent()`
* LangGraph compiled graph

runtime-core はこれらを直接理解しない。
代わりに integration 層で包む。

### 方針 3: runtime-core は TaskHandler だけを知る

`RunnableLike` を直接 core の契約にせず、core は `TaskHandler` という最小実行契約を扱う。
LangChain / LangGraph の Runnable は integration 層で `TaskHandler` に変換する。

### 方針 4: artifact / notification は core の外に置く

artifact や notification は task 実行結果として発生する副作用であり、core 自体はそれらのドメイン知識を持たない。

### 方針 5: main / worker の違いは role ではなく task kind と handler で表現する

main agent と worker agent は別クラスにしてもよいが、runtime 側から見ると違いは

* どの task kind を処理するか
* どの handler / adapter で包むか

で表現すべきである。

---

## 2.2 アーキテクチャ概要

```text
User / Trigger / Scheduler
        |
        v
   Task Enqueue
        |
        v
   Runtime Core
   - Queue
   - Scheduler
   - State Machine
   - Lease / Retry / Timeout
   - Dispatch
        |
        +------------------+
        |                  |
        v                  v
 Main TaskHandler     Worker TaskHandler
        |                  |
        v                  v
 LangChain /          Deep Agent /
 LangGraph Agent      LangGraph Graph
        |                  |
        |                  +--> Artifact Service
        |                  +--> Notification Task
        v
 Immediate Response / Next Tasks
```

---

## 2.3 レイヤ分割

### A. runtime-core

責務:

* task enqueue / dequeue / lease
* task state 管理
* retry / timeout / cancellation
* handler dispatch
* handler result の commit

知らないもの:

* LLM
* LangChain
* LangGraph
* Deep Agents
* artifact の中身
* notification の配送先詳細

### B. integration layer

責務:

* LangChain / LangGraph / Deep Agents の runnable を包む
* task payload を agent input へ変換する
* config を注入する
* agent raw output を TaskResult へ変換する

### C. application layer

責務:

* main agent / worker agent の具体実装
* artifact service 実装
* notification service 実装
* scheduler ポリシー
* task payload schema

---

## 2.4 コアインターフェース

### Task

```python
from dataclasses import dataclass, field
from typing import Any, Literal

TaskStatus = Literal[
    "queued",
    "leased",
    "running",
    "succeeded",
    "failed",
    "cancelled",
]

@dataclass
class Task:
    id: str
    kind: str
    payload: dict[str, Any]
    status: TaskStatus = "queued"
    run_after: float | None = None
    parent_task_id: str | None = None
    dedupe_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

### TaskContext

```python
from dataclasses import dataclass

@dataclass
class TaskContext:
    task: Task
    attempt: int
    deadline_unix: float | None = None
    cancellation_requested: bool = False
```

### TaskResult

```python
from dataclasses import dataclass, field
from typing import Any, Literal

ResultStatus = Literal["succeeded", "failed", "retry"]

@dataclass
class TaskResult:
    status: ResultStatus
    output: dict[str, Any] = field(default_factory=dict)
    next_tasks: list[Task] = field(default_factory=list)
    error: str | None = None
```

### TaskHandler

```python
from typing import Protocol

class TaskHandler(Protocol):
    async def run(self, ctx: TaskContext) -> TaskResult: ...
```

この 4 つを runtime-core の公開面の中心にする。

---

## 2.5 Runtime の責務

### Runtime が行うこと

* 次に実行可能な task を取得する
* その task kind に対応する handler を選ぶ
* handler を実行する
* 成功・失敗・再試行を反映する
* `next_tasks` を enqueue する
* task 状態を保存する

### Runtime が行わないこと

* task payload の意味理解
* agent prompt の生成
* artifact の内容解釈
* notification 本文の組み立て
* LLM の直接呼び出し

---

## 2.6 Task 種別の初期案

初期段階では次の task kind を持つ。

### `user_request`

* ユーザーの入力を表す
* main handler が処理する
* 即時応答と追加 task 生成が主目的

### `worker_run`

* 非同期 worker の実行を表す
* worker handler が処理する
* artifact 生成、要約、通知 task 生成が主目的

### `notification`

* ユーザーへ後送する通知を表す
* notification handler が処理する
* CLI / Web / Discord などのチャネルへ配送する

将来的には以下を追加可能:

* `periodic_trigger`
* `artifact_maintenance`
* `memory_refresh`
* `interest_research`

---

## 2.7 状態遷移

### 基本状態

```text
queued -> leased -> running -> succeeded
                        |-> failed
                        |-> cancelled
                        |-> retry(= queued に戻す)
```

### 備考

* `leased` は複数 worker / process を考慮した余地として残す
* 初期実装では単一プロセスでもよい
* retry は backoff を伴って `run_after` を未来に更新する

---

## 2.8 Registry / Dispatch

### HandlerRegistry

```python
class HandlerRegistry:
    def __init__(self):
        self._handlers: dict[str, TaskHandler] = {}

    def register(self, kind: str, handler: TaskHandler) -> None:
        self._handlers[kind] = handler

    def resolve(self, kind: str) -> TaskHandler:
        return self._handlers[kind]
```

runtime は task kind から handler を引いて実行する。

---

## 2.9 LangChain / LangGraph との接続

### なぜ adapter 層を置くか

runtime-core は LangChain / LangGraph 非依存でいたい。
一方、アプリケーションでは `create_agent()` や `create_deep_agent()` をそのまま使いたい。

そのため、integration 層として `RunnableTaskHandler` を置く。

### RunnableTaskHandler のイメージ

```python
class RunnableTaskHandler:
    def __init__(self, runnable, input_mapper, output_mapper, config_mapper=None):
        self.runnable = runnable
        self.input_mapper = input_mapper
        self.output_mapper = output_mapper
        self.config_mapper = config_mapper or (lambda ctx: None)

    async def run(self, ctx: TaskContext) -> TaskResult:
        inp = self.input_mapper(ctx)
        config = self.config_mapper(ctx)
        raw = await self.runnable.ainvoke(inp, config=config)
        return self.output_mapper(ctx, raw)
```

### 役割

* `input_mapper`: task payload → agent input
* `config_mapper`: task context → LangGraph/LangChain config
* `output_mapper`: agent output → TaskResult

この adapter により、core からフレームワーク知識を切り離す。

---

## 2.10 Main / Worker のおすすめ設計

### Main

役割:

* user_request を処理する
* すぐ返せることは返す
* 後ろで処理すべきことは worker_run task にする
* 必要に応じて既存 artifact を参照する

期待する出力:

* 即時返答テキスト
* `worker_run` task の配列

### Worker

役割:

* 独立した目的に対してまとまった処理を行う
* artifact を作る / 読む / 要約する
* 終了時に notification task を生成する

期待する出力:

* artifact reference
* 完了通知用 payload
* 必要なら follow-up worker task

### Subagent と Worker の違い

* Deep Agents の subagent: main の処理中に呼ばれる in-band delegation
* 今回の worker: runtime 管理下で動く out-of-band execution

この違いは文書・命名でも明示する。

---

## 2.11 Artifact 境界

runtime-core は artifact を知らない。
artifact は application layer 側の service として持つ。

### ArtifactService の最小案

```python
from typing import Protocol

class ArtifactService(Protocol):
    def put_text(self, namespace: str, path: str, text: str, metadata: dict) -> str: ...
    def read_text(self, artifact_id: str) -> str: ...
    def list_by_task(self, task_id: str) -> list[str]: ...
```

### Deep Agents との関係

Deep Agents の filesystem は worker 内の作業領域として活用できる。
ただし runtime-core の公開契約に filesystem を持ち込まない。

推奨:

* worker 内部: Deep Agents backend の `/workspace` `/artifacts` `/memories`
* application 外部参照: `artifact_id` ベース

---

## 2.12 Notification 境界

notification も runtime-core は詳細を知らない。
単に `notification` task を処理する handler を登録する。

### NotificationService の最小案

```python
class NotificationService(Protocol):
    async def send(self, payload: dict) -> None: ...
```

### 初期戦略

まずは notification handler 内で service を呼ぶだけにし、配送チャネル差し替えを容易にする。

---

## 2.13 Scheduler 境界

scheduler は runtime-core の一部、または近傍コンポーネントとする。

責務:

* `run_after <= now` の task を実行可能にする
* periodic rule から新しい task を生成する
* retry backoff を扱う

初期実装ではシンプルでよい。

### 初期実装で十分なもの

* 単一プロセス
* in-memory queue または SQLite ベース
* fixed / exponential backoff
* cron は後回しでもよい

---

## 2.14 Persistence の考え方

### runtime-core の保存対象

* task
* task attempt count
* state transition history
* next run time
* last error

### 保存しないもの

* agent の内部 state schema
* artifact の中身
* LLM provider 固有情報

### LangGraph との関係

LangGraph 側で checkpoint / persistence を使う場合でも、それは handler 内部の実行基盤として扱う。
runtime の task persistence とは別物として分ける。

---

## 2.15 エラーハンドリング

### 分類

* 一時的失敗: retry
* 恒久的失敗: failed
* ユーザー中断 / 明示停止: cancelled

### 方針

* handler は例外を投げてもよい
* runtime が分類し、必要なら retry に変換する
* ただし domain 固有の retry 判定は application 層に寄せてもよい

---

## 2.16 ディレクトリ構成案

```text
project/
  runtime_core/
    __init__.py
    models.py          # Task, TaskContext, TaskResult
    runtime.py         # Runtime
    registry.py        # HandlerRegistry
    scheduler.py
    repository.py
    errors.py
  runtime_langchain/
    __init__.py
    runnable_handler.py
    mappers.py
  app/
    agents/
      main_agent.py
      worker_agent.py
    handlers/
      main_handler.py
      worker_handler.py
      notification_handler.py
    services/
      artifact_service.py
      notification_service.py
    bootstrap.py
```

---

## 2.17 初期実装ステップ

### Step 1

runtime-core の最小実装

* Task
* TaskContext
* TaskResult
* HandlerRegistry
* Runtime.tick()
* InMemoryTaskRepository

### Step 2

task 種別の追加

* user_request
* worker_run
* notification

### Step 3

LangChain integration

* RunnableTaskHandler
* create_agent を包んで実行

### Step 4

Deep Agents integration

* worker 用 deep agent を包む
* artifact service を接続

### Step 5

scheduler / retry / timeout

* run_after
* backoff
* cancellation

### Step 6

永続化

* SQLite または Postgres を検討

---

## 2.18 この設計の賛成意見

* runtime をライブラリとして切り出しやすい
* agent 側を既存フレームワークに寄せられる
* task 契約と agent 実行契約を分離できる
* main / worker の増加に耐えやすい
* artifact / notification の責務が明確

## 2.19 懸念点 / 反対意見

### 懸念 1: 抽象化のしすぎ

TaskHandler の上にさらに抽象を重ねると複雑化する。
初期段階では `TaskHandler` と `RunnableTaskHandler` までで止めるべき。

### 懸念 2: output mapping の肥大化

agent の生出力を自由形式にすると mapper が汚くなる。
Pydantic / TypedDict などで agent 出力を構造化する方向が望ましい。

### 懸念 3: runtime-core が薄すぎる可能性

薄くしすぎるとアプリ側に責務が漏れる。
ただし今回はまず薄く始め、必要に応じて共通化する方針を取る。

---

# 3. 実装時のおすすめ

## 3.1 最初からやるべきこと

* task model の固定
* state transition の明確化
* handler registry の明確化
* runtime-core と integration layer の import 依存を分ける
* main / worker / notification の 3 フローをまず通す

## 3.2 後回しでよいこと

* 汎用的な plugin システム
* 分散キュー
* 複雑な cron parser
* artifact metadata の完全設計
* 高度な observability

## 3.3 agent 出力の構造化例

```python
from pydantic import BaseModel, Field

class MainAgentOutput(BaseModel):
    reply_text: str = Field(default="")
    spawn_worker_tasks: list[dict] = Field(default_factory=list)

class WorkerAgentOutput(BaseModel):
    summary_text: str = Field(default="")
    artifact_refs: list[str] = Field(default_factory=list)
    notifications: list[dict] = Field(default_factory=list)
```

これを `output_mapper` に渡すと、runtime 側の処理が安定しやすい。

---

# 4. 今後の更新候補

* task payload schema の詳細化
* repository の DB schema
* scheduler の periodic rule 仕様
* artifact 命名規則 / metadata 設計
* cancellation の扱い
* observability / logging / tracing
* security boundary
* multi-process / distributed 実行

---

# 5. まとめ

このプロジェクトは、agent 自体を作ることよりも、**agent を安全に・継続的に・非同期で動かす runtime を整備すること** を主目的とする。
LangChain / LangGraph / Deep Agents はその上に載る実行エンジンとして扱い、runtime-core はそれらに依存しない task execution library として保つ。

初期段階では、

* runtime-core は `Task / TaskContext / TaskResult / TaskHandler` を中心に設計する
* LangChain / LangGraph / Deep Agents は adapter で包む
* main / worker / notification の 3 種 task フローを通す
* artifact / notification は service 境界の外に置く

という方針で進めるのがもっとも実装しやすく、今後の拡張にも耐えやすい。
