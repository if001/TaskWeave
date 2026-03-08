# deep_agent_runtime example

TaskWeave の `runtime_core` と `runtime_langchain` を使って、
main agent が必要時に tool 経由で worker task をキュー投入し、runtime が実行を捌くサンプルです。

## 必要環境

- Python 3.11+
- TaskWeave の依存関係
- examples extra（`deepagents==0.4.4`, `langchain>=1.2.0` を含む）
- 実 Deep Agent 実行を有効化する場合は OpenAI API キー

## 実行

```bash
python -m examples.deep_agent_runtime.main
```

実行後はターミナル対話モードに入り、1ユーザーとの継続会話として各ターンの main/worker 実行結果を表示します。

## worker 起動フロー

この example では、main agent (`create_agent`) に以下の worker 起動 tool を渡しています。

- `request_worker_now`: すぐ実行する worker
- `request_worker_at`: 任意時間後に1回実行する worker
- `request_worker_periodic`: 特定の間隔で繰り返し実行する worker

処理責務は以下です。

1. main agent が必要に応じて tool を呼び、worker 実行要求を作る
2. main task の `TaskResult.next_tasks` に worker task が積まれる
3. runtime が queue から ready task を lease して順次実行する
4. periodic worker は worker task 自身が次回 task を enqueue して継続する

## 対話モード

- `you>` で入力した内容が main task (`main_research`) として enqueue されます。
- `research / deep / investigate / 調査 / 深掘り` を含む入力で即時 worker が要求されます。
- `later` を含む入力で遅延1回 worker（10秒後）が追加されます。
- `daily` または `periodic` を含む入力で periodic worker（5秒後開始・60秒間隔・3回）が追加されます。
- `exit` / `quit` / `:q` で終了します。
