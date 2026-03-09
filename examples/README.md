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


実 Deep Agent 実行時のバックエンド切り替え:

- `EXAMPLE_USE_REAL_DEEP_AGENT=1` で実 LLM 実行を有効化
- `EXAMPLE_REAL_AGENT_BACKEND=langchain` (default) で worker に `create_agent` を使用
- `EXAMPLE_REAL_AGENT_BACKEND=deepagent` で worker に `create_deep_agent` を使用
- `SIMPLE_CLIENT_BASE_URL=<URL>` を設定すると、deepagent worker に web検索ツール (`web_list`, `web_page`) が有効化されます
- deepagent worker は `CompositeBackend` を使い、`/memories/` を `StoreBackend`（long-term memory）、`/artifacts/` を `FilesystemBackend`（ローカル保存）へ route します
- Web検索アーティファクト保存先は `EXAMPLE_DEEPAGENT_ARTIFACT_DIR`（未指定時は一時ディレクトリ配下）です
- `web_list` は `POST <base_url>/list` (`{"q": q, "k": k}`)、`web_page` は `POST <base_url>/page` (`{"url": url}`) を利用します
- deepagent worker の検索/取得生レスポンスは `/artifacts/*` として保存され（実体は `EXAMPLE_DEEPAGENT_ARTIFACT_DIR` 配下）、会話コンテキストへは要約のみ返します（web_tools単体利用時は `EXAMPLE_WEB_SEARCH_DIR` を参照）

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

## Discord 入出力サンプル

`examples/deep_agent_runtime/discord_bot.py` は Discord のメンション入力を `main_research` task として enqueue し、
main agent の結果と worker 完了通知を Discord チャンネルへ返すサンプルです。

### 使い方

1. `discord.py` をインストール
2. Discord Bot Token を環境変数に設定

```bash
export EXAMPLE_DISCORD_BOT_TOKEN="<your-token>"
python -m examples.deep_agent_runtime.discord_bot
```

### 振る舞い

- Bot をメンションすると `main_research` task が作成される
- メンション受信から main 応答を返すまで、Bot は Discord 上で typing 表示を継続する
- main 処理の完了時に `notification` task が生成され、main の結果が Discord に送信される
- worker が起動した場合、worker 完了時にも `notification` task が生成され、完了メッセージが Discord に送信される
