# deep_agent_runtime example

TaskWeave の `runtime_core` と `runtime_langchain` を使って、Deep Agent 風ワークロードを task として実行するサンプルです。

## 必要環境

- Python 3.11+
- TaskWeave の依存関係
- examples extra（`deepagents==0.4.4`, `langchain>=1.2.0` を含む）
- 実 Deep Agent 実行を有効化する場合は OpenAI API キー

## インストール

```bash
uv sync --extra examples
```

または:

```bash
pip install -e '.[examples]'
```

## 環境変数

- `EXAMPLE_USE_REAL_DEEP_AGENT`
  - `0` (default): モック runnable を利用
  - `1`: LangChain agent を利用
- `EXAMPLE_MODEL`
  - default: `gpt-4o-mini`
- `OPENAI_API_KEY`
  - `EXAMPLE_USE_REAL_DEEP_AGENT=1` の場合に必要

## 実行

```bash
python -m examples.deep_agent_runtime.main
```

実行後、`example:deep:1` の状態と task transition 履歴を表示します。

## 補足

この example は `app/bootstrap.py` を変更せず、サンプル専用の `examples/deep_agent_runtime/bootstrap.py` で runtime を構築しています。
