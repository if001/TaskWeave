## コード品質基準
- 重複を徹底的に排除する
- 命名と構造を通じて意図を明確に表現する
- 依存関係を明示的にする
- メソッドを小さくし、単一の責任に集中させる
- 状態と副作用を最小限に抑える
- 可能な限り最もシンプルな解決策を使用する
- Anyは使わず、ライブラリに存在する型やプリミティブ型を利用する。
- objectに関しても最終的なjsonやpayloadを示す型としては利用しても良いが、できる限りライブラリに存在する型を利用する。
- Protcolsを継承しinterfaceを作成する。実装は、Protcolsを継承したクラスを更に継承する形で実装すること。

### 関数化方針
### 関数化の価値があるケース
- 処理がまとまっていて、名前を付けると意図が明確になる
- 似た処理が複数箇所で使われる（再利用性）
- テスト・差し替え・変更の影響範囲を小さくできる
- 副作用や責務を分離できる

### 価値が薄いケース
- 1回しか使わない
- 単なる1〜2行のラッピングで意味が増えない
- 逆に読みの流れが途切れるだけ

### import/export
- __init__.pyには外部公開用の機能のみを記述
- package内のimportには相対importを用いる

## test
- `uv run pytest`で実行
testsディレクトリ以下に、unit, integrationなどのディレクトリを作成し、packageごとにテストを作成する。
例:
- project_root/tests/unit/{package_name}/test_xxx.py
- project_root/tests/integration/{package_name}/test_xxx.py

### 方針
- ユニットテストは、そのモジュール自身の責務だけをテストする
- インテグレーションテストは、境界面だけをテストする。ユニットテストでテスト済のものはテストしてはいけない。
- モックは境界に限定し、内部実装を過剰に固定しない
- 高レベルテストは数を絞り、重要フローだけにする
- 共通のセットアップはできるだけ少なく保ち、増やしすぎない

## logger
- richパッケージとloggerを使いログを標準出力に表示

## packages
`uv sync --extra examples`

## 実行
`uv run -m examples.main`

`uv run -m examples.discord_bot`

## check
修正後は以下で確認すること
`ruff check .`

`pyright .`
