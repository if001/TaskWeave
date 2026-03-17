---
name: use-worker
description: メインエージェントが「自分で調査するか」「worker に依頼するか」を判断し、worker 依頼文を標準化する。
---
# Skill: Research Triage / 依頼生成

## 目的
メインエージェントが「自分で調査するか」「worker に依頼するか」を判断し、worker 依頼文を標準化する。

## 発動条件
- 5分以上かかりそうな調査
- 多数のソース/ドキュメントの精査が必要
- 定期的な追跡や継続監視が必要
- 情報量が多く、単発の web_list / web_page で収束しない

## 手順
1) 目的を1行で定義する（Goal）。
2) 成功条件を3つ以内で列挙する（Success Criteria）。
3) 制約/対象範囲を明示する（Constraints / Scope）。
4) 成果物の形式を明示する（Deliverable Format）。
5) 必須項目（結論・根拠・未解決）を必ず含める。
6) 不足情報があれば「不足時の扱い」を指示する。

## 出力テンプレート（worker依頼文）
目的: ...
成功条件:
- ...
制約/対象範囲:
- ...
成果物の形式:
- ...
必須項目:
- 結論:
- 根拠:
- 未解決:
不足時の扱い:
- ...

## worker起動の指針
- 単発: request_worker(query=...)
- 遅延実行: request_worker_at(query=..., delay_seconds=...)
- 定期実行: request_worker_periodic(query=..., start_in_seconds=..., interval_seconds=..., repeat_count=...)

## 注意
- 依頼文は具体的に、検索語や観点も含める。
- 依頼内容はユーザーの意図を逸脱しない。
