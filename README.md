# README
`uv pip install -e .`

`uv sync --extra examples`


`ruff check .`

`pyright .`


## trace
https://us.cloud.langfuse.com/project/cmmpuv8g0040dad07ihyem6vp/traces?peek=a998fe8525ebaa600a819b80a3c5cacd&timestamp=2026-03-14T06%3A04%3A12.008Z

## Runtime クラス関係図（`src/runtime_core/runtime`, `src/runtime_langchain`）

```mermaid
classDiagram
    class Runtime {
      -TaskRepository _repository
      -HandlerRegistry _registry
      -RetryPolicy _retry_policy
      -TaskScheduler _scheduler
      -WorkerLaunchRecorder _recorder
      +tick(now_unix)
      +execute_task(task, now_unix)
    }
    class HandlerRegistry {
      -dict~str, TaskHandler~ _handlers
      +register(kind, handler)
      +resolve(kind)
    }
    class TaskHandler {
      <<Protocol>>
      +run(ctx) TaskResult
    }
    class TaskRepository {
      <<Protocol>>
      +enqueue(task)
      +lease_next_ready(now_unix)
      +mark_status(task_id, to_status, reason)
    }
    class _TaskRepositoryBase
    class InMemoryTaskRepository
    class FileTaskRepository
    class TransitionPolicy {
      <<Protocol>>
      +validate(from_status, to_status)
    }
    class DefaultTransitionPolicy
    class RetryPolicy
    class PeriodicRule
    class TaskScheduler {
      +next_retry_time(now_unix, attempt, retry_policy)
      +generate_periodic_tasks(now_unix, rules)
    }
    class RuntimeRunner {
      -Runtime _runtime
      -RunnerPolicy _policy
      +run_once()
      +run_forever()
    }
    class ResearchRuntimeBuilder {
      -Runtime _runtime
      -TaskOrchestrator _orchestrator
      +register_main(registry, kind, runnable, ...)
      +register_worker(registry, kind, runnable, ...)
      +register_notification(registry, kind, sender)
    }
    class TaskOrchestrator {
      -TaskResultConfig _config
      -WorkerLaunchRecorder _recorder
      +worker_request_tools()
      +build_main_result(ctx, raw)
      +build_worker_result(ctx, raw)
    }
    class RunnableTaskHandler {
      -ainvoke
      -input_mapper
      -output_mapper
      +run(ctx) TaskResult
    }
    class ResearchTaskHandler
    class NotificationTaskHandler

    Runtime --> TaskRepository : uses
    Runtime --> HandlerRegistry : uses
    Runtime --> RetryPolicy : uses
    Runtime --> TaskScheduler : uses
    Runtime --> PeriodicRule : uses list

    HandlerRegistry o--> TaskHandler : registers

    _TaskRepositoryBase ..|> TaskRepository
    InMemoryTaskRepository --|> _TaskRepositoryBase
    FileTaskRepository --|> _TaskRepositoryBase
    DefaultTransitionPolicy ..|> TransitionPolicy
    _TaskRepositoryBase --> TransitionPolicy : validates

    RuntimeRunner --> Runtime : executes
    RuntimeRunner --> TaskScheduler : periodic enqueue

    ResearchRuntimeBuilder --> Runtime : owns
    ResearchRuntimeBuilder --> TaskOrchestrator : owns
    ResearchRuntimeBuilder --> HandlerRegistry : registers to
    ResearchRuntimeBuilder --> RunnableTaskHandler : builds
    ResearchRuntimeBuilder --> NotificationTaskHandler : registers

    ResearchTaskHandler --|> RunnableTaskHandler
    ResearchTaskHandler --> TaskOrchestrator : uses
    TaskOrchestrator --> WorkerLaunchRecorder : uses
```
