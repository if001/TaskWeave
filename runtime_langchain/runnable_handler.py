from __future__ import annotations

from typing import Any, Callable, Protocol

from runtime_core.models import TaskContext, TaskResult


class AsyncRunnable(Protocol):
    async def ainvoke(self, inp: Any, config: Any = None) -> Any: ...


InputMapper = Callable[[TaskContext], Any]
OutputMapper = Callable[[TaskContext, Any], TaskResult]
ConfigMapper = Callable[[TaskContext], Any]


class RunnableTaskHandler:
    def __init__(
        self,
        runnable: AsyncRunnable,
        input_mapper: InputMapper,
        output_mapper: OutputMapper,
        config_mapper: ConfigMapper | None = None,
    ) -> None:
        self._runnable = runnable
        self._input_mapper = input_mapper
        self._output_mapper = output_mapper
        self._config_mapper = config_mapper or (lambda _: None)

    async def run(self, ctx: TaskContext) -> TaskResult:
        inp = self._input_mapper(ctx)
        config = self._config_mapper(ctx)
        raw = await self._runnable.ainvoke(inp, config=config)
        return self._output_mapper(ctx, raw)
