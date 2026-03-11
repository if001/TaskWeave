from __future__ import annotations

from collections.abc import Awaitable
from typing import Callable, TypeVar

from runtime_core.types import TaskContext, TaskResult

InputT = TypeVar("InputT", contravariant=True)
OutputT = TypeVar("OutputT", covariant=True)
ConfigT = TypeVar("ConfigT", contravariant=True)

InputMapper = Callable[[TaskContext], InputT]
OutputMapper = Callable[[TaskContext, OutputT], TaskResult]
ConfigMapper = Callable[[TaskContext], ConfigT | None]
InvokeKwargsMapper = Callable[[TaskContext], dict[str, object]]
BeforeInvoke = Callable[[TaskContext, InputT], InputT]
AfterInvoke = Callable[[TaskContext, OutputT], OutputT]
AinvokeCallable = Callable[..., Awaitable[OutputT]]


def _default_config_mapper(_: TaskContext) -> None:
    return None


def _default_invoke_kwargs(_: TaskContext) -> dict[str, object]:
    return {}


def _default_before_invoke(_: TaskContext, inp: InputT) -> InputT:
    return inp


def _default_after_invoke(_: TaskContext, raw: OutputT) -> OutputT:
    return raw


class RunnableTaskHandler:
    def __init__(
        self,
        ainvoke: AinvokeCallable[OutputT],
        input_mapper: InputMapper[InputT],
        output_mapper: OutputMapper[OutputT],
        config_mapper: ConfigMapper[ConfigT] | None = None,
        invoke_kwargs_mapper: InvokeKwargsMapper | None = None,
        before_invoke: BeforeInvoke[InputT] | None = None,
        after_invoke: AfterInvoke[OutputT] | None = None,
    ) -> None:
        self._ainvoke = ainvoke
        self._input_mapper = input_mapper
        self._output_mapper = output_mapper
        self._config_mapper = config_mapper or _default_config_mapper
        self._invoke_kwargs_mapper = invoke_kwargs_mapper or _default_invoke_kwargs
        self._before_invoke = before_invoke or _default_before_invoke
        self._after_invoke = after_invoke or _default_after_invoke

    async def run(self, ctx: TaskContext) -> TaskResult:
        inp = self._input_mapper(ctx)
        inp = self._before_invoke(ctx, inp)
        config = self._config_mapper(ctx)
        invoke_kwargs = self._invoke_kwargs_mapper(ctx)
        if invoke_kwargs:
            raw = await self._ainvoke(inp, config=config, **invoke_kwargs)
        else:
            raw = await self._ainvoke(inp, config=config)
        raw = self._after_invoke(ctx, raw)
        return self._output_mapper(ctx, raw)
