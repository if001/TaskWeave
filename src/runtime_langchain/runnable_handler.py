from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Mapping, Protocol, TypeVar, cast

from runtime_core.models import TaskContext, TaskResult
from runtime_core.agent_types import AgentConfig

InputT = TypeVar("InputT", contravariant=True)
OutputT = TypeVar("OutputT", covariant=True)
ConfigT = TypeVar("ConfigT", contravariant=True)


class AsyncRunnable(Protocol[InputT, OutputT, ConfigT]):
    async def ainvoke(self, inp: InputT, config: ConfigT | None = None) -> OutputT: ...


class CompiledStateGraphLike(Protocol):
    async def ainvoke(
        self,
        input: Any,
        config: "LangChainRunnableConfig | None" = None,
    ) -> object: ...


if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig as LangChainRunnableConfig
else:
    LangChainRunnableConfig = Mapping[str, object]


InputMapper = Callable[[TaskContext], InputT]
OutputMapper = Callable[[TaskContext, OutputT], TaskResult]
ConfigMapper = Callable[[TaskContext], ConfigT | None]
BeforeInvoke = Callable[[TaskContext, InputT], InputT]
AfterInvoke = Callable[[TaskContext, OutputT], OutputT]


def wrap_compiled_state_graph(
    graph: CompiledStateGraphLike,
) -> AsyncRunnable[object, object, AgentConfig]:
    class _Adapter:
        async def ainvoke(
            self, inp: object, config: AgentConfig | None = None
        ) -> object:
            return await graph.ainvoke(
                input=inp,
                config=_to_langchain_config(config),
            )

    return _Adapter()


def _to_langchain_config(
    config: AgentConfig | None,
) -> LangChainRunnableConfig | None:
    if config is None:
        return None
    return cast(LangChainRunnableConfig, dict(config))


class RunnableTaskHandler:
    def __init__(
        self,
        runnable: AsyncRunnable[InputT, OutputT, ConfigT],
        input_mapper: InputMapper[InputT],
        output_mapper: OutputMapper[OutputT],
        config_mapper: ConfigMapper[ConfigT] | None = None,
        before_invoke: BeforeInvoke[InputT] | None = None,
        after_invoke: AfterInvoke[OutputT] | None = None,
    ) -> None:
        self._runnable = runnable
        self._input_mapper = input_mapper
        self._output_mapper = output_mapper
        self._config_mapper = config_mapper or (lambda _: None)
        self._before_invoke = before_invoke or (lambda _, inp: inp)
        self._after_invoke = after_invoke or (lambda _, raw: raw)

    async def run(self, ctx: TaskContext) -> TaskResult:
        inp = self._input_mapper(ctx)
        inp = self._before_invoke(ctx, inp)
        config = self._config_mapper(ctx)
        raw = await self._runnable.ainvoke(inp, config=config)
        raw = self._after_invoke(ctx, raw)
        return self._output_mapper(ctx, raw)
