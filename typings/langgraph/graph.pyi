from __future__ import annotations

from typing import Callable, Protocol, TypeVar

from langchain_core.runnables import RunnableConfig


InputT = TypeVar("InputT", contravariant=True)
OutputT = TypeVar("OutputT", covariant=True)
StateT = TypeVar("StateT")


class CompiledStateGraph(Protocol[InputT, OutputT]):
    async def ainvoke(
        self, input: InputT, config: RunnableConfig | None = None
    ) -> OutputT: ...

    @classmethod
    def __class_getitem__(
        cls, item: object
    ) -> type["CompiledStateGraph[InputT, OutputT]"]: ...


class StateGraph:
    def __init__(self, state_type: type[StateT]) -> None: ...
    def add_node(self, name: str, action: Callable[[StateT], StateT]) -> None: ...
    def set_entry_point(self, name: str) -> None: ...
    def add_edge(self, start: str, end: str) -> None: ...
    def compile(self) -> CompiledStateGraph[StateT, StateT]: ...

    @classmethod
    def __class_getitem__(cls, item: object) -> type["StateGraph"]: ...


END: str
