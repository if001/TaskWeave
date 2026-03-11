import unittest

from runtime_core.infra import UnknownTaskKindError
from runtime_core.runtime import HandlerRegistry
from runtime_core.types import TaskContext, TaskResult


class _NoopHandler:
    async def run(self, ctx: TaskContext) -> TaskResult:
        _ = ctx
        return TaskResult(status="succeeded")


class HandlerRegistryTests(unittest.TestCase):
    def test_resolve_returns_registered_handler(self) -> None:
        registry = HandlerRegistry()
        handler = _NoopHandler()
        registry.register("demo", handler)

        self.assertIs(registry.resolve("demo"), handler)

    def test_resolve_raises_for_unknown_kind(self) -> None:
        registry = HandlerRegistry()

        with self.assertRaises(UnknownTaskKindError):
            registry.resolve("missing")


if __name__ == "__main__":
    unittest.main()
