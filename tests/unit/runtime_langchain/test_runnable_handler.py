import asyncio
import unittest

from runtime_core.types import JsonValue, Task, TaskContext, TaskResult
from runtime_langchain.runnable_handler import RunnableTaskHandler


class _FakeRunnable:
    def __init__(self) -> None:
        self.last_input: dict[str, JsonValue] | None = None
        self.last_config: dict[str, JsonValue] | None = None

    async def ainvoke(
        self,
        input: dict[str, JsonValue],
        config: dict[str, JsonValue] | None = None,
    ) -> dict[str, JsonValue]:
        self.last_input = input
        self.last_config = config
        return {"echo": input, "config": config}


class RunnableTaskHandlerTests(unittest.TestCase):
    def test_maps_input_config_and_output(self) -> None:
        runnable = _FakeRunnable()
        task = Task(id="t1", kind="demo", payload={"value": "hello"})
        ctx = TaskContext(task=task, attempt=2)

        def _input(c: TaskContext) -> dict[str, JsonValue]:
            return {"text": str(c.task.payload["value"])}

        def _config(c: TaskContext) -> dict[str, JsonValue]:
            return {"attempt": c.attempt}

        def _output(_ctx: TaskContext, raw: dict[str, JsonValue]) -> TaskResult:
            echo = raw["echo"]
            config = raw["config"]
            assert isinstance(echo, dict)
            assert isinstance(config, dict)
            attempt_value = config["attempt"]
            assert isinstance(attempt_value, (int, float, bool, str))
            return TaskResult(
                status="succeeded",
                output={
                    "reply": str(echo["text"]),
                    "attempt": int(attempt_value),
                },
            )

        handler = RunnableTaskHandler(
            runnable=runnable,
            input_mapper=_input,
            config_mapper=_config,
            output_mapper=_output,
        )

        result = asyncio.run(handler.run(ctx))

        self.assertEqual(runnable.last_input, {"text": "hello"})
        self.assertEqual(runnable.last_config, {"attempt": 2})
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.output, {"reply": "hello", "attempt": 2})

    def test_uses_default_none_config(self) -> None:
        runnable = _FakeRunnable()
        ctx = TaskContext(task=Task(id="t2", kind="demo", payload={}), attempt=1)

        def _input(_c: TaskContext) -> dict[str, JsonValue]:
            return {"x": 1}

        def _output(_ctx: TaskContext, _raw: dict[str, JsonValue]) -> TaskResult:
            return TaskResult(status="succeeded")

        handler = RunnableTaskHandler(
            runnable=runnable,
            input_mapper=_input,
            output_mapper=_output,
        )

        asyncio.run(handler.run(ctx))

        self.assertIsNone(runnable.last_config)


if __name__ == "__main__":
    unittest.main()
