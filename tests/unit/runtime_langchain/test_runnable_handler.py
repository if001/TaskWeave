import asyncio
import unittest

from runtime_core.types import Task, TaskContext, TaskResult
from runtime_langchain.runnable_handler import RunnableTaskHandler


class _FakeRunnable:
    def __init__(self) -> None:
        self.last_input = None
        self.last_config = None

    async def ainvoke(self, inp, config=None):
        self.last_input = inp
        self.last_config = config
        return {"echo": inp, "config": config}


class RunnableTaskHandlerTests(unittest.TestCase):
    def test_maps_input_config_and_output(self) -> None:
        runnable = _FakeRunnable()
        task = Task(id="t1", kind="demo", payload={"value": "hello"})
        ctx = TaskContext(task=task, attempt=2)

        handler = RunnableTaskHandler(
            runnable=runnable,
            input_mapper=lambda c: {"text": c.task.payload["value"]},
            config_mapper=lambda c: {"attempt": c.attempt},
            output_mapper=lambda _ctx, raw: TaskResult(
                status="succeeded",
                output={"reply": raw["echo"]["text"], "attempt": raw["config"]["attempt"]},
            ),
        )

        result = asyncio.run(handler.run(ctx))

        self.assertEqual(runnable.last_input, {"text": "hello"})
        self.assertEqual(runnable.last_config, {"attempt": 2})
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.output, {"reply": "hello", "attempt": 2})

    def test_uses_default_none_config(self) -> None:
        runnable = _FakeRunnable()
        ctx = TaskContext(task=Task(id="t2", kind="demo", payload={}), attempt=1)

        handler = RunnableTaskHandler(
            runnable=runnable,
            input_mapper=lambda _c: {"x": 1},
            output_mapper=lambda _ctx, _raw: TaskResult(status="succeeded"),
        )

        asyncio.run(handler.run(ctx))

        self.assertIsNone(runnable.last_config)


if __name__ == "__main__":
    unittest.main()
