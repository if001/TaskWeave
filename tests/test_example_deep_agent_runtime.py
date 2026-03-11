import asyncio
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

from runtime_core.models import Task

from examples.deep_agent_runtime.bootstrap import (
    EXAMPLE_TASK_ID,
    TASK_KIND_MAIN_RESEARCH,
    _resolve_real_agent_backend,
    build_example_runtime,
    seed_example_task,
)
from examples.deep_agent_runtime.web_tools import (
    web_list_and_store_artifact,
    web_page_and_store_artifact,
)


class _SearchHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        payload = json.loads(body) if body else {}

        if self.path == "/list":
            response = {
                "query": payload.get("q", ""),
                "k": payload.get("k", 0),
                "results": [
                    {
                        "rank": 1,
                        "title": "sample",
                        "url": "https://example.com",
                        "snippet": "snippet",
                        "published_date": None,
                    }
                ],
            }
        elif self.path == "/page":
            response = {
                "docs": [
                    {
                        "url": payload.get("url", ""),
                        "title": "Example page",
                        "markdown": "# heading\nbody",
                    }
                ]
            }
        else:
            self.send_response(404)
            self.end_headers()
            return

        raw = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, format: str, *args: object) -> None:
        _ = (format, args)


class DeepAgentRuntimeExampleTests(unittest.TestCase):
    def test_example_runtime_completes_seeded_main_task(self) -> None:
        bundle = build_example_runtime()
        seed_example_task(bundle.repository, topic="test topic")

        async def _run() -> None:
            while await bundle.runtime.tick(now_unix=0.0):
                pass

        asyncio.run(_run())

        main_task = bundle.repository.get(EXAMPLE_TASK_ID)
        self.assertIsNotNone(main_task)
        assert main_task is not None
        self.assertEqual(main_task.status, "succeeded")

        self.assertIsNone(bundle.repository.get(f"worker:{EXAMPLE_TASK_ID}:now:1"))

    def test_delayed_worker_runs_after_run_after(self) -> None:
        bundle = build_example_runtime()
        bundle.repository.enqueue(
            Task(
                id="main:delayed:1",
                kind=TASK_KIND_MAIN_RESEARCH,
                payload={
                    "topic": "schedule later",
                    "delayed_jobs": [{"query": "delayed query", "delay_seconds": 10.0}],
                    "periodic_jobs": [],
                },
                metadata={"enqueued_at_unix": 100.0},
            )
        )

        self.assertTrue(asyncio.run(bundle.runtime.tick(now_unix=100.0)))
        delayed_task = bundle.repository.get("worker:main:delayed:1:delayed:1")
        self.assertIsNotNone(delayed_task)
        assert delayed_task is not None
        self.assertEqual(delayed_task.status, "queued")
        self.assertEqual(delayed_task.run_after, 110.0)

        self.assertFalse(asyncio.run(bundle.runtime.tick(now_unix=109.0)))
        self.assertTrue(asyncio.run(bundle.runtime.tick(now_unix=110.0)))
        delayed_done = bundle.repository.get("worker:main:delayed:1:delayed:1")
        self.assertIsNotNone(delayed_done)
        assert delayed_done is not None
        self.assertEqual(delayed_done.status, "succeeded")

    def test_periodic_worker_is_reenqueued_until_repeat_count(self) -> None:
        bundle = build_example_runtime()
        bundle.repository.enqueue(
            Task(
                id="main:periodic:1",
                kind=TASK_KIND_MAIN_RESEARCH,
                payload={
                    "topic": "periodic",
                    "delayed_jobs": [],
                    "periodic_jobs": [
                        {
                            "query": "periodic query",
                            "start_in_seconds": 0.0,
                            "interval_seconds": 60.0,
                            "repeat_count": 3,
                        }
                    ],
                },
                metadata={"enqueued_at_unix": 200.0},
            )
        )

        self.assertTrue(asyncio.run(bundle.runtime.tick(now_unix=200.0)))
        self.assertTrue(asyncio.run(bundle.runtime.tick(now_unix=200.0)))
        second = bundle.repository.get("worker:main:periodic:1:periodic:1:2")
        self.assertIsNotNone(second)
        assert second is not None
        self.assertEqual(second.status, "queued")
        self.assertEqual(second.run_after, 260.0)

        self.assertTrue(asyncio.run(bundle.runtime.tick(now_unix=260.0)))
        third = bundle.repository.get("worker:main:periodic:1:periodic:1:3")
        self.assertIsNotNone(third)
        assert third is not None
        self.assertEqual(third.status, "queued")
        self.assertEqual(third.run_after, 320.0)

        self.assertTrue(asyncio.run(bundle.runtime.tick(now_unix=320.0)))
        final_periodic = bundle.repository.get("worker:main:periodic:1:periodic:1:3")
        self.assertIsNotNone(final_periodic)
        assert final_periodic is not None
        self.assertEqual(final_periodic.status, "succeeded")

    def test_example_runtime_skips_worker_when_not_needed(self) -> None:
        bundle = build_example_runtime()
        seed_example_task(bundle.repository, topic="no worker")

        async def _run() -> None:
            while await bundle.runtime.tick(now_unix=0.0):
                pass

        asyncio.run(_run())

        self.assertIsNone(bundle.repository.get(f"worker:{EXAMPLE_TASK_ID}:now:1"))

    def test_web_tools_store_artifact_files(self) -> None:
        import os

        with TemporaryDirectory() as temp_dir:
            original_dir = os.environ.get("EXAMPLE_WEB_SEARCH_DIR")
            os.environ["EXAMPLE_WEB_SEARCH_DIR"] = temp_dir
            server = ThreadingHTTPServer(("127.0.0.1", 0), _SearchHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base_url = f"http://127.0.0.1:{server.server_address[1]}"

                list_result = web_list_and_store_artifact(
                    query="llm agents", k=3, base_url=base_url
                )
                self.assertEqual(list_result["status"], "ok")
                list_artifact = Path(list_result["artifact_path"])
                self.assertTrue(list_artifact.exists())
                list_stored = json.loads(list_artifact.read_text(encoding="utf-8"))
                self.assertEqual(list_stored["kind"], "web_list")
                self.assertEqual(list_stored["request"]["q"], "llm agents")
                self.assertIn("results", list_stored["response"])

                page_result = web_page_and_store_artifact(
                    url="https://example.com", base_url=base_url
                )
                self.assertEqual(page_result["status"], "ok")
                page_artifact = Path(page_result["artifact_path"])
                self.assertTrue(page_artifact.exists())
                page_stored = json.loads(page_artifact.read_text(encoding="utf-8"))
                self.assertEqual(page_stored["kind"], "web_page")
                self.assertEqual(page_stored["request"]["url"], "https://example.com")
                self.assertIn("docs", page_stored["response"])

                writes: list[str] = []

                def _writer(payload_text: str) -> str:
                    writes.append(payload_text)
                    return "/artifacts/custom_artifact.json"

                custom_result = web_list_and_store_artifact(
                    query="custom writer",
                    k=2,
                    base_url=base_url,
                    artifact_writer=_writer,
                )
                self.assertEqual(custom_result["status"], "ok")
                self.assertEqual(
                    custom_result["artifact_path"], "/artifacts/custom_artifact.json"
                )
                self.assertEqual(len(writes), 1)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=1)
                if original_dir is None:
                    os.environ.pop("EXAMPLE_WEB_SEARCH_DIR", None)
                else:
                    os.environ["EXAMPLE_WEB_SEARCH_DIR"] = original_dir

    def test_real_agent_backend_resolution(self) -> None:
        import os

        original = os.environ.get("EXAMPLE_REAL_AGENT_BACKEND")
        try:
            os.environ["EXAMPLE_REAL_AGENT_BACKEND"] = "deepagent"
            self.assertEqual(_resolve_real_agent_backend(), "deepagent")

            os.environ["EXAMPLE_REAL_AGENT_BACKEND"] = "unknown"
            self.assertEqual(_resolve_real_agent_backend(), "langchain")
        finally:
            if original is None:
                os.environ.pop("EXAMPLE_REAL_AGENT_BACKEND", None)
            else:
                os.environ["EXAMPLE_REAL_AGENT_BACKEND"] = original


if __name__ == "__main__":
    unittest.main()
