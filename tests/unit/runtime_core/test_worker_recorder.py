import unittest

from runtime_core.tasks import WorkerLaunchRecorder, collect_worker_requests
from runtime_core.types import MainAgentInput


class WorkerRecorderTests(unittest.TestCase):
    def test_collect_worker_requests_tracks_and_drains(self) -> None:
        recorder = WorkerLaunchRecorder()
        request: MainAgentInput = {
            "topic": "demo",
            "delayed_jobs": [{"query": "later", "delay_seconds": 5.0}],
            "periodic_jobs": [
                {
                    "query": "repeat",
                    "start_in_seconds": 0.0,
                    "interval_seconds": 10.0,
                    "repeat_count": 2,
                }
            ],
        }

        result = collect_worker_requests(recorder, request)

        self.assertEqual(result["delayed_queries"][0]["query"], "later")
        self.assertEqual(result["periodic_queries"][0]["query"], "repeat")
        self.assertEqual(recorder.immediate_queries, [])
        self.assertEqual(recorder.delayed_queries, [])
        self.assertEqual(recorder.periodic_queries, [])

    def test_request_worker_now_ignores_empty_query(self) -> None:
        recorder = WorkerLaunchRecorder()
        recorder.request_worker_now("   ")

        self.assertEqual(recorder.immediate_queries, [])


if __name__ == "__main__":
    unittest.main()
