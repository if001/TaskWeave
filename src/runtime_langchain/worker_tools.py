from __future__ import annotations

from langchain_core.tools import BaseTool, tool

from runtime_core.logging_utils import get_logger
from runtime_core.worker_recorder import WorkerLaunchRecorder

logger = get_logger(__name__)


def build_worker_request_tools(recorder: WorkerLaunchRecorder) -> list[BaseTool]:
    @tool("request_worker_now")
    def request_worker_now(query: str) -> str:
        """Queue an immediate deep-research worker task.

        Use when the query needs background research right away.
        Args:
            query: Research topic or question to hand off to the worker.
        Returns:
            A status string indicating the request was queued.
        Side effects:
            Records the request in the worker launch recorder.
        """
        logger.info("call worker as tool!!!!!!!!!!")
        return recorder.request_worker_now(query)

    @tool("request_worker_at")
    def request_worker_at(query: str, delay_seconds: float) -> str:
        """Queue a one-time worker task after a delay (seconds).

        Use when the work should start later (e.g., cooldown or scheduled check).
        Args:
            query: Research topic or question for the worker.
            delay_seconds: Seconds from now to start the task (will be clamped to >= 0).
        Returns:
            A status string indicating the delayed request was queued.
        Side effects:
            Records the request in the worker launch recorder.
        """
        return recorder.request_worker_at(query, delay_seconds)

    @tool("request_worker_periodic")
    def request_worker_periodic(
        query: str,
        start_in_seconds: float,
        interval_seconds: float,
        repeat_count: int,
    ) -> str:
        """Queue a periodic worker task with a start delay and repeat interval.

        Use for recurring research (e.g., monitoring a topic).
        Args:
            query: Research topic or question for the worker.
            start_in_seconds: Seconds from now to the first run (clamped to >= 0).
            interval_seconds: Seconds between runs (clamped to >= 1).
            repeat_count: Number of runs (clamped to >= 1).
        Returns:
            A status string indicating the periodic request was queued.
        Side effects:
            Records the request in the worker launch recorder.
        """
        return recorder.request_worker_periodic(
            query, start_in_seconds, interval_seconds, repeat_count
        )

    return [request_worker_now, request_worker_at, request_worker_periodic]
