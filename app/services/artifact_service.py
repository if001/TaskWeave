from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


class ArtifactService(Protocol):
    def put_text(self, namespace: str, path: str, text: str, metadata: dict) -> str: ...

    def read_text(self, artifact_id: str) -> str: ...

    def list_by_task(self, task_id: str) -> list[str]: ...


@dataclass(slots=True)
class InMemoryArtifactService:
    _content_by_id: dict[str, str] = field(default_factory=dict)
    _ids_by_task: dict[str, list[str]] = field(default_factory=dict)

    def put_text(self, namespace: str, path: str, text: str, metadata: dict) -> str:
        artifact_id = f"{namespace}:{path}"
        self._content_by_id[artifact_id] = text

        task_id = str(metadata.get("task_id", ""))
        if task_id:
            self._ids_by_task.setdefault(task_id, []).append(artifact_id)
        return artifact_id

    def read_text(self, artifact_id: str) -> str:
        return self._content_by_id[artifact_id]

    def list_by_task(self, task_id: str) -> list[str]:
        return list(self._ids_by_task.get(task_id, []))
