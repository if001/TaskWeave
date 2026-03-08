from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class InMemoryNotificationService:
    sent_messages: list[str] = field(default_factory=list)

    async def send(self, payload: dict) -> None:
        self.sent_messages.append(str(payload.get("message", "")))
