from __future__ import annotations

from runtime_core.types import JsonValue


def normalize_text(value: JsonValue) -> str:
    return str(value).strip()
