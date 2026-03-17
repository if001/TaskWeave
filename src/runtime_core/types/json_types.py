from __future__ import annotations

from typing import TypeAlias

JsonPrimitive: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]


def ensure_json_value(value: object) -> JsonValue | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        items_list: list[JsonValue] = []
        for item in value:
            coerced = ensure_json_value(item)
            if coerced is None:
                return None
            items_list.append(coerced)
        return items_list
    if isinstance(value, dict):
        items_dict: dict[str, JsonValue] = {}
        for key, raw_value in value.items():
            if not isinstance(key, str):
                return None
            coerced = ensure_json_value(raw_value)
            if coerced is None:
                return None
            items_dict[key] = coerced
        return items_dict
    return None
