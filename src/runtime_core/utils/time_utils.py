from __future__ import annotations

import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo


def get_timezone() -> ZoneInfo:
    tz_name = os.environ.get("TRIKERNEL_TIMEZONE", "Asia/Tokyo")
    try:
        return ZoneInfo(tz_name)
    except Exception:
        return ZoneInfo("Asia/Tokyo")


def now_iso() -> str:
    return datetime.now(get_timezone()).isoformat()


def to_timezone(value: datetime) -> datetime:
    tz = get_timezone()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(tz)
