from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from icalendar import Calendar
import recurring_ical_events


@dataclass(frozen=True)
class BusyInterval:
    start: datetime
    end: datetime


def load_ics(
    path: str | Path,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> tuple[BusyInterval, ...]:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"could not read calendar file: {path}") from error
    return parse_ics(text, start=start, end=end)


def parse_ics(
    text: str,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> tuple[BusyInterval, ...]:
    try:
        calendar = Calendar.from_ical(text)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid iCalendar data") from error

    if (start is None) != (end is None):
        raise ValueError("calendar recurrence range requires both start and end")
    events = (
        recurring_ical_events.of(calendar).between(start, end)
        if start is not None and end is not None
        else calendar.walk("VEVENT")
    )

    intervals: list[BusyInterval] = []
    for event in events:
        interval = _event_interval(event)
        if interval is not None:
            intervals.append(interval)
    return tuple(intervals)


def _event_interval(event: Any) -> BusyInterval | None:
    status = str(event.get("STATUS", "")).upper()
    transparency = str(event.get("TRANSP", "")).upper()
    if status == "CANCELLED" or transparency == "TRANSPARENT":
        return None

    start_property = event.get("DTSTART")
    if start_property is None:
        return None

    raw_start = start_property.dt
    start = _as_datetime(raw_start)
    end_property = event.get("DTEND")
    end = _as_datetime(end_property.dt) if end_property is not None else None
    if end is None:
        duration_property = event.get("DURATION")
        if duration_property is not None:
            end = start + duration_property.dt
        elif isinstance(raw_start, date) and not isinstance(raw_start, datetime):
            end = start + timedelta(days=1)
        else:
            return None
    return BusyInterval(start, end) if end > start else None


def _as_datetime(value: date | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    raise ValueError(f"unsupported iCalendar date value: {value!r}")
