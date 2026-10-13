import unittest
from datetime import datetime

from airgem.calendar import parse_ics


class TestCalendar(unittest.TestCase):
    def test_parses_timed_all_day_and_ignored_events(self) -> None:
        calendar = parse_ics(
            """BEGIN:VCALENDAR
BEGIN:VEVENT
DTSTART:20261106T090000
DTEND:20261106T100000
END:VEVENT
BEGIN:VEVENT
DTSTART;VALUE=DATE:20261107
DTEND;VALUE=DATE:20261108
END:VEVENT
BEGIN:VEVENT
STATUS:CANCELLED
DTSTART:20261106T110000
DTEND:20261106T120000
END:VEVENT
END:VCALENDAR
"""
        )

        self.assertEqual(len(calendar), 2)
        self.assertEqual(calendar[0].start, datetime(2026, 11, 6, 9))
        self.assertEqual(calendar[0].end, datetime(2026, 11, 6, 10))
        self.assertEqual(calendar[1].start, datetime(2026, 11, 7))
        self.assertEqual(calendar[1].end, datetime(2026, 11, 8))

    def test_expands_recurring_events_in_requested_window(self) -> None:
        calendar = parse_ics(
            """BEGIN:VCALENDAR
BEGIN:VEVENT
DTSTART:20270101T060000
DTEND:20270101T120000
RRULE:FREQ=DAILY;COUNT=20
END:VEVENT
END:VCALENDAR
""",
            start=datetime(2027, 1, 1),
            end=datetime(2027, 1, 9),
        )

        self.assertEqual(len(calendar), 8)
        self.assertEqual(calendar[-1].start, datetime(2027, 1, 8, 6))
