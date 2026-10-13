import unittest
from datetime import datetime

from airgem.application import continue_session, start_session
from airgem.search import FlightOption, FlightSearchResult


class FakeLLM:
    def complete(self, *, system: str, user: str) -> str:
        return '{"origin":"GRU","destination":"POA","departure_date":"2026-11-06"}'


class FakeSearch:
    def search(self, request):
        return FlightSearchResult(
            outbound=(
                FlightOption(
                    provider="fake",
                    origin="GRU",
                    destination="POA",
                    departure=datetime(2026, 11, 6, 8),
                    arrival=datetime(2026, 11, 6, 10),
                    price=100,
                    currency="USD",
                    stops=0,
                ),
            )
        )


class FollowUpLLM:
    def complete(self, *, system: str, user: str) -> str:
        if "Current date:" in user:
            return '{"origin":"GRU","destination":"POA","departure_date":"2026-11-06"}'
        return '{"action":"select_candidate","candidate_id":"one-way-0"}'


class FollowUpSearch(FakeSearch):
    def search(self, request):
        result = super().search(request)
        first = result.outbound[0]
        second = first.__class__(
            provider="fake",
            origin="GRU",
            destination="POA",
            departure=datetime(2026, 11, 6, 10),
            arrival=datetime(2026, 11, 6, 11),
            price=120,
            currency="USD",
            stops=1,
            airline="Test Air",
        )
        return FlightSearchResult(outbound=(first, second))


class TestApplication(unittest.TestCase):
    def test_start_session_reuses_shared_application_path(self) -> None:
        session, payload = start_session("cheap flight", FakeLLM(), FakeSearch())

        self.assertEqual(session.session.request.origin, "GRU")
        self.assertEqual(payload["selected"]["price"], 100)
        self.assertEqual(payload["alternatives"], [])
        self.assertEqual(payload["rationale"]["selected_id"], "one-way-0")
        self.assertIn("Selected one-way-0", payload["agent_message"])

    def test_chat_messages_and_candidate_action_survive_follow_up(self) -> None:
        session, payload = start_session("cheap flight", FollowUpLLM(), FollowUpSearch())

        self.assertEqual([message["role"] for message in payload["messages"]], ["user", "assistant"])
        self.assertEqual(payload["available_actions"], [{"label": "Select one-way-1", "prompt": "Select one-way-1"}])

        updated = continue_session(session, "select the other option")

        self.assertEqual(updated["selected"]["id"], "one-way-1")
        self.assertEqual(len(updated["messages"]), 4)
        self.assertEqual(updated["messages"][-2]["content"], "select the other option")
        self.assertEqual(updated["available_actions"], [{"label": "Select one-way-0", "prompt": "Select one-way-0"}])

    def test_calendar_text_is_parsed_for_web_uploads(self) -> None:
        calendar = """BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
DTSTART:20261106T070000
DTEND:20261106T080000
SUMMARY:Breakfast
END:VEVENT
END:VCALENDAR
"""

        _, payload = start_session(
            "cheap flight",
            FakeLLM(),
            FakeSearch(),
            calendar_text=calendar,
            buffer_minutes=45,
        )

        self.assertIn("Applied 1 calendar event(s) with a 45-minute travel buffer", payload["notes"][-1])
