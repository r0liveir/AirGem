import unittest
from datetime import datetime

from airgem.cli import run
from airgem.search import FlightOption, FlightSearchResult


class FakeLLM:
    def complete(self, *, system: str, user: str) -> str:
        return '{"origin":"GRU","destination":"POA","departure_date":"2026-11-06"}'


class FakeSearch:
    def __init__(self) -> None:
        self.request = None

    def search(self, request):
        self.request = request
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
                    airline="Test Air",
                ),
            )
        )


class TestCli(unittest.TestCase):
    def test_run_connects_parse_to_search_and_serializes_results(self) -> None:
        search = FakeSearch()

        result = run("cheap flight", FakeLLM(), search)

        self.assertEqual(search.request.origin, "GRU")
        self.assertEqual(search.request.currency, "USD")
        self.assertEqual(result["request"]["departure_date"], "2026-11-06")
        self.assertEqual(result["outbound"][0]["airline"], "Test Air")
        self.assertNotIn("booking_url", result["outbound"][0])

    def test_run_rejects_incomplete_request_before_search(self) -> None:
        class MissingLLM(FakeLLM):
            def complete(self, *, system: str, user: str) -> str:
                return '{"destination":"POA"}'

        with self.assertRaisesRegex(ValueError, "missing required flight fields"):
            run("cheap flight", MissingLLM(), FakeSearch())
