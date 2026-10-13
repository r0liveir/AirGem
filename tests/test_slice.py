from datetime import date
import unittest

from airgem.core import FlightRequest, parse_flight_request


class FakeProvider:
    def __init__(self, response: str):
        self.response = response
        self.last_user = ""

    def complete(self, *, system: str, user: str) -> str:
        self.last_user = user
        return self.response


class TestFlightParsing(unittest.TestCase):
    def test_parse_valid_flight_request(self) -> None:
        provider = FakeProvider(
            '{"origin":"GRU","destination":"POA","departure_date":"2026-11-06",'
            '"return_date":"2026-11-08","max_price":500,"currency":"brl",'
            '"max_stops":0,"sort_by_price":true}'
        )
        result = parse_flight_request("cheap weekend trip", provider, today=date(2026, 10, 13))

        self.assertEqual(result, FlightRequest("GRU", "POA", date(2026, 11, 6), date(2026, 11, 8), 500, "BRL", 0, True))
        self.assertIn("Current date: 2026-10-13", provider.last_user)
        self.assertEqual(result.missing, [])

    def test_parse_allows_missing_values_without_guessing(self) -> None:
        result = parse_flight_request("cheap flights to Rio", FakeProvider('{"destination":"GIG"}'))

        self.assertEqual(result.destination, "GIG")
        self.assertEqual(result.missing, ["origin", "departure_date"])

    def test_schema_rejects_unknown_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown fields"):
            parse_flight_request("trip", FakeProvider('{"origin":"GRU","surprise":true}'))
