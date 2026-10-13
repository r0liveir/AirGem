import unittest
from datetime import date, datetime

from airgem.core import FlightRequest
from airgem.search import SerpApiProvider
from airgem.search.serpapi import _request_url


def request_data(params: dict[str, str]) -> dict:
    if "booking_token" in params:
        return {
            "booking_options": [
                {"together": {"price": 120, "booking_request": {"url": "https://expensive.example"}}},
                {"together": {"price": 90, "booking_request": {"url": "https://cheap.example"}}},
            ]
        }
    return {
        "search_parameters": {"currency": "BRL"},
        "best_flights": [
            {
                "flights": [
                    {
                        "departure_airport": {"id": params["departure_id"], "time": f"{params['outbound_date']} 08:00"},
                        "arrival_airport": {"id": params["arrival_id"], "time": f"{params['outbound_date']} 10:00"},
                    }
                ],
                "price": 100,
                "booking_token": f"token-{params['outbound_date']}",
            }
        ],
    }


class TestSerpApi(unittest.TestCase):
    def test_request_url_merges_existing_query(self) -> None:
        url = _request_url(
            "https://serpapi.com/search.json?engine=google_flights",
            {"departure_id": "GRU", "arrival_id": "POA"},
        )

        self.assertEqual(url.count("?"), 1)
        self.assertIn("engine=google_flights", url)
        self.assertIn("departure_id=GRU", url)
        self.assertIn("arrival_id=POA", url)

    def test_one_way_uses_deep_search_price_sort_and_booking_url(self) -> None:
        calls: list[dict[str, str]] = []
        provider = SerpApiProvider(api_key="test", fetch_json=lambda params: calls.append(params) or request_data(params))
        request = FlightRequest("GRU", "POA", date(2026, 11, 6), sort_by_price=True, currency="BRL")

        result = provider.search(request)

        self.assertEqual(len(result.outbound), 1)
        option = result.outbound[0]
        self.assertEqual(option.booking_url, "https://cheap.example")
        self.assertEqual(option.departure, datetime(2026, 11, 6, 8))
        self.assertEqual(calls[0]["type"], "2")
        self.assertEqual(calls[0]["sort_by"], "2")
        self.assertEqual(calls[0]["deep_search"], "true")
        self.assertNotIn("hl", calls[0])
        self.assertNotIn("gl", calls[0])
        self.assertEqual(calls[1]["departure_id"], "GRU")
        self.assertEqual(calls[1]["arrival_id"], "POA")
        self.assertEqual(calls[1]["outbound_date"], "2026-11-06")
        self.assertEqual(calls[1]["type"], "2")

    def test_round_trip_runs_two_independent_one_way_searches(self) -> None:
        calls: list[dict[str, str]] = []
        provider = SerpApiProvider(api_key="test", fetch_json=lambda params: calls.append(params) or request_data(params))
        request = FlightRequest("GRU", "POA", date(2026, 11, 6), date(2026, 11, 8))

        result = provider.search(request)

        self.assertTrue(result.round_trip)
        self.assertEqual(len(result.outbound), 1)
        self.assertEqual(len(result.inbound), 1)
        self.assertEqual(result.inbound[0].origin, "POA")
        self.assertIn("work in progress", result.notes[0])
        searches = [call for call in calls if "booking_token" not in call]
        self.assertEqual([(call["departure_id"], call["arrival_id"]) for call in searches], [("GRU", "POA"), ("POA", "GRU")])
