import unittest
from datetime import date, datetime

from airgem.core import FlightRequest
from airgem.search import SerpApiProvider
from airgem.search.serpapi import _request_url


def request_data(params: dict[str, str]) -> dict:
    outbound_date = params.get("outbound_date", "2026-11-08")
    departure_id = params.get("departure_id", "POA")
    arrival_id = params.get("arrival_id", "GRU")
    return {
        "search_parameters": {"currency": "BRL"},
        "best_flights": [
            {
                "flights": [
                    {
                        "departure_airport": {"id": departure_id, "time": f"{outbound_date} 08:00"},
                        "arrival_airport": {"id": arrival_id, "time": f"{outbound_date} 10:00"},
                        "airline": "Test Air",
                    }
                ],
                "price": 100,
                "total_duration": 120,
                "departure_token": f"token-{outbound_date}",
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

    def test_one_way_uses_deep_search_and_price_sort_without_booking_lookup(self) -> None:
        calls: list[dict[str, str]] = []
        provider = SerpApiProvider(api_key="test", fetch_json=lambda params: calls.append(params) or request_data(params))
        request = FlightRequest("GRU", "POA", date(2026, 11, 6), sort_by_price=True, currency="BRL")

        result = provider.search(request)

        self.assertEqual(len(result.outbound), 1)
        option = result.outbound[0]
        self.assertEqual(option.departure, datetime(2026, 11, 6, 8))
        self.assertEqual(option.airline, "Test Air")
        self.assertEqual(option.duration_minutes, 120)
        self.assertEqual(option.candidate_id, "candidate-0")
        self.assertEqual(calls[0]["type"], "2")
        self.assertEqual(calls[0]["sort_by"], "2")
        self.assertEqual(calls[0]["deep_search"], "true")
        self.assertNotIn("hl", calls[0])
        self.assertNotIn("gl", calls[0])
        self.assertEqual(len(calls), 1)

    def test_round_trip_uses_native_search(self) -> None:
        calls: list[dict[str, str]] = []
        provider = SerpApiProvider(api_key="test", fetch_json=lambda params: calls.append(params) or request_data(params))
        request = FlightRequest("GRU", "POA", date(2026, 11, 6), date(2026, 11, 8))

        result = provider.search(request)

        self.assertTrue(result.round_trip)
        self.assertEqual(len(result.outbound), 1)
        self.assertEqual(result.inbound, ())
        self.assertIn("Native round-trip", result.notes[0])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["type"], "1")
        self.assertEqual(calls[0]["return_date"], "2026-11-08")
        self.assertEqual(result.outbound[0].candidate_id, "candidate-0")
        self.assertTrue(result.price_is_total)

    def test_expands_only_selected_round_trip_candidate(self) -> None:
        calls: list[dict[str, str]] = []

        def fetch(params: dict[str, str]) -> dict:
            calls.append(params)
            return request_data(params)

        provider = SerpApiProvider(api_key="test", fetch_json=fetch)
        request = FlightRequest("GRU", "POA", date(2026, 11, 6), date(2026, 11, 8))
        search = provider.search(request)

        expanded = provider.expand_round_trip(search.outbound[0].candidate_id or "")

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1]["departure_token"], "token-2026-11-06")
        self.assertEqual(expanded.outbound[0].candidate_id, "candidate-0")
        self.assertEqual(expanded.inbound[0].candidate_id, "candidate-0/return-0")
        self.assertTrue(expanded.price_is_total)

    def test_rejects_unknown_round_trip_candidate(self) -> None:
        provider = SerpApiProvider(api_key="test", fetch_json=lambda params: request_data(params))

        with self.assertRaisesRegex(ValueError, "unknown round-trip candidate"):
            provider.expand_round_trip("candidate-404")
