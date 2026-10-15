import unittest
import json
from datetime import date, datetime

from airgem.core import FlightRequest
from airgem.search import SerpApiProvider
from airgem.search.serpapi import _request_url


def request_data(params: dict[str, str]) -> dict:
    if "booking_token" in params or "selected_flights_json" in params:
        return {
            "booking_options": [
                {
                    "together": {
                        "book_with": "Test Air",
                        "booking_request": {
                            "url": "https://booking.example/flight",
                            "post_data": "flight=TA+100&currency=BRL",
                        },
                    }
                }
            ]
        }
    outbound_date = "2026-11-08" if "departure_token" in params else params["outbound_date"]
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
                        "flight_number": "TA 100",
                    }
                ],
                "price": 100,
                "total_duration": 120,
                "departure_token": f"token-{outbound_date}",
                "booking_token": f"booking-{outbound_date}",
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
        self.assertEqual(option.segments[0].flight_number, "TA 100")
        self.assertEqual(calls[0]["type"], "2")
        self.assertEqual(calls[0]["sort_by"], "2")
        self.assertEqual(calls[0]["deep_search"], "true")
        self.assertNotIn("hl", calls[0])
        self.assertNotIn("gl", calls[0])
        self.assertEqual(len(calls), 1)

    def test_selected_one_way_flights_request_booking_options_as_one_way(self) -> None:
        calls: list[dict[str, str]] = []
        provider = SerpApiProvider(api_key="test", fetch_json=lambda params: calls.append(params) or request_data(params))

        action = provider.resolve_booking_action(
            ("candidate-0",),
            selected_flights_json=(
                '{"outbound":[{"flight_number":"TA100","departure_id":"GRU",'
                '"arrival_id":"POA","date":"2026-11-06"}]}'
            ),
        )

        self.assertIsNotNone(action)
        self.assertEqual(calls[0]["type"], "2")

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

    def test_resolves_direct_booking_url_for_selected_candidate(self) -> None:
        calls: list[dict[str, str]] = []
        provider = SerpApiProvider(
            api_key="test",
            fetch_json=lambda params: calls.append(params) or request_data(params),
        )
        request = FlightRequest("GRU", "POA", date(2026, 11, 6), sort_by_price=True)

        result = provider.search(request)
        url = provider.resolve_booking_url((result.outbound[0].candidate_id or "",))

        self.assertEqual(url, "https://booking.example/flight")
        self.assertEqual(calls[1]["booking_token"], "booking-2026-11-06")

    def test_resolves_booking_action_with_provider_post_data(self) -> None:
        provider = SerpApiProvider(api_key="test", fetch_json=lambda params: request_data(params))
        request = FlightRequest("GRU", "POA", date(2026, 11, 6), sort_by_price=True)

        result = provider.search(request)
        action = provider.resolve_booking_action((result.outbound[0].candidate_id or "",))

        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action.url, "https://booking.example/flight")
        self.assertEqual(action.post_data, "flight=TA+100&currency=BRL")

    def test_resolves_round_trip_booking_url_from_selected_flights_json(self) -> None:
        calls: list[dict[str, str]] = []
        provider = SerpApiProvider(
            api_key="test",
            fetch_json=lambda params: calls.append(params) or request_data(params),
        )
        selected = {
            "outbound": [
                {
                    "flight_number": "TA 100",
                    "departure_id": "GRU",
                    "arrival_id": "POA",
                    "date": "2026-11-06",
                }
            ],
            "return": [
                {
                    "flight_number": "TA 101",
                    "departure_id": "POA",
                    "arrival_id": "GRU",
                    "date": "2026-11-08",
                }
            ],
        }

        url = provider.resolve_booking_url(
            (),
            selected_flights_json=json.dumps(selected, separators=(",", ":")),
        )

        self.assertEqual(url, "https://booking.example/flight")
        self.assertEqual(calls[0]["selected_flights_json"], json.dumps(selected, separators=(",", ":")))

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
        self.assertEqual(calls[1]["departure_id"], "GRU")
        self.assertEqual(calls[1]["arrival_id"], "POA")
        self.assertEqual(calls[1]["outbound_date"], "2026-11-06")
        self.assertEqual(calls[1]["return_date"], "2026-11-08")
        self.assertEqual(expanded.outbound[0].candidate_id, "candidate-0")
        self.assertEqual(expanded.inbound[0].candidate_id, "candidate-0/return-0")
        self.assertTrue(expanded.price_is_total)

    def test_rejects_unknown_round_trip_candidate(self) -> None:
        provider = SerpApiProvider(api_key="test", fetch_json=lambda params: request_data(params))

        with self.assertRaisesRegex(ValueError, "unknown round-trip candidate"):
            provider.expand_round_trip("candidate-404")
