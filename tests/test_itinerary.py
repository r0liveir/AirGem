import unittest
from dataclasses import replace
from datetime import date, datetime

from airgem.itinerary import build_itineraries
from airgem.search import FlightOption, FlightSearchResult


def option(
    origin: str,
    destination: str,
    departure_hour: int,
    price: float,
    *,
    stops: int = 0,
    duration_minutes: int = 120,
) -> FlightOption:
    departure = datetime(2026, 11, 6, departure_hour)
    return FlightOption(
        provider="test",
        origin=origin,
        destination=destination,
        departure=departure,
        arrival=departure.replace(hour=departure_hour + 2),
        price=price,
        currency="USD",
        stops=stops,
        airline="Test Air",
        duration_minutes=duration_minutes,
    )


class TestItinerary(unittest.TestCase):
    def test_round_trip_pairs_legs_and_sums_metrics(self) -> None:
        result = FlightSearchResult(
            outbound=(option("GRU", "POA", 8, 100, stops=1),),
            inbound=(option("POA", "GRU", 18, 80),),
            round_trip=True,
        )

        itineraries = build_itineraries(result)

        self.assertEqual(len(itineraries), 1)
        self.assertEqual(itineraries[0].id, "round-trip-0-0")
        self.assertEqual(itineraries[0].price, 180)
        self.assertEqual(itineraries[0].duration_minutes, 240)
        self.assertEqual(itineraries[0].stops, 1)
        self.assertTrue(itineraries[0].estimated)

    def test_sorts_and_applies_total_price_limit(self) -> None:
        result = FlightSearchResult(
            outbound=(
                option("GRU", "POA", 8, 120),
                option("GRU", "POA", 10, 100),
            )
        )

        itineraries = build_itineraries(result, max_price=110)

        self.assertEqual([item.price for item in itineraries], [100])
        self.assertEqual(itineraries[0].id, "one-way-1")

    def test_drops_return_leg_before_outbound_arrival(self) -> None:
        result = FlightSearchResult(
            outbound=(option("GRU", "POA", 18, 100),),
            inbound=(option("POA", "GRU", 10, 80),),
            round_trip=True,
        )

        self.assertEqual(build_itineraries(result), ())

    def test_native_round_trip_keeps_price_and_marks_return_pending(self) -> None:
        result = FlightSearchResult(
            outbound=(option("GRU", "POA", 8, 180),),
            round_trip=True,
        )

        itinerary = build_itineraries(result)[0]

        self.assertEqual(itinerary.price, 180)
        self.assertIsNone(itinerary.duration_minutes)
        self.assertTrue(itinerary.return_pending)

    def test_expanded_native_round_trip_keeps_total_price(self) -> None:
        result = FlightSearchResult(
            outbound=(
                replace(option("GRU", "POA", 8, 180), candidate_id="candidate-0"),
            ),
            inbound=(
                replace(option("POA", "GRU", 18, 0), candidate_id="candidate-0/return-0"),
            ),
            round_trip=True,
            price_is_total=True,
        )

        itinerary = build_itineraries(result)[0]

        self.assertEqual(itinerary.price, 180)
        self.assertEqual(itinerary.id, "candidate-0/return-0")
        self.assertFalse(itinerary.estimated)
        self.assertFalse(itinerary.return_pending)

    def test_applies_small_hard_constraints(self) -> None:
        result = FlightSearchResult(
            outbound=(
                option("GRU", "POA", 8, 100, stops=0),
                option("GRU", "POA", 10, 90, stops=1),
            )
        )

        itineraries = build_itineraries(
            result,
            max_stops=0,
            departure_date=date(2026, 11, 6),
        )

        self.assertEqual([item.price for item in itineraries], [100])

    def test_filters_wrong_calendar_date_for_both_legs(self) -> None:
        result = FlightSearchResult(
            outbound=(option("GRU", "POA", 8, 100),),
            inbound=(option("POA", "GRU", 18, 80),),
            round_trip=True,
        )

        self.assertEqual(
            build_itineraries(
                result,
                departure_date=date(2026, 11, 7),
                return_date=date(2026, 11, 8),
            ),
            (),
        )
