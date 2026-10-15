from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol

from airgem.core import FlightRequest


@dataclass(frozen=True)
class FlightOption:
    provider: str
    origin: str
    destination: str
    departure: datetime
    arrival: datetime
    price: float
    currency: str
    stops: int
    airline: str | None = None
    duration_minutes: int | None = None
    candidate_id: str | None = None
    segments: tuple["FlightSegment", ...] = ()


@dataclass(frozen=True)
class FlightSegment:
    flight_number: str
    origin: str
    destination: str
    departure: datetime
    arrival: datetime


@dataclass(frozen=True)
class BookingAction:
    url: str
    post_data: str | None = None


@dataclass(frozen=True)
class FlightSearchResult:
    """Search candidates plus optional return-leg candidates."""

    outbound: tuple[FlightOption, ...]
    inbound: tuple[FlightOption, ...] = ()
    round_trip: bool = False
    price_is_total: bool = False
    notes: tuple[str, ...] = ()


class FlightSearchProvider(Protocol):
    name: str

    def search(self, request: FlightRequest) -> FlightSearchResult:
        """Return verified flight options for normalized request."""

    def resolve_booking_action(
        self,
        candidate_ids: tuple[str, ...],
        *,
        selected_flights_json: str | None = None,
    ) -> BookingAction | None:
        """Return a provider booking action for selected candidates, if available."""

    def resolve_booking_url(
        self,
        candidate_ids: tuple[str, ...],
        *,
        selected_flights_json: str | None = None,
    ) -> str | None:
        """Return a direct HTTPS booking URL for selected candidates, if available."""


def search_dates(request: FlightRequest) -> tuple[date, date | None]:
    """Shared guard before provider adapters are called."""
    if request.origin is None or request.destination is None:
        raise ValueError("origin and destination required before search")
    if request.departure_date is None:
        raise ValueError("departure_date required before search")
    return request.departure_date, request.return_date
