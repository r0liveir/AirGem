from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol, Sequence

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
    booking_url: str | None = None


class FlightSearchProvider(Protocol):
    name: str

    def search(self, request: FlightRequest) -> Sequence[FlightOption]:
        """Return verified flight options for normalized request."""


def search_dates(request: FlightRequest) -> tuple[date, date | None]:
    """Shared guard before provider adapters are called."""
    if request.origin is None or request.destination is None:
        raise ValueError("origin and destination required before search")
    if request.departure_date is None:
        raise ValueError("departure_date required before search")
    return request.departure_date, request.return_date

