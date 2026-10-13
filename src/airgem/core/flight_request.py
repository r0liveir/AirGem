from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any


@dataclass(frozen=True)
class FlightRequest:
    """Normalized user intent. Missing values stay None; model must not guess."""

    origin: str | None = None
    destination: str | None = None
    departure_date: date | None = None
    return_date: date | None = None
    max_price: float | None = None
    currency: str | None = None
    max_stops: int | None = None
    sort_by_price: bool = False

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "FlightRequest":
        allowed = {
            "origin",
            "destination",
            "departure_date",
            "return_date",
            "max_price",
            "currency",
            "max_stops",
            "sort_by_price",
        }
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"Unknown fields: {sorted(unknown)}")

        origin = _airport(raw.get("origin"), "origin")
        destination = _airport(raw.get("destination"), "destination")
        departure_date = _date(raw.get("departure_date"), "departure_date")
        return_date = _date(raw.get("return_date"), "return_date")
        if departure_date and return_date and return_date < departure_date:
            raise ValueError("return_date cannot precede departure_date")

        max_price = raw.get("max_price")
        if max_price is not None:
            if isinstance(max_price, bool) or not isinstance(max_price, (int, float)) or max_price < 0:
                raise ValueError("max_price must be a non-negative number")

        max_stops = raw.get("max_stops")
        if max_stops is not None:
            if isinstance(max_stops, bool) or not isinstance(max_stops, int) or max_stops < 0:
                raise ValueError("max_stops must be a non-negative integer")

        sort_by_price = raw.get("sort_by_price", False)
        if not isinstance(sort_by_price, bool):
            raise ValueError("sort_by_price must be boolean")

        currency = raw.get("currency")
        if currency is not None:
            if not isinstance(currency, str) or not re.fullmatch(r"[A-Za-z]{3}", currency):
                raise ValueError("currency must be a 3-letter code")
            currency = currency.upper()

        return cls(origin, destination, departure_date, return_date, max_price, currency, max_stops, sort_by_price)

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        for field in ("departure_date", "return_date"):
            if values[field]:
                values[field] = values[field].isoformat()
        return values

    @property
    def missing(self) -> list[str]:
        return [
            field
            for field in ("origin", "destination", "departure_date")
            if getattr(self, field) is None
        ]


def _airport(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z]{3}", value.strip()):
        raise ValueError(f"{field} must be a 3-letter IATA code or null")
    return value.strip().upper()


def _date(value: Any, field: str) -> date | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be ISO date or null")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field} must be ISO date or null") from error
