from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from airgem.search import FlightOption, FlightSearchResult


@dataclass(frozen=True)
class Itinerary:
    id: str
    outbound: FlightOption
    inbound: FlightOption | None
    price: float
    currency: str
    duration_minutes: int | None
    stops: int | None
    estimated: bool = False
    return_pending: bool = False


def build_itineraries(
    result: FlightSearchResult,
    *,
    max_price: float | None = None,
) -> tuple[Itinerary, ...]:
    if result.round_trip and result.inbound:
        itineraries = (
            Itinerary(
                id=f"round-trip-{outbound_index}-{inbound_index}",
                outbound=outbound,
                inbound=inbound,
                price=outbound.price + inbound.price,
                currency=outbound.currency,
                duration_minutes=_duration(outbound) + _duration(inbound),
                stops=outbound.stops + inbound.stops,
                estimated=True,
            )
            for outbound_index, outbound in enumerate(result.outbound)
            for inbound_index, inbound in enumerate(result.inbound)
            if outbound.currency == inbound.currency
            and _can_pair(outbound, inbound)
        )
    elif result.round_trip:
        itineraries = tuple(
            Itinerary(
                id=f"round-trip-{index}",
                outbound=option,
                inbound=None,
                price=option.price,
                currency=option.currency,
                duration_minutes=None,
                stops=None,
                return_pending=True,
            )
            for index, option in enumerate(result.outbound)
        )
    else:
        itineraries = tuple(
            Itinerary(
                id=f"one-way-{index}",
                outbound=option,
                inbound=None,
                price=option.price,
                currency=option.currency,
                duration_minutes=_duration(option),
                stops=option.stops,
            )
            for index, option in enumerate(result.outbound)
        )

    filtered = (
        itinerary
        for itinerary in itineraries
        if max_price is None or itinerary.price <= max_price
    )
    return tuple(sorted(filtered, key=lambda item: (item.price, item.duration_minutes, item.id)))


def itinerary_to_dict(itinerary: Itinerary) -> dict[str, Any]:
    values = asdict(itinerary)
    values["outbound"] = _flight_option_to_dict(itinerary.outbound)
    values["inbound"] = (
        _flight_option_to_dict(itinerary.inbound) if itinerary.inbound is not None else None
    )
    return values


def _flight_option_to_dict(option: FlightOption) -> dict[str, Any]:
    values = asdict(option)
    values["departure"] = option.departure.isoformat()
    values["arrival"] = option.arrival.isoformat()
    return values


def _duration(option: FlightOption) -> int:
    if option.duration_minutes is not None:
        return option.duration_minutes
    return max(0, int((option.arrival - option.departure).total_seconds() // 60))


def _can_pair(outbound: FlightOption, inbound: FlightOption) -> bool:
    try:
        return inbound.departure > outbound.arrival
    except TypeError:
        return True
