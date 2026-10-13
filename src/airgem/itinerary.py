from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from typing import Any

from airgem.calendar import BusyInterval
from airgem.pareto import bounded_pareto_front, pareto_front
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
    max_stops: int | None = None,
    departure_date: date | None = None,
    return_date: date | None = None,
    busy_intervals: tuple[BusyInterval, ...] = (),
    buffer_minutes: int = 0,
    candidate_limit: int | None = None,
) -> tuple[Itinerary, ...]:
    if buffer_minutes < 0:
        raise ValueError("buffer_minutes must be non-negative")
    if candidate_limit is not None and candidate_limit < 1:
        raise ValueError("candidate_limit must be positive")
    if result.round_trip and result.inbound:
        itineraries = (
            Itinerary(
                id=_paired_id(outbound, inbound, outbound_index, inbound_index),
                outbound=outbound,
                inbound=inbound,
                price=outbound.price if result.price_is_total else outbound.price + inbound.price,
                currency=outbound.currency,
                duration_minutes=_duration(outbound) + _duration(inbound),
                stops=outbound.stops + inbound.stops,
                estimated=not result.price_is_total,
            )
            for outbound_index, outbound in enumerate(result.outbound)
            for inbound_index, inbound in enumerate(result.inbound)
            if _fits(outbound, departure_date, max_stops)
            and _fits(inbound, return_date, max_stops)
            and _available(outbound, busy_intervals, buffer_minutes)
            and _available(inbound, busy_intervals, buffer_minutes)
            and outbound.currency == inbound.currency
            and _can_pair(outbound, inbound)
        )
    elif result.round_trip:
        itineraries = tuple(
            Itinerary(
                id=option.candidate_id or f"round-trip-{index}",
                outbound=option,
                inbound=None,
                price=option.price,
                currency=option.currency,
                duration_minutes=None,
                stops=None,
                return_pending=True,
            )
            for index, option in enumerate(result.outbound)
            if _fits(option, departure_date, max_stops)
            and _available(option, busy_intervals, buffer_minutes)
        )
    else:
        itineraries = tuple(
            Itinerary(
                id=option.candidate_id or f"one-way-{index}",
                outbound=option,
                inbound=None,
                price=option.price,
                currency=option.currency,
                duration_minutes=_duration(option),
                stops=option.stops,
            )
            for index, option in enumerate(result.outbound)
            if _fits(option, departure_date, max_stops)
            and _available(option, busy_intervals, buffer_minutes)
        )

    filtered = (
        itinerary
        for itinerary in itineraries
        if max_price is None or itinerary.price <= max_price
    )
    candidates = tuple(filtered)
    if candidate_limit is None:
        candidates = pareto_front(candidates, _metrics)
    else:
        candidates = bounded_pareto_front(candidates, _metrics, limit=candidate_limit)
    return tuple(sorted(candidates, key=_sort_key))


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


def _fits(option: FlightOption, expected_date: date | None, max_stops: int | None) -> bool:
    return (
        (expected_date is None or option.departure.date() == expected_date)
        and (max_stops is None or option.stops <= max_stops)
    )


def _available(
    option: FlightOption,
    busy_intervals: tuple[BusyInterval, ...],
    buffer_minutes: int,
) -> bool:
    if not busy_intervals:
        return True
    buffer = timedelta(minutes=buffer_minutes)
    travel_start = option.departure - buffer
    travel_end = option.arrival + buffer
    return not any(_overlaps(travel_start, travel_end, busy) for busy in busy_intervals)


def _overlaps(start: datetime, end: datetime, busy: BusyInterval) -> bool:
    try:
        return busy.start < end and busy.end > start
    except TypeError:
        # ponytail: SerpApi/ICS timezone normalization belongs in a future timezone layer.
        return busy.start.replace(tzinfo=None) < end.replace(tzinfo=None) and busy.end.replace(tzinfo=None) > start.replace(tzinfo=None)


def _metrics(itinerary: Itinerary) -> tuple[float, ...]:
    if itinerary.return_pending:
        # Native round-trip result exposes outbound metrics before token expansion.
        return (itinerary.price, _duration(itinerary.outbound), itinerary.outbound.stops)
    return (
        itinerary.price,
        float(itinerary.duration_minutes or 0),
        float(itinerary.stops or 0),
    )


def _sort_key(itinerary: Itinerary) -> tuple[float, float, str]:
    return (
        itinerary.price,
        float(itinerary.duration_minutes or 0),
        itinerary.id,
    )


def _paired_id(
    outbound: FlightOption,
    inbound: FlightOption,
    outbound_index: int,
    inbound_index: int,
) -> str:
    if inbound.candidate_id:
        return inbound.candidate_id
    if outbound.candidate_id:
        return outbound.candidate_id
    return f"round-trip-{outbound_index}-{inbound_index}"


def _can_pair(outbound: FlightOption, inbound: FlightOption) -> bool:
    try:
        return inbound.departure > outbound.arrival
    except TypeError:
        return True
