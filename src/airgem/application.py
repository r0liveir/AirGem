from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, time, timedelta
from typing import Any, Callable, Literal

from airgem.agent import AgentResult, AgentSession, FlightAgent
from airgem.calendar import BusyInterval, load_ics, parse_ics
from airgem.core import FlightRequest, parse_flight_request
from airgem.itinerary import itinerary_to_dict
from airgem.llm import LLMProvider
from airgem.search import BookingAction, FlightSearchProvider, FlightSearchResult


@dataclass
class ApplicationSession:
    """In-memory application handle joining agent and provider state."""

    agent: FlightAgent
    session: AgentSession
    busy_intervals: tuple[BusyInterval, ...] = ()
    buffer_minutes: int = 0
    messages: list["ChatMessage"] = field(default_factory=list)
    booking_action: BookingAction | None = None


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str


def start_session(
    prompt: str,
    llm_provider: LLMProvider,
    search_provider: FlightSearchProvider,
    *,
    busy_intervals: tuple[BusyInterval, ...] = (),
    buffer_minutes: int = 0,
    calendar_path: str | None = None,
    calendar_text: str | None = None,
    stage_callback: Callable[[str], None] | None = None,
) -> tuple[ApplicationSession, dict[str, Any]]:
    if stage_callback is not None:
        stage_callback("parsing_request")
    request = parse_flight_request(prompt, llm_provider)
    if request.missing:
        missing = ", ".join(request.missing)
        raise ValueError(f"missing required flight fields: {missing}")
    if calendar_path and calendar_text:
        raise ValueError("provide either calendar_path or calendar_text, not both")
    if calendar_path or calendar_text:
        if stage_callback is not None:
            stage_callback("loading_calendar")
        calendar_end_date = request.return_date or request.departure_date
        calendar_start = datetime.combine(request.departure_date - timedelta(days=1), time.min)
        calendar_end = datetime.combine(calendar_end_date + timedelta(days=2), time.min)
        if calendar_text is not None:
            busy_intervals = parse_ics(calendar_text, start=calendar_start, end=calendar_end)
        else:
            busy_intervals = load_ics(
                calendar_path,
                start=calendar_start,
                end=calendar_end,
            )

    agent = FlightAgent(llm_provider, search_provider)
    session, result = agent.start(
        prompt,
        request,
        busy_intervals=busy_intervals,
        buffer_minutes=buffer_minutes,
        stage_callback=stage_callback,
    )
    result, booking_action = _attach_booking_action(result, search_provider, stage_callback)
    application_session = ApplicationSession(
        agent,
        session,
        busy_intervals,
        buffer_minutes,
        booking_action=booking_action,
    )
    application_session.messages.append(ChatMessage("user", prompt))
    payload = result_to_payload(result, request, busy_intervals, buffer_minutes)
    application_session.messages.append(ChatMessage("assistant", payload["agent_message"]))
    return application_session, _session_payload(application_session, payload)


def _attach_booking_action(
    result: AgentResult,
    search_provider: FlightSearchProvider,
    stage_callback: Callable[[str], None] | None,
) -> tuple[AgentResult, BookingAction | None]:
    if result.selected is None:
        return result, None
    selected_flights_json = _selected_flights_json(result.selected)
    candidate_ids = tuple(
        candidate_id
        for candidate_id in (
            result.selected.outbound.candidate_id,
            result.selected.inbound.candidate_id if result.selected.inbound else None,
        )
        if candidate_id
    )
    if not candidate_ids and not selected_flights_json:
        return result, None
    if stage_callback is not None:
        stage_callback("fetching_booking")
    try:
        action_resolver = getattr(search_provider, "resolve_booking_action", None)
        if callable(action_resolver):
            booking_action = action_resolver(
                candidate_ids,
                selected_flights_json=selected_flights_json,
            )
        else:
            url_resolver = getattr(search_provider, "resolve_booking_url", None)
            if not callable(url_resolver) or selected_flights_json:
                return result, None
            booking_url = url_resolver(candidate_ids)
            booking_action = BookingAction(booking_url) if isinstance(booking_url, str) else None
    except (RuntimeError, ValueError):
        return result, None
    if not isinstance(booking_action, BookingAction):
        return result, None
    selected = replace(result.selected, booking_url=booking_action.url)
    itineraries = tuple(
        selected if itinerary.id == selected.id else itinerary for itinerary in result.itineraries
    )
    return replace(result, itineraries=itineraries, selected=selected), booking_action


def _selected_flights_json(itinerary: Any) -> str | None:
    if itinerary.inbound is None:
        return None
    if not itinerary.outbound.segments or not itinerary.inbound.segments:
        return None

    def serialize(segments: object) -> list[dict[str, str]]:
        return [
            {
                "flight_number": "".join(segment.flight_number.split()),
                "departure_id": segment.origin,
                "arrival_id": segment.destination,
                "date": segment.departure.date().isoformat(),
            }
            for segment in segments
        ]

    return json.dumps(
        {
            "outbound": serialize(itinerary.outbound.segments),
            "return": serialize(itinerary.inbound.segments),
        },
        separators=(",", ":"),
    )


def continue_session(
    application_session: ApplicationSession,
    prompt: str,
    *,
    stage_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    session = application_session.session
    if stage_callback is not None:
        session.stage_callback = stage_callback
    result = application_session.agent.continue_session(session, prompt)
    result, application_session.booking_action = _attach_booking_action(
        result,
        application_session.agent.search,
        stage_callback,
    )
    payload = result_to_payload(
        result,
        session.request,
        application_session.busy_intervals,
        application_session.buffer_minutes,
    )
    application_session.messages.append(ChatMessage("user", prompt))
    application_session.messages.append(ChatMessage("assistant", payload["agent_message"]))
    return _session_payload(application_session, payload)


def _session_payload(application_session: ApplicationSession, payload: dict[str, Any]) -> dict[str, Any]:
    payload["messages"] = [asdict(message) for message in application_session.messages]
    payload["available_actions"] = _available_actions(application_session.session)
    return payload


def _available_actions(session: AgentSession) -> list[dict[str, str]]:
    actions: list[dict[str, str]] = []
    for itinerary in session.deferred_outbound:
        actions.append(
            {
                "label": f"Explore {itinerary.id}",
                "prompt": f"Try outbound {itinerary.id}",
            }
        )
    if not session.deferred_outbound and not (
        session.search_result and session.search_result.round_trip
    ):
        for itinerary in session.outbound_candidates:
            if session.selected and itinerary.id == session.selected.id:
                continue
            actions.append(
                {
                    "label": f"Select {itinerary.id}",
                    "prompt": f"Select {itinerary.id}",
                }
            )
    return actions


def result_to_payload(
    result: AgentResult,
    request: FlightRequest,
    busy_intervals: tuple[BusyInterval, ...] = (),
    buffer_minutes: int = 0,
) -> dict[str, Any]:
    notes = list(result.search_result.notes + result.notes)
    if busy_intervals:
        notes.append(
            f"Applied {len(busy_intervals)} calendar event(s) with a "
            f"{buffer_minutes}-minute travel buffer before Pareto filtering."
        )
    return {
        "request": request.to_dict(),
        "itineraries": [itinerary_to_dict(item) for item in result.itineraries],
        "selected": itinerary_to_dict(result.selected) if result.selected else None,
        "alternatives": [item.id for item in result.alternatives],
        "deferred_outbound": [itinerary_to_dict(item) for item in result.deferred_outbound],
        "rationale": _rationale(result, request, busy_intervals, buffer_minutes),
        "agent_message": _agent_message(result),
        "agent_state": result.state.value,
        **_search_result_to_dict(result.search_result),
        "notes": notes,
    }


def _rationale(
    result: AgentResult,
    request: FlightRequest,
    busy_intervals: tuple[BusyInterval, ...],
    buffer_minutes: int,
) -> dict[str, Any] | None:
    """Return verified selection facts, never generated justification."""
    if result.selected is None:
        return None
    selected = result.selected
    return {
        "selected_id": selected.id,
        "evaluated_candidates": len(result.itineraries),
        "alternative_ids": [item.id for item in result.alternatives],
        "deferred_outbound_ids": [item.id for item in result.deferred_outbound],
        "metrics": {
            "price": selected.price,
            "currency": selected.currency,
            "duration_minutes": selected.duration_minutes,
            "stops": selected.stops,
        },
        "constraints": {
            "max_price": request.max_price,
            "max_stops": request.max_stops,
            "calendar_events": len(busy_intervals),
            "buffer_minutes": buffer_minutes if busy_intervals else None,
        },
    }


def _agent_message(result: AgentResult) -> str:
    if result.selected is None:
        return "No itinerary selected."
    message = f"Selected {result.selected.id}."
    if result.alternatives:
        message += f" {len(result.alternatives)} evaluated alternative(s) available."
    if result.deferred_outbound:
        message += f" {len(result.deferred_outbound)} outbound option(s) still have pending return details."
    return message


def _search_result_to_dict(result: FlightSearchResult) -> dict[str, Any]:
    return {
        "outbound": [_flight_option_to_dict(option) for option in result.outbound],
        "inbound": [_flight_option_to_dict(option) for option in result.inbound],
        "round_trip": result.round_trip,
        "price_is_total": result.price_is_total,
        "notes": list(result.notes),
    }


def _flight_option_to_dict(option: Any) -> dict[str, Any]:
    values = asdict(option)
    values["departure"] = option.departure.isoformat()
    values["arrival"] = option.arrival.isoformat()
    values["segments"] = [
        {
            "flight_number": segment.flight_number,
            "origin": segment.origin,
            "destination": segment.destination,
            "departure": segment.departure.isoformat(),
            "arrival": segment.arrival.isoformat(),
        }
        for segment in option.segments
    ]
    return values
