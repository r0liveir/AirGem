from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, time, timedelta
from typing import Any, Literal

from airgem.agent import AgentResult, AgentSession, FlightAgent
from airgem.calendar import BusyInterval, load_ics
from airgem.core import FlightRequest, parse_flight_request
from airgem.itinerary import itinerary_to_dict
from airgem.llm import LLMProvider
from airgem.search import FlightSearchProvider, FlightSearchResult


@dataclass
class ApplicationSession:
    """In-memory application handle joining agent and provider state."""

    agent: FlightAgent
    session: AgentSession
    busy_intervals: tuple[BusyInterval, ...] = ()
    buffer_minutes: int = 0
    messages: list["ChatMessage"] = field(default_factory=list)


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
) -> tuple[ApplicationSession, dict[str, Any]]:
    request = parse_flight_request(prompt, llm_provider)
    if request.missing:
        missing = ", ".join(request.missing)
        raise ValueError(f"missing required flight fields: {missing}")
    if calendar_path:
        calendar_end_date = request.return_date or request.departure_date
        busy_intervals = load_ics(
            calendar_path,
            start=datetime.combine(request.departure_date - timedelta(days=1), time.min),
            end=datetime.combine(calendar_end_date + timedelta(days=2), time.min),
    )

    agent = FlightAgent(llm_provider, search_provider)
    session, result = agent.start(
        prompt,
        request,
        busy_intervals=busy_intervals,
        buffer_minutes=buffer_minutes,
    )
    application_session = ApplicationSession(agent, session, busy_intervals, buffer_minutes)
    application_session.messages.append(ChatMessage("user", prompt))
    payload = result_to_payload(result, request, busy_intervals, buffer_minutes)
    application_session.messages.append(ChatMessage("assistant", payload["agent_message"]))
    return application_session, _session_payload(application_session, payload)


def continue_session(application_session: ApplicationSession, prompt: str) -> dict[str, Any]:
    session = application_session.session
    result = application_session.agent.continue_session(session, prompt)
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
    return values
