from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable

from airgem.calendar import BusyInterval
from airgem.core import FlightRequest
from airgem.itinerary import Itinerary, build_itineraries, itinerary_to_dict
from airgem.llm import LLMProvider
from airgem.search import FlightSearchProvider, FlightSearchResult

logger = logging.getLogger(__name__)


class AgentState(StrEnum):
    SEARCHING = "searching"
    SELECTING_OUTBOUND = "selecting_outbound"
    EXPANDING_RETURN = "expanding_return"
    SELECTING_RETURN = "selecting_return"
    COMPLETE = "complete"


@dataclass
class AgentSession:
    """In-memory state for bounded follow-up actions."""

    prompt: str
    request: FlightRequest
    busy_intervals: tuple[BusyInterval, ...] = ()
    buffer_minutes: int = 0
    outbound_candidates: tuple[Itinerary, ...] = ()
    deferred_outbound: tuple[Itinerary, ...] = ()
    expanded_itineraries: tuple[Itinerary, ...] = ()
    selected: Itinerary | None = None
    search_result: FlightSearchResult | None = None
    state: AgentState = AgentState.SEARCHING
    turns: int = 0
    selection_notes: list[str] = field(default_factory=list)
    selection_reason: str | None = None
    expanded_outbound_ids: set[str] = field(default_factory=set)
    stage_callback: Callable[[str], None] | None = None


@dataclass(frozen=True)
class AgentResult:
    state: AgentState
    search_result: FlightSearchResult
    itineraries: tuple[Itinerary, ...]
    selected: Itinerary | None
    reason: str | None = None
    notes: tuple[str, ...] = ()
    alternatives: tuple[Itinerary, ...] = ()
    deferred_outbound: tuple[Itinerary, ...] = ()


class FlightAgent:
    """Small bounded controller around deterministic flight tools."""

    def __init__(self, llm: LLMProvider, search: FlightSearchProvider, *, max_turns: int = 2) -> None:
        self.llm = llm
        self.search = search
        self.max_turns = max(0, max_turns)
        self.state = AgentState.SEARCHING

    def run(
        self,
        prompt: str,
        request: FlightRequest,
        *,
        busy_intervals: tuple[BusyInterval, ...] = (),
        buffer_minutes: int = 0,
    ) -> AgentResult:
        _, result = self.start(
            prompt,
            request,
            busy_intervals=busy_intervals,
            buffer_minutes=buffer_minutes,
        )
        return result

    def start(
        self,
        prompt: str,
        request: FlightRequest,
        *,
        busy_intervals: tuple[BusyInterval, ...] = (),
        buffer_minutes: int = 0,
        stage_callback: Callable[[str], None] | None = None,
    ) -> tuple[AgentSession, AgentResult]:
        session = AgentSession(
            prompt=prompt,
            request=request,
            busy_intervals=busy_intervals,
            buffer_minutes=buffer_minutes,
            stage_callback=stage_callback,
        )
        self._set_state(session, AgentState.SEARCHING)
        result = self.search.search(request)
        session.search_result = result
        logger.debug(
            "search complete round_trip=%s outbound=%d inbound=%d",
            result.round_trip,
            len(result.outbound),
            len(result.inbound),
        )
        self._stage(session, "evaluating_options")
        itineraries = _build(result, request, busy_intervals, buffer_minutes)
        logger.debug("candidate pool=%d ids=%s", len(itineraries), [item.id for item in itineraries])
        session.outbound_candidates = itineraries
        if not itineraries:
            session.selection_notes.append("No valid candidates found.")
            self._set_state(session, AgentState.COMPLETE)
            return session, self._result(session)

        self._set_state(session, AgentState.SELECTING_OUTBOUND)
        outbound = self._select(session, "select_candidate", prompt, itineraries)
        if not result.round_trip:
            session.expanded_itineraries = itineraries
            session.selected = outbound
            self._set_state(session, AgentState.COMPLETE)
            return session, self._result(session)

        session.deferred_outbound = _without(itineraries, outbound)
        return session, self._expand_outbound(session, outbound, prompt)

    def continue_session(self, session: AgentSession, prompt: str) -> AgentResult:
        """Apply a bounded follow-up to the current candidate set."""
        session.prompt = prompt
        session.turns = 0
        if session.deferred_outbound:
            if not hasattr(self.search, "expand_round_trip"):
                raise ValueError("search provider cannot expand round trips")
            self._set_state(session, AgentState.SELECTING_OUTBOUND)
            outbound = self._select(
                session,
                "select_deferred_outbound",
                prompt,
                session.deferred_outbound,
            )
            session.deferred_outbound = _without(session.deferred_outbound, outbound)
            return self._expand_outbound(session, outbound, prompt)

        candidates = _without(session.outbound_candidates, session.selected) if session.selected else session.outbound_candidates
        if not candidates:
            self._set_state(session, AgentState.COMPLETE)
            return self._result(session)
        self._set_state(session, AgentState.SELECTING_OUTBOUND)
        session.selected = self._select(session, "select_candidate", prompt, candidates)
        self._set_state(session, AgentState.COMPLETE)
        return self._result(session)

    def _expand_outbound(
        self,
        session: AgentSession,
        outbound: Itinerary,
        prompt: str,
    ) -> AgentResult:
        if not hasattr(self.search, "expand_round_trip"):
            raise ValueError("search provider cannot expand round trips")

        self._set_state(session, AgentState.EXPANDING_RETURN)
        logger.debug("expanding selected candidate id=%s", outbound.id)
        expanded = self.search.expand_round_trip(outbound.id)  # type: ignore[attr-defined]
        session.expanded_outbound_ids.add(outbound.id)
        session.search_result = expanded
        self._stage(session, "evaluating_return_options")
        expanded_itineraries = _build(
            expanded,
            session.request,
            session.busy_intervals,
            session.buffer_minutes,
        )
        session.expanded_itineraries = _merge_unique(session.expanded_itineraries, expanded_itineraries)
        logger.debug(
            "return expansion candidates=%d ids=%s",
            len(expanded_itineraries),
            [item.id for item in expanded_itineraries],
        )
        if not expanded_itineraries:
            note = "Selected candidate returned no valid return flights."
            if note not in session.selection_notes:
                session.selection_notes.append(note)
            self._set_state(session, AgentState.COMPLETE)
            return self._result(session)

        self._set_state(session, AgentState.SELECTING_RETURN)
        session.selected = self._select(session, "select_return", prompt, expanded_itineraries)
        self._set_state(session, AgentState.COMPLETE)
        return self._result(session)

    def _result(self, session: AgentSession) -> AgentResult:
        if session.search_result is None:
            raise RuntimeError("agent session has no search result")
        alternatives = (
            _without(session.expanded_itineraries, session.selected)
            if session.selected is not None
            else session.expanded_itineraries
        )
        return AgentResult(
            state=session.state,
            search_result=session.search_result,
            itineraries=session.expanded_itineraries,
            selected=session.selected,
            reason=session.selection_reason,
            notes=tuple(session.selection_notes),
            alternatives=alternatives,
            deferred_outbound=session.deferred_outbound,
        )

    def _select(
        self,
        session: AgentSession,
        action: str,
        prompt: str,
        candidates: tuple[Itinerary, ...],
    ) -> Itinerary:
        if len(candidates) == 1:
            logger.debug("single candidate id=%s; skip LLM selection", candidates[0].id)
            session.selection_reason = "It is the only feasible option after applying your constraints."
            return candidates[0]
        if session.turns >= self.max_turns:
            return self._fallback(session, candidates, "Agent turn budget exhausted; used Pareto fallback.")

        session.turns += 1
        logger.debug("LLM selection turn=%d action=%s candidates=%d", session.turns, action, len(candidates))
        candidate_ids = [candidate.id for candidate in candidates]
        response = self.llm.complete(
            system=(
                "You are a flight candidate selector. "
                "Choose exactly one candidate from an allowed list. "
                "Return one JSON object only. No markdown. No extra keys. "
                f'The action value must be exactly "{action}". '
                "The candidate_id value must be copied exactly from the allowed IDs. "
                "The reason must be one short sentence grounded only in the user request "
                "and supplied candidate details; do not invent preferences or facts. "
                "Never use an index, price, airline, or invented value as the candidate_id. "
                "Prefer the user's stated priorities; otherwise choose lowest price, "
                "then fewest stops, then shortest duration."
            ),
            user=(
                f"User request: {prompt}\n"
                f"Required action: {action}\n"
                "Allowed candidate IDs:\n"
                + "\n".join(f"- {candidate_id}" for candidate_id in candidate_ids)
                + "\nCandidate details:\n"
                + "\n".join(
                    json.dumps(_candidate_to_dict(candidate), separators=(",", ":"))
                    for candidate in candidates
                )
                + "\nReturn exactly this shape, replacing candidate-0 with one allowed ID:\n"
                + json.dumps(
                    {
                        "action": action,
                        "candidate_id": candidate_ids[0],
                        "reason": "It best matches the user's stated priorities.",
                    }
                )
            ),
        )
        candidate_id, reason = _action_choice(response, action)
        logger.debug("LLM action=%s candidate_id=%s", action, candidate_id or "<invalid>")
        for candidate in candidates:
            if candidate.id == candidate_id:
                session.selection_reason = reason
                logger.debug("selected candidate id=%s", candidate.id)
                return candidate
        return self._fallback(session, candidates, "LLM chose invalid candidate; used Pareto fallback.")

    def _fallback(self, session: AgentSession, candidates: tuple[Itinerary, ...], note: str) -> Itinerary:
        if note not in session.selection_notes:
            session.selection_notes.append(note)
        session.selection_reason = "It leads the deterministic ranking for price, stops, and duration."
        logger.debug("fallback candidate id=%s reason=%s", candidates[0].id, note)
        return candidates[0]

    def _set_state(self, session: AgentSession, state: AgentState) -> None:
        self.state = state
        session.state = state
        self._stage(session, state.value)
        logger.debug("state=%s", state.value)

    @staticmethod
    def _stage(session: AgentSession, stage: str) -> None:
        if session.stage_callback is not None:
            session.stage_callback(stage)


def _build(
    result: FlightSearchResult,
    request: FlightRequest,
    busy_intervals: tuple[BusyInterval, ...] = (),
    buffer_minutes: int = 0,
) -> tuple[Itinerary, ...]:
    return build_itineraries(
        result,
        max_price=request.max_price,
        max_stops=request.max_stops,
        departure_date=request.departure_date,
        return_date=request.return_date,
        busy_intervals=busy_intervals,
        buffer_minutes=buffer_minutes,
        candidate_limit=6,
    )


def _candidate_to_dict(itinerary: Itinerary) -> dict[str, Any]:
    values = itinerary_to_dict(itinerary)
    return {
        "id": values["id"],
        "price": values["price"],
        "currency": values["currency"],
        "duration_minutes": values["duration_minutes"],
        "stops": values["stops"],
        "airline": values["outbound"].get("airline"),
        "outbound_departure": values["outbound"]["departure"],
        "inbound_departure": values["inbound"]["departure"] if values["inbound"] else None,
    }


def _without(candidates: tuple[Itinerary, ...], excluded: Itinerary) -> tuple[Itinerary, ...]:
    return tuple(candidate for candidate in candidates if candidate.id != excluded.id)


def _merge_unique(
    current: tuple[Itinerary, ...], additions: tuple[Itinerary, ...]
) -> tuple[Itinerary, ...]:
    seen = {candidate.id for candidate in current}
    merged = list(current)
    for candidate in additions:
        if candidate.id not in seen:
            merged.append(candidate)
            seen.add(candidate.id)
    return tuple(merged)


def _action_choice(response: str, expected_action: str) -> tuple[str | None, str | None]:
    try:
        value = response.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, re.DOTALL | re.IGNORECASE)
        payload = json.loads(fenced.group(1) if fenced else value)
    except (json.JSONDecodeError, TypeError):
        return None, None
    if not isinstance(payload, dict):
        return None, None
    if payload.get("action") != expected_action or not isinstance(payload.get("candidate_id"), str):
        return None, None
    reason = payload.get("reason")
    reason = reason.strip() if isinstance(reason, str) and len(reason.strip()) <= 500 else None
    return payload["candidate_id"], reason
