from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

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


@dataclass(frozen=True)
class AgentResult:
    state: AgentState
    search_result: FlightSearchResult
    itineraries: tuple[Itinerary, ...]
    selected: Itinerary | None
    notes: tuple[str, ...] = ()


class FlightAgent:
    """Small bounded controller around deterministic flight tools."""

    def __init__(self, llm: LLMProvider, search: FlightSearchProvider, *, max_turns: int = 2) -> None:
        self.llm = llm
        self.search = search
        self.max_turns = max(0, max_turns)
        self.state = AgentState.SEARCHING
        self._turns = 0
        self._selection_notes: list[str] = []

    def run(self, prompt: str, request: FlightRequest) -> AgentResult:
        self._set_state(AgentState.SEARCHING)
        self._turns = 0
        self._selection_notes = []
        result = self.search.search(request)
        logger.debug(
            "search complete round_trip=%s outbound=%d inbound=%d",
            result.round_trip,
            len(result.outbound),
            len(result.inbound),
        )
        itineraries = _build(result, request)
        logger.debug("Pareto candidates=%d ids=%s", len(itineraries), [item.id for item in itineraries])
        if not itineraries:
            self._set_state(AgentState.COMPLETE)
            return AgentResult(self.state, result, (), None, ("No valid candidates found.",))

        self._set_state(AgentState.SELECTING_OUTBOUND)
        outbound = self._select("select_candidate", prompt, itineraries)
        if not result.round_trip:
            self._set_state(AgentState.COMPLETE)
            return AgentResult(self.state, result, itineraries, outbound, self._notes)

        if not hasattr(self.search, "expand_round_trip"):
            raise ValueError("search provider cannot expand round trips")

        self._set_state(AgentState.EXPANDING_RETURN)
        logger.debug("expanding selected candidate id=%s", outbound.id)
        expanded = self.search.expand_round_trip(outbound.id)  # type: ignore[attr-defined]
        expanded_itineraries = _build(expanded, request)
        logger.debug("return expansion candidates=%d ids=%s", len(expanded_itineraries), [item.id for item in expanded_itineraries])
        if not expanded_itineraries:
            self._set_state(AgentState.COMPLETE)
            return AgentResult(
                self.state,
                expanded,
                (),
                None,
                self._notes + ("Selected candidate returned no valid return flights.",),
            )

        self._set_state(AgentState.SELECTING_RETURN)
        selected = self._select("select_return", prompt, expanded_itineraries)
        self._set_state(AgentState.COMPLETE)
        return AgentResult(self.state, expanded, expanded_itineraries, selected, self._notes)

    @property
    def _notes(self) -> tuple[str, ...]:
        return tuple(self._selection_notes)

    def _select(self, action: str, prompt: str, candidates: tuple[Itinerary, ...]) -> Itinerary:
        if len(candidates) == 1:
            logger.debug("single candidate id=%s; skip LLM selection", candidates[0].id)
            return candidates[0]
        if self._turns >= self.max_turns:
            return self._fallback(candidates, "Agent turn budget exhausted; used Pareto fallback.")

        self._turns += 1
        logger.debug("LLM selection turn=%d action=%s candidates=%d", self._turns, action, len(candidates))
        response = self.llm.complete(
            system=(
                "You select flight candidates. Return JSON only. "
                f"Required action: {action}. Use one candidate id from the list. "
                "Never invent IDs. Prefer user-stated priorities; otherwise choose cheapest."
            ),
            user=json.dumps(
                {
                    "request": prompt,
                    "candidates": [_candidate_to_dict(candidate) for candidate in candidates],
                    "output": {"action": action, "candidate_id": "..."},
                }
            ),
        )
        candidate_id = _action_candidate_id(response, action)
        logger.debug("LLM action=%s candidate_id=%s", action, candidate_id or "<invalid>")
        for candidate in candidates:
            if candidate.id == candidate_id:
                logger.debug("selected candidate id=%s", candidate.id)
                return candidate
        return self._fallback(candidates, "LLM chose invalid candidate; used Pareto fallback.")

    def _fallback(self, candidates: tuple[Itinerary, ...], note: str = "Used deterministic Pareto fallback.") -> Itinerary:
        if note not in self._selection_notes:
            self._selection_notes.append(note)
        logger.debug("fallback candidate id=%s reason=%s", candidates[0].id, note)
        return candidates[0]

    def _set_state(self, state: AgentState) -> None:
        self.state = state
        logger.debug("state=%s", state.value)


def _build(result: FlightSearchResult, request: FlightRequest) -> tuple[Itinerary, ...]:
    return build_itineraries(
        result,
        max_price=request.max_price,
        max_stops=request.max_stops,
        departure_date=request.departure_date,
        return_date=request.return_date,
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


def _action_candidate_id(response: str, expected_action: str) -> str | None:
    try:
        value = response.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, re.DOTALL | re.IGNORECASE)
        payload = json.loads(fenced.group(1) if fenced else value)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("action") != expected_action or not isinstance(payload.get("candidate_id"), str):
        return None
    return payload["candidate_id"]
