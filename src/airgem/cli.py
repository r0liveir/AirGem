from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from datetime import datetime, time, timedelta
from typing import Any

from airgem.agent import FlightAgent
from airgem.calendar import BusyInterval, load_ics
from airgem.core import parse_flight_request
from airgem.itinerary import itinerary_to_dict
from airgem.llm import OllamaProvider, OpenRouterProvider
from airgem.search import FlightSearchProvider, FlightSearchResult, SerpApiProvider


def run(
    prompt: str,
    llm_provider: Any,
    search_provider: FlightSearchProvider,
    *,
    busy_intervals: tuple[BusyInterval, ...] = (),
    buffer_minutes: int = 0,
    calendar_path: str | None = None,
) -> dict[str, Any]:
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
    agent_result = FlightAgent(llm_provider, search_provider).run(
        prompt,
        request,
        busy_intervals=busy_intervals,
        buffer_minutes=buffer_minutes,
    )
    notes = list(agent_result.search_result.notes + agent_result.notes)
    if busy_intervals:
        notes.append(
            f"Applied {len(busy_intervals)} calendar event(s) with a "
            f"{buffer_minutes}-minute travel buffer before Pareto filtering."
        )
    return {
        "request": request.to_dict(),
        "itineraries": [itinerary_to_dict(item) for item in agent_result.itineraries],
        "selected": itinerary_to_dict(agent_result.selected) if agent_result.selected else None,
        "agent_state": agent_result.state.value,
        **_search_result_to_dict(agent_result.search_result),
        "notes": notes,
    }


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


def _format_human(payload: dict[str, Any]) -> str:
    request = payload["request"]
    route = f"{request['origin']} → {request['destination']}"
    dates = request["departure_date"]
    if request["return_date"]:
        dates += f" → {request['return_date']}"
    lines = [f"{route} · {dates} · {request['currency']}", ""]

    itineraries = payload["itineraries"]
    if not itineraries:
        lines.append("No flights found.")
    for index, itinerary in enumerate(itineraries, start=1):
        outbound = itinerary["outbound"]
        airlines = [outbound["airline"]] if outbound["airline"] else []
        if itinerary["inbound"] and itinerary["inbound"]["airline"]:
            airlines.append(itinerary["inbound"]["airline"])
        airline = " + ".join(dict.fromkeys(airlines)) or "Unknown airline"
        stops = _stops_label(itinerary["stops"])
        duration = _duration_label(itinerary["duration_minutes"])
        status = " · return details pending" if itinerary["return_pending"] else ""
        if itinerary["estimated"]:
            status = " · estimated round trip"
        lines.append(
            f"{index}. {airline} · {itinerary['currency']} {itinerary['price']:.2f} · "
            f"{stops} · {duration}{status}"
        )
        lines.append(f"   {outbound['origin']} → {outbound['destination']} · {outbound['departure']}")
        if itinerary["inbound"]:
            inbound = itinerary["inbound"]
            lines.append(f"   {inbound['origin']} → {inbound['destination']} · {inbound['departure']}")

    if payload["notes"]:
        lines.extend(["", "Notes:"])
        lines.extend(f"- {note}" for note in payload["notes"])
    return "\n".join(lines)


def _stops_label(stops: int | None) -> str:
    if stops is None:
        return "stops pending"
    return "nonstop" if stops == 0 else f"{stops} stop" if stops == 1 else f"{stops} stops"


def _duration_label(minutes: int | None) -> str:
    if minutes is None:
        return "duration pending"
    hours, remainder = divmod(minutes, 60)
    return f"{hours}h {remainder:02d}m" if hours else f"{remainder}m"


def main() -> None:
    parser = argparse.ArgumentParser(description="Search flights from a natural-language request")
    parser.add_argument("prompt")
    parser.add_argument(
        "--provider",
        "--llm-provider",
        dest="llm_provider",
        choices=("ollama", "openrouter"),
        default=os.getenv("LLM_PROVIDER", "ollama"),
    )
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument("--debug", action="store_true", help="print agent decision trace")
    parser.add_argument("--calendar", help="path to a local .ics calendar export")
    parser.add_argument(
        "--buffer-minutes",
        type=int,
        default=60,
        help="free time required before departure and after arrival when using --calendar",
    )
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="DEBUG %(name)s: %(message)s",
            stream=sys.stdout,
            force=True,
        )

    try:
        if args.buffer_minutes < 0:
            raise ValueError("--buffer-minutes must be non-negative")
        llm_provider = OllamaProvider() if args.llm_provider == "ollama" else OpenRouterProvider()
        search_provider = SerpApiProvider()
        payload = run(
            args.prompt,
            llm_provider,
            search_provider,
            buffer_minutes=args.buffer_minutes,
            calendar_path=args.calendar,
        )
        print(json.dumps(payload, indent=2) if args.json else _format_human(payload))
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
