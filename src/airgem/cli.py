from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Any

from airgem.core import parse_flight_request
from airgem.itinerary import build_itineraries, itinerary_to_dict
from airgem.llm import OllamaProvider, OpenRouterProvider
from airgem.search import FlightSearchProvider, FlightSearchResult, SerpApiProvider


def run(prompt: str, llm_provider: Any, search_provider: FlightSearchProvider) -> dict[str, Any]:
    request = parse_flight_request(prompt, llm_provider)
    if request.missing:
        missing = ", ".join(request.missing)
        raise ValueError(f"missing required flight fields: {missing}")
    result = search_provider.search(request)
    itineraries = build_itineraries(result, max_price=request.max_price)
    return {
        "request": request.to_dict(),
        "itineraries": [itinerary_to_dict(item) for item in itineraries],
        **_search_result_to_dict(result),
    }


def _search_result_to_dict(result: FlightSearchResult) -> dict[str, Any]:
    return {
        "outbound": [_flight_option_to_dict(option) for option in result.outbound],
        "inbound": [_flight_option_to_dict(option) for option in result.inbound],
        "round_trip": result.round_trip,
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
    args = parser.parse_args()

    try:
        llm_provider = OllamaProvider() if args.llm_provider == "ollama" else OpenRouterProvider()
        search_provider = SerpApiProvider()
        payload = run(args.prompt, llm_provider, search_provider)
        print(json.dumps(payload, indent=2) if args.json else _format_human(payload))
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
