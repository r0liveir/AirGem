from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Any

from airgem.core import parse_flight_request
from airgem.llm import OllamaProvider, OpenRouterProvider
from airgem.search import FlightSearchProvider, FlightSearchResult, SerpApiProvider


def run(prompt: str, llm_provider: Any, search_provider: FlightSearchProvider) -> dict[str, Any]:
    request = parse_flight_request(prompt, llm_provider)
    if request.missing:
        missing = ", ".join(request.missing)
        raise ValueError(f"missing required flight fields: {missing}")
    result = search_provider.search(request)
    return {"request": request.to_dict(), **_search_result_to_dict(result)}


def _search_result_to_dict(result: FlightSearchResult) -> dict[str, Any]:
    return {
        "outbound": [_flight_option_to_dict(option) for option in result.outbound],
        "inbound": [_flight_option_to_dict(option) for option in result.inbound],
        "round_trip": result.round_trip,
        "notes": list(result.notes),
    }


def _flight_option_to_dict(option: Any) -> dict[str, Any]:
    values = asdict(option)
    values.pop("booking_token", None)
    values["departure"] = option.departure.isoformat()
    values["arrival"] = option.arrival.isoformat()
    return values


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
    args = parser.parse_args()

    try:
        llm_provider = OllamaProvider() if args.llm_provider == "ollama" else OpenRouterProvider()
        search_provider = SerpApiProvider()
        print(json.dumps(run(args.prompt, llm_provider, search_provider), indent=2))
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
