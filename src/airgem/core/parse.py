from __future__ import annotations

import json
import logging
import re
from datetime import date
from typing import Any

from airgem.llm import LLMProvider

from .flight_request import FlightRequest

SYSTEM_PROMPT = """You extract flight-search intent from user text.
Return JSON only. No markdown. Exact keys:
origin, destination, departure_date, return_date, max_price, currency, max_stops, sort_by_price.
Use 3-letter IATA airport codes. Use ISO dates (YYYY-MM-DD). Resolve relative dates
from the supplied current date. Use null when user did not provide a value. Never guess.
max_stops is 0 for nonstop, otherwise a non-negative integer. max_price is numeric.
Set sort_by_price true for requests emphasizing cheap/lowest-price flights.
"""

logger = logging.getLogger(__name__)


def parse_flight_request(
    text: str,
    provider: LLMProvider,
    *,
    today: date | None = None,
) -> FlightRequest:
    if not text.strip():
        raise ValueError("flight request cannot be empty")
    current_date = today or date.today()
    raw_text = provider.complete(
        system=SYSTEM_PROMPT,
        user=f"Current date: {current_date.isoformat()}\nRequest: {text.strip()}",
    )
    try:
        data = json.loads(_json_text(raw_text))
    except json.JSONDecodeError as error:
        raise ValueError(f"LLM returned invalid JSON: {raw_text!r}") from error
    if not isinstance(data, dict):
        raise ValueError("LLM JSON must be an object")
    request = FlightRequest.from_dict(data)
    logger.debug("intent parsed request=%s", request.to_dict())
    return request


def _json_text(value: str) -> str:
    value = value.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, re.DOTALL | re.IGNORECASE)
    return fenced.group(1) if fenced else value
