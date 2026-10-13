from __future__ import annotations

import json
import os
from datetime import date, datetime
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from airgem.core import FlightRequest

from .provider import FlightOption, FlightSearchResult

JsonFetcher = Callable[[dict[str, str]], dict[str, Any]]


class SerpApiProvider:
    """Google Flights adapter using SerpApi's JSON endpoint."""

    name = "serpapi"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        *,
        deep_search: bool = True,
        timeout: float = 120,
        fetch_json: JsonFetcher | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("SERPAPI_API_KEY")
        self.base_url = base_url or os.getenv("SERPAPI_BASE_URL", "https://serpapi.com/search.json")
        self.deep_search = deep_search
        self.timeout = timeout
        self._fetch_json = fetch_json or self._get_json
        if not self.api_key:
            raise ValueError("SERPAPI_API_KEY is required for SerpApiProvider")

    def search(self, request: FlightRequest) -> FlightSearchResult:
        _require_route(request)
        outbound = self.search_one_way(
            departure_id=request.origin,
            arrival_id=request.destination,
            outbound_date=request.departure_date,
            request=request,
        )
        if request.return_date is None:
            return FlightSearchResult(outbound=outbound)

        # TODO: native round-trip search; independent legs intentionally stay unpaired.
        inbound = self.search_one_way(
            departure_id=request.destination,
            arrival_id=request.origin,
            outbound_date=request.return_date,
            request=request,
        )
        return FlightSearchResult(
            outbound=outbound,
            inbound=inbound,
            round_trip=True,
            notes=("Round-trip pairing is work in progress; outbound/inbound options are independent.",),
        )

    def search_one_way(
        self,
        *,
        departure_id: str,
        arrival_id: str,
        outbound_date: date,
        request: FlightRequest,
    ) -> tuple[FlightOption, ...]:
        # TODO: support multiple airport results when request schema supports airport lists.
        params = {
            "engine": "google_flights",
            "api_key": self.api_key,
            "type": "2",
            "departure_id": departure_id,
            "arrival_id": arrival_id,
            "outbound_date": outbound_date.isoformat(),
            "deep_search": str(self.deep_search).lower(),
            "sort_by": "2",  # price
        }
        if request.currency:
            params["currency"] = request.currency
        if request.max_price is not None:
            params["max_price"] = str(request.max_price)
        if request.max_stops is not None:
            params["stops"] = str(request.max_stops + 1)

        payload = self._fetch_json(params)
        _raise_api_error(payload)
        currency = str(payload.get("search_parameters", {}).get("currency", request.currency or "USD")).upper()
        options = tuple(
            self._flight_option(item, currency, outbound_date)
            for item in _flight_groups(payload)
        )
        return tuple(option for option in options if option is not None)

    def _flight_option(
        self,
        item: dict[str, Any],
        currency: str,
        outbound_date: date,
    ) -> FlightOption | None:
        flights = item.get("flights")
        price = item.get("price")
        if not isinstance(flights, list) or not flights or not isinstance(price, (int, float)):
            return None
        try:
            first = flights[0]
            last = flights[-1]
            departure_airport = first["departure_airport"]
            arrival_airport = last["arrival_airport"]
            departure = _parse_datetime(departure_airport["time"])
            arrival = _parse_datetime(arrival_airport["time"])
            origin = str(departure_airport["id"]).upper()
            destination = str(arrival_airport["id"]).upper()
        except (KeyError, TypeError, ValueError):
            return None

        booking_token = item.get("booking_token")
        booking_url = (
            self._booking_url(booking_token, origin, destination, outbound_date)
            if isinstance(booking_token, str)
            else None
        )
        return FlightOption(
            provider=self.name,
            origin=origin,
            destination=destination,
            departure=departure,
            arrival=arrival,
            price=float(price),
            currency=currency,
            stops=max(0, len(flights) - 1),
            booking_url=booking_url,
            booking_token=booking_token if isinstance(booking_token, str) else None,
        )

    def _booking_url(
        self,
        booking_token: str,
        departure_id: str,
        arrival_id: str,
        outbound_date: date,
    ) -> str | None:
        payload = self._fetch_json(
            {
                "engine": "google_flights",
                "api_key": self.api_key,
                "type": "2",
                "booking_token": booking_token,
                "departure_id": departure_id,
                "arrival_id": arrival_id,
                "outbound_date": outbound_date.isoformat(),
            }
        )
        _raise_api_error(payload)
        candidates: list[tuple[float, str]] = []
        for option in payload.get("booking_options", []):
            if not isinstance(option, dict):
                continue
            for key in ("together", "separate"):
                details = option.get(key)
                if not isinstance(details, dict):
                    continue
                booking_request = details.get("booking_request")
                url = booking_request.get("url") if isinstance(booking_request, dict) else None
                price = details.get("price")
                if isinstance(url, str) and isinstance(price, (int, float)):
                    candidates.append((float(price), url))
        return min(candidates)[1] if candidates else None

    def _get_json(self, params: dict[str, str]) -> dict[str, Any]:
        request = Request(_request_url(self.base_url, params), method="GET")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as error:
            detail = error.read().decode(errors="replace")
            raise RuntimeError(f"SerpApi request failed ({error.code}): {detail}") from error
        except URLError as error:
            raise RuntimeError(f"SerpApi request failed: {error.reason}") from error


def _flight_groups(payload: dict[str, Any]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for key in ("best_flights", "other_flights"):
        values = payload.get(key, [])
        if isinstance(values, list):
            groups.extend(value for value in values if isinstance(value, dict))
    return groups


def _request_url(base_url: str, params: dict[str, str]) -> str:
    parts = urlsplit(base_url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.update(params)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _parse_datetime(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("flight time must be string")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _raise_api_error(payload: dict[str, Any]) -> None:
    error = payload.get("error")
    if error:
        raise RuntimeError(f"SerpApi returned an error: {error}")


def _require_route(request: FlightRequest) -> None:
    if request.origin is None or request.destination is None:
        raise ValueError("origin and destination required before search")
    if request.departure_date is None:
        raise ValueError("departure_date required before search")
    # TODO: flexible dates; first slice requires exact departure/return dates.
