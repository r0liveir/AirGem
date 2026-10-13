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
        self._round_trip_candidates: dict[str, tuple[FlightOption, str, FlightRequest]] = {}
        if not self.api_key:
            raise ValueError("SERPAPI_API_KEY is required for SerpApiProvider")

    def search(self, request: FlightRequest) -> FlightSearchResult:
        _require_route(request)
        self._round_trip_candidates.clear()
        if request.return_date is not None:
            return self.search_round_trip(request)
        outbound = self.search_one_way(
            departure_id=request.origin,
            arrival_id=request.destination,
            outbound_date=request.departure_date,
            request=request,
        )
        return FlightSearchResult(outbound=outbound)

    def search_round_trip(self, request: FlightRequest) -> FlightSearchResult:
        _require_route(request)
        if request.return_date is None:
            raise ValueError("return_date required for round-trip search")
        self._round_trip_candidates.clear()
        params = self._base_params(
            departure_id=request.origin,
            arrival_id=request.destination,
            outbound_date=request.departure_date,
            request=request,
            flight_type="1",
        )
        params["return_date"] = request.return_date.isoformat()
        options = self._search_options(params, request, candidate_prefix="candidate", store_tokens=True)
        return FlightSearchResult(
            outbound=options,
            round_trip=True,
            price_is_total=True,
            notes=(
                "Native round-trip prices returned; select a candidate to expand return-leg details.",
            ),
        )

    def expand_round_trip(self, candidate_id: str) -> FlightSearchResult:
        """Expand one native round-trip candidate into its return-leg options."""
        try:
            outbound, departure_token, request = self._round_trip_candidates[candidate_id]
        except KeyError as error:
            raise ValueError(f"unknown round-trip candidate: {candidate_id}") from error

        params = {
            "engine": "google_flights",
            "api_key": self.api_key,
            "departure_token": departure_token,
        }
        if request.currency:
            params["currency"] = request.currency
        inbound = self._search_options(params, request, candidate_prefix=f"{candidate_id}/return")
        return FlightSearchResult(
            outbound=(outbound,),
            inbound=inbound,
            round_trip=True,
            price_is_total=True,
            notes=(f"Expanded return leg for {candidate_id}.",),
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
        params = self._base_params(
            departure_id=departure_id,
            arrival_id=arrival_id,
            outbound_date=outbound_date,
            request=request,
            flight_type="2",
        )
        return self._search_options(params, request, candidate_prefix="candidate")

    def _base_params(
        self,
        *,
        departure_id: str,
        arrival_id: str,
        outbound_date: date,
        request: FlightRequest,
        flight_type: str,
    ) -> dict[str, str]:
        params = {
            "engine": "google_flights",
            "api_key": self.api_key,
            "type": flight_type,
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
        return params

    def _search_options(
        self,
        params: dict[str, str],
        request: FlightRequest,
        candidate_prefix: str | None = None,
        store_tokens: bool = False,
    ) -> tuple[FlightOption, ...]:
        payload = self._fetch_json(params)
        _raise_api_error(payload)
        currency = str(payload.get("search_parameters", {}).get("currency", request.currency or "USD")).upper()
        options: list[FlightOption] = []
        for index, item in enumerate(_flight_groups(payload)):
            candidate_id = f"{candidate_prefix}-{index}" if candidate_prefix else None
            option = self._flight_option(item, currency, candidate_id=candidate_id)
            if option is None:
                continue
            options.append(option)
            if store_tokens:
                token = item.get("departure_token")
                if isinstance(token, str) and token:
                    self._round_trip_candidates[option.candidate_id or ""] = (option, token, request)
        return tuple(options)

    def _flight_option(
        self,
        item: dict[str, Any],
        currency: str,
        *,
        candidate_id: str | None = None,
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

        airlines = tuple(
            dict.fromkeys(
                str(flight["airline"])
                for flight in flights
                if isinstance(flight, dict) and flight.get("airline")
            )
        )
        duration_minutes = item.get("total_duration")
        if isinstance(duration_minutes, bool) or not isinstance(duration_minutes, int) or duration_minutes < 0:
            duration_minutes = None
        return FlightOption(
            provider=self.name,
            origin=origin,
            destination=destination,
            departure=departure,
            arrival=arrival,
            price=float(price),
            currency=currency,
            stops=max(0, len(flights) - 1),
            airline=", ".join(airlines) or None,
            duration_minutes=duration_minutes,
            candidate_id=candidate_id,
        )

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
