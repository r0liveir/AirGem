import unittest
from unittest.mock import patch

from fastapi import HTTPException
from starlette.requests import Request

from airgem.search import BookingAction
from airgem.web import (
    SessionEntry,
    _cleanup_expired,
    _client_owner,
    _enforce_rate_limit,
    _global_rate_events,
    _index_path,
    _public_error,
    _rate_events,
    _register_booking_action,
    _sessions,
    ui_booking,
)


def _request(host: str = "203.0.113.10") -> Request:
    return Request(
        {"type": "http", "method": "POST", "path": "/", "headers": [], "client": (host, 1234)}
    )


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def geturl(self):
        return "https://booking.example/flight"

    def read(self, limit):
        return (
            b'<html><head><meta content="0;url=\'https://airline.example/checkout\'" '
            b'http-equiv="refresh"></head></html>'
        )[:limit]


class TestWebBooking(unittest.TestCase):
    def test_booking_action_replays_post_data_and_redirects(self) -> None:
        request = _request()
        booking_id = _register_booking_action(
            BookingAction(
                url="https://booking.example/flight",
                post_data="flight=TA+100&currency=BRL",
            ),
            _client_owner(request),
        )

        with patch("airgem.web.urlopen", return_value=_FakeResponse()) as open_url:
            response = ui_booking(booking_id, request)

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "https://airline.example/checkout")
        request = open_url.call_args.args[0]
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.data, b"flight=TA+100&currency=BRL")

    def test_booking_action_is_isolated_by_client_ip(self) -> None:
        owner_request = _request("203.0.113.10")
        booking_id = _register_booking_action(
            BookingAction(url="https://booking.example/flight"),
            _client_owner(owner_request),
        )

        with self.assertRaises(HTTPException) as raised:
            ui_booking(booking_id, _request("203.0.113.11"))

        self.assertEqual(raised.exception.status_code, 404)


class TestDemoRateLimit(unittest.TestCase):
    def setUp(self) -> None:
        _rate_events.clear()
        _global_rate_events.clear()

    def test_limits_each_client_without_blocking_another_ip(self) -> None:
        with patch("airgem.web._RATE_LIMIT_PER_IP", 1), patch("airgem.web._RATE_LIMIT_GLOBAL", 10):
            _enforce_rate_limit("client-a", now=100)
            _enforce_rate_limit("client-b", now=100)

            with self.assertRaises(HTTPException) as raised:
                _enforce_rate_limit("client-a", now=101)

        self.assertEqual(raised.exception.status_code, 429)
        self.assertEqual(raised.exception.headers["Retry-After"], "3600")

    def test_expired_session_and_provider_key_are_released(self) -> None:
        provider = type("Provider", (), {"api_key": "private-serpapi-key"})()
        session = type("Session", (), {"agent": type("Agent", (), {"search": provider})()})()
        _sessions["expired"] = SessionEntry("client-a", session, expires_at=10)  # type: ignore[arg-type]

        _cleanup_expired(now=11)

        self.assertNotIn("expired", _sessions)

    def test_public_errors_do_not_echo_provider_secrets(self) -> None:
        message = _public_error(RuntimeError("request failed with private-serpapi-key"))

        self.assertNotIn("private-serpapi-key", message)

    def test_public_errors_identify_the_failed_provider_stage(self) -> None:
        self.assertIn(
            "OpenRouter",
            _public_error(RuntimeError("private detail"), stage="parsing_request"),
        )
        self.assertIn(
            "SerpApi",
            _public_error(RuntimeError("private detail"), stage="searching"),
        )


class TestWebPage(unittest.TestCase):
    def test_web_uses_server_configured_llm_provider(self) -> None:
        page = _index_path.read_text(encoding="utf-8")

        self.assertNotIn('name="provider"', page)

    def test_serpapi_key_is_cleared_only_after_success(self) -> None:
        page = _index_path.read_text(encoding="utf-8")

        self.assertIn('payload.status === "complete"', page)
        self.assertNotIn('document.addEventListener("htmx:afterRequest"', page)

    def test_about_section_explains_project_and_demo_protections(self) -> None:
        page = _index_path.read_text(encoding="utf-8")

        self.assertIn('id="about"', page)
        self.assertIn("Bounded AI", page)
        self.assertIn("Transient by design", page)
        self.assertIn("Demo protections", page)
