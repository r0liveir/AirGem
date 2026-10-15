import unittest
from unittest.mock import patch

from airgem.search import BookingAction
from airgem.web import _register_booking_action, ui_booking


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def geturl(self):
        return "https://booking.example/flight/confirmed"


class TestWebBooking(unittest.TestCase):
    def test_booking_action_replays_post_data_and_redirects(self) -> None:
        booking_id = _register_booking_action(
            BookingAction(
                url="https://booking.example/flight",
                post_data="flight=TA+100&currency=BRL",
            )
        )

        with patch("airgem.web.urlopen", return_value=_FakeResponse()) as open_url:
            response = ui_booking(booking_id)

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "https://booking.example/flight/confirmed")
        request = open_url.call_args.args[0]
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.data, b"flight=TA+100&currency=BRL")

