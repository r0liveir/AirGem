import unittest
from datetime import date, datetime

from airgem.agent import AgentState, FlightAgent
from airgem.core import FlightRequest
from airgem.search import FlightOption, FlightSearchResult


def option(
    candidate_id: str,
    price: float,
    hour: int,
    *,
    day: int = 6,
    origin: str = "GRU",
    destination: str = "POA",
    stops: int = 0,
    duration_minutes: int = 120,
) -> FlightOption:
    departure = datetime(2026, 11, day, hour)
    return FlightOption(
        provider="test",
        origin=origin,
        destination=destination,
        departure=departure,
        arrival=departure.replace(hour=hour + 2),
        price=price,
        currency="USD",
        stops=stops,
        airline="Test Air",
        duration_minutes=duration_minutes,
        candidate_id=candidate_id,
    )


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses

    def complete(self, *, system: str, user: str) -> str:
        return self.responses.pop(0)


class OneWaySearch:
    def search(self, request: FlightRequest) -> FlightSearchResult:
        return FlightSearchResult(
            outbound=(
                option("candidate-0", 100, 8),
                option("candidate-1", 120, 10, stops=1, duration_minutes=60),
            )
        )


class RoundTripSearch:
    def __init__(self) -> None:
        self.expanded_id = None

    def search(self, request: FlightRequest) -> FlightSearchResult:
        return FlightSearchResult(
            outbound=(
                option("candidate-0", 180, 8),
                option("candidate-1", 200, 10, stops=1, duration_minutes=60),
            ),
            round_trip=True,
            price_is_total=True,
        )

    def expand_round_trip(self, candidate_id: str) -> FlightSearchResult:
        self.expanded_id = candidate_id
        outbound_price = 180 if candidate_id == "candidate-0" else 200
        outbound_hour = 8 if candidate_id == "candidate-0" else 10
        return FlightSearchResult(
            outbound=(option(candidate_id, outbound_price, outbound_hour),),
            inbound=(
                option(f"{candidate_id}/return-0", 0, 18, day=8, origin="POA", destination="GRU"),
                option(
                    f"{candidate_id}/return-1",
                    0,
                    20,
                    day=8,
                    origin="POA",
                    destination="GRU",
                    stops=1,
                    duration_minutes=60,
                ),
            ),
            round_trip=True,
            price_is_total=True,
        )


class TestFlightAgent(unittest.TestCase):
    request = FlightRequest("GRU", "POA", date(2026, 11, 6), date(2026, 11, 8))

    def test_one_way_selects_candidate_without_expansion(self) -> None:
        agent = FlightAgent(
            FakeLLM(
                [
                    '{"action":"select_candidate","candidate_id":"candidate-1",'
                    '"reason":"It matches the requested faster trip."}'
                ]
            ),
            OneWaySearch(),
        )

        result = agent.run("fast flight", self.request)

        self.assertEqual(result.state, AgentState.COMPLETE)
        self.assertEqual(result.selected.id, "candidate-1")
        self.assertEqual(result.reason, "It matches the requested faster trip.")
        self.assertEqual(tuple(item.id for item in result.alternatives), ("candidate-0",))
        self.assertEqual(result.deferred_outbound, ())

    def test_round_trip_expands_selected_candidate_then_selects_return(self) -> None:
        search = RoundTripSearch()
        agent = FlightAgent(
            FakeLLM(
                [
                    '{"action":"select_candidate","candidate_id":"candidate-0"}',
                    '{"action":"select_return","candidate_id":"candidate-0/return-1"}',
                ]
            ),
            search,
        )

        result = agent.run("cheap but few stops", self.request)

        self.assertEqual(search.expanded_id, "candidate-0")
        self.assertEqual(result.selected.id, "candidate-0/return-1")
        self.assertEqual(result.selected.price, 180)
        self.assertEqual(tuple(item.id for item in result.alternatives), ("candidate-0/return-0",))
        self.assertEqual(tuple(item.id for item in result.deferred_outbound), ("candidate-1",))

    def test_session_expands_deferred_outbound_on_follow_up(self) -> None:
        search = RoundTripSearch()
        agent = FlightAgent(
            FakeLLM(
                [
                    '{"action":"select_candidate","candidate_id":"candidate-0"}',
                    '{"action":"select_return","candidate_id":"candidate-0/return-1"}',
                    '{"action":"select_deferred_outbound","candidate_id":"candidate-1"}',
                    '{"action":"select_return","candidate_id":"candidate-1/return-0"}',
                ]
            ),
            search,
        )

        session, first = agent.start("cheap flight", self.request)
        second = agent.continue_session(session, "try the other outbound")

        self.assertEqual(first.deferred_outbound[0].id, "candidate-1")
        self.assertEqual(search.expanded_id, "candidate-1")
        self.assertEqual(second.selected.id, "candidate-1/return-0")
        self.assertEqual(second.deferred_outbound, ())
        self.assertEqual(
            tuple(item.id for item in second.alternatives),
            ("candidate-0/return-1", "candidate-0/return-0", "candidate-1/return-1"),
        )

    def test_explicit_return_candidate_reuses_saved_expansion(self) -> None:
        search = RoundTripSearch()
        agent = FlightAgent(
            FakeLLM(
                [
                    '{"action":"select_candidate","candidate_id":"candidate-0"}',
                    '{"action":"select_return","candidate_id":"candidate-0/return-0"}',
                ]
            ),
            search,
        )

        session, _ = agent.start("cheap flight", self.request)
        result = agent.continue_session(session, "candidate-0, return-1 booking url")

        self.assertEqual(search.expanded_id, "candidate-0")
        self.assertEqual(result.selected.id, "candidate-0/return-1")

    def test_follow_up_model_can_choose_previously_expanded_outbound(self) -> None:
        search = RoundTripSearch()
        agent = FlightAgent(
            FakeLLM(
                [
                    '{"action":"select_candidate","candidate_id":"candidate-0"}',
                    '{"action":"select_return","candidate_id":"candidate-0/return-0"}',
                    '{"action":"select_deferred_outbound","candidate_id":"candidate-0"}',
                    '{"action":"select_return","candidate_id":"candidate-0/return-1"}',
                ]
            ),
            search,
        )

        session, _ = agent.start("cheap flight", self.request)
        result = agent.continue_session(session, "show me the outbound at 08:15")

        self.assertEqual(result.selected.id, "candidate-0/return-1")

    def test_invalid_action_falls_back_to_first_pareto_candidate(self) -> None:
        agent = FlightAgent(FakeLLM(["not json"]), OneWaySearch())

        result = agent.run("cheap flight", self.request)

        self.assertEqual(result.selected.id, "candidate-0")
        self.assertEqual(tuple(item.id for item in result.alternatives), ("candidate-1",))
