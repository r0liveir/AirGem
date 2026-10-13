import unittest

from airgem.pareto import bounded_pareto_front, pareto_front


class TestPareto(unittest.TestCase):
    def test_keeps_tradeoffs_and_drops_dominated_candidates(self) -> None:
        candidates = ((100, 300, 1), (120, 180, 0), (130, 360, 2))

        result = pareto_front(candidates, lambda item: tuple(float(value) for value in item))

        self.assertEqual(result, candidates[:2])

    def test_bounded_front_keeps_ranked_alternatives(self) -> None:
        candidates = ((100, 120), (110, 180), (120, 60))

        result = bounded_pareto_front(
            candidates,
            lambda item: tuple(float(value) for value in item),
            limit=3,
        )

        self.assertEqual(result, ((100, 120), (120, 60), (110, 180)))
