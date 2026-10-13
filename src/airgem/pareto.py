from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

Metric = tuple[float, ...]
T = TypeVar("T")


def pareto_front(items: Iterable[T], metrics: Callable[[T], Metric]) -> tuple[T, ...]:
    """Keep candidates not dominated across minimization metrics."""
    candidates = tuple(items)
    return tuple(
        candidate
        for candidate in candidates
        if not any(
            other != candidate and _dominates(metrics(other), metrics(candidate))
            for other in candidates
        )
    )


def bounded_pareto_front(
    items: Iterable[T],
    metrics: Callable[[T], Metric],
    *,
    limit: int,
) -> tuple[T, ...]:
    """Keep the frontier and fill it with ranked feasible alternatives."""
    if limit < 1:
        raise ValueError("candidate limit must be positive")
    candidates = tuple(items)
    frontier = pareto_front(candidates, metrics)
    if len(frontier) >= limit:
        return tuple(sorted(frontier, key=metrics)[:limit])

    ranked = sorted(candidates, key=metrics)
    extras = [candidate for candidate in ranked if candidate not in frontier]
    return frontier + tuple(extras[: limit - len(frontier)])


def _dominates(left: Metric, right: Metric) -> bool:
    return all(left_value <= right_value for left_value, right_value in zip(left, right)) and any(
        left_value < right_value for left_value, right_value in zip(left, right)
    )
