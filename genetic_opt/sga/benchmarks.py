"""Standard benchmark functions for evaluating optimisation algorithms.

Each function is defined as a **minimisation** problem.  The module also
exposes helper metadata (bounds, known optimum) so that examples and
tests can consume them uniformly.
"""

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple


@dataclass
class BenchmarkProblem:
    """Descriptor for a benchmark optimisation problem.

    Attributes:
        name: Human-readable name.
        func: Objective function ``f(x) -> float`` (to be minimised).
        bounds: Default ``[(lo, hi), ...]`` per dimension.
        optimum: Known global minimum value.
        optimal_x: One known minimiser (per dimension, repeated for any *n*).
        default_ndim: Suggested dimensionality for testing.
    """

    name: str
    func: Callable[[List[float]], float]
    bounds: Tuple[float, float]
    optimum: float
    optimal_x: float
    default_ndim: int = 10


# ---------------------------------------------------------------------------
# Objective functions
# ---------------------------------------------------------------------------


def sphere(x: List[float]) -> float:
    """Sphere function: ``f(x) = sum(xi^2)``.

    Simple, unimodal, separable.  Global minimum is 0 at the origin.
    """
    return sum(xi ** 2 for xi in x)


def rastrigin(x: List[float]) -> float:
    """Rastrigin function.

    Highly multimodal with many regularly-spaced local minima.
    Global minimum is 0 at the origin.
    """
    n = len(x)
    return 10 * n + sum(xi ** 2 - 10 * math.cos(2 * math.pi * xi) for xi in x)


def rosenbrock(x: List[float]) -> float:
    """Rosenbrock function (banana function).

    Unimodal but has a narrow curved valley that is hard to navigate.
    Global minimum is 0 at ``(1, 1, ..., 1)``.
    """
    return sum(
        100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
        for i in range(len(x) - 1)
    )


def ackley(x: List[float]) -> float:
    """Ackley function.

    Nearly flat outer region with a deep hole at the centre.
    Global minimum is 0 at the origin.
    """
    n = len(x)
    sum_sq = sum(xi ** 2 for xi in x) / n
    sum_cos = sum(math.cos(2 * math.pi * xi) for xi in x) / n
    return -20 * math.exp(-0.2 * math.sqrt(sum_sq)) - math.exp(sum_cos) + 20 + math.e


def griewank(x: List[float]) -> float:
    """Griewank function.

    Many widespread local minima; the product term creates a complex
    interaction structure.  Global minimum is 0 at the origin.
    """
    sum_sq = sum(xi ** 2 for xi in x) / 4000
    prod_cos = 1.0
    for i, xi in enumerate(x):
        prod_cos *= math.cos(xi / math.sqrt(i + 1))
    return sum_sq - prod_cos + 1


def schwefel(x: List[float]) -> float:
    """Schwefel function.

    The global minimum is geometrically distant from the next-best local
    minima, making it a good test for global search.
    Global minimum ≈ 0 at ``(420.9687, ...)``.
    """
    n = len(x)
    return 418.9829 * n - sum(xi * math.sin(math.sqrt(abs(xi))) for xi in x)


# ---------------------------------------------------------------------------
# Problem registry
# ---------------------------------------------------------------------------

BENCHMARKS = {
    "sphere": BenchmarkProblem(
        name="Sphere",
        func=sphere,
        bounds=(-5.12, 5.12),
        optimum=0.0,
        optimal_x=0.0,
    ),
    "rastrigin": BenchmarkProblem(
        name="Rastrigin",
        func=rastrigin,
        bounds=(-5.12, 5.12),
        optimum=0.0,
        optimal_x=0.0,
    ),
    "rosenbrock": BenchmarkProblem(
        name="Rosenbrock",
        func=rosenbrock,
        bounds=(-5.0, 10.0),
        optimum=0.0,
        optimal_x=1.0,
    ),
    "ackley": BenchmarkProblem(
        name="Ackley",
        func=ackley,
        bounds=(-5.0, 5.0),
        optimum=0.0,
        optimal_x=0.0,
    ),
    "griewank": BenchmarkProblem(
        name="Griewank",
        func=griewank,
        bounds=(-600.0, 600.0),
        optimum=0.0,
        optimal_x=0.0,
    ),
    "schwefel": BenchmarkProblem(
        name="Schwefel",
        func=schwefel,
        bounds=(-500.0, 500.0),
        optimum=0.0,
        optimal_x=420.9687,
        default_ndim=5,
    ),
}


def get_benchmark(name: str, ndim: Optional[int] = None) -> dict:
    """Return a ready-to-use dict for a benchmark problem.

    Args:
        name: Key in :data:`BENCHMARKS`.
        ndim: Number of dimensions (defaults to the benchmark's suggestion).

    Returns:
        ``{"func", "bounds", "ndim", "optimum", "name"}``
    """
    bp = BENCHMARKS[name]
    if ndim is None:
        ndim = bp.default_ndim
    return {
        "func": bp.func,
        "bounds": [(bp.bounds[0], bp.bounds[1])] * ndim,
        "ndim": ndim,
        "optimum": bp.optimum,
        "name": bp.name,
    }
