"""Constrained optimization example.

Minimise a function subject to inequality constraints using the
penalty-based approach built into :class:`SimpleGeneticAlgorithm`.

Problem (Himmelblau-style with constraints)::

    minimise  f(x) = (x1^2 + x2 - 11)^2 + (x1 + x2^2 - 7)^2
    subject to:
        g1(x) = (x1 - 5)^2 + x2^2 - 25 <= 0   (inside a circle)
        g2(x) = -x1 + 1 <= 0                    (x1 >= 1)

Usage::

    python -m genetic_opt.sga.examples.constrained_optimization
"""

import random
from typing import List

from genetic_opt.sga.api import minimize


def himmelblau(x: List[float]) -> float:
    """Himmelblau's function (4 local minima without constraints)."""
    x1, x2 = x[0], x[1]
    return (x1 ** 2 + x2 - 11) ** 2 + (x1 + x2 ** 2 - 7) ** 2


def constraint_circle(x: List[float]) -> float:
    """g1: must lie inside the circle centred at (5, 0) with radius 5."""
    return (x[0] - 5) ** 2 + x[1] ** 2 - 25


def constraint_x1_lower(x: List[float]) -> float:
    """g2: x1 >= 1."""
    return -x[0] + 1


def main():
    random.seed(42)

    print("Constrained Optimization Example")
    print("=" * 50)
    print("Minimise Himmelblau's function subject to:")
    print("  g1: (x1-5)^2 + x2^2 <= 25  (inside circle)")
    print("  g2: x1 >= 1")
    print()

    bounds = [(-5.0, 10.0), (-5.0, 10.0)]

    # --- Unconstrained run ---
    result_free = minimize(
        himmelblau,
        bounds,
        n_generations=200,
        population_size=100,
        crossover_type="sbx",
    )
    print(f"Unconstrained best: x={[round(v,4) for v in result_free.x]}, "
          f"f(x)={result_free.fun:.6f}")

    # --- Constrained run ---
    result_con = minimize(
        himmelblau,
        bounds,
        n_generations=200,
        population_size=100,
        crossover_type="sbx",
        constraints=[constraint_circle, constraint_x1_lower],
        penalty_weight=1000.0,
    )
    x = result_con.x
    g1 = constraint_circle(x)
    g2 = constraint_x1_lower(x)
    print(f"Constrained best:   x={[round(v,4) for v in x]}, "
          f"f(x)={result_con.fun:.6f}")
    print(f"  g1={(g1):.4f} {'(feasible)' if g1 <= 0 else '*** VIOLATED ***'}")
    print(f"  g2={(g2):.4f} {'(feasible)' if g2 <= 0 else '*** VIOLATED ***'}")
    print()
    print(f"Generations: {result_con.n_generations}  |  "
          f"Converged: {result_con.converged}")


if __name__ == "__main__":
    main()
