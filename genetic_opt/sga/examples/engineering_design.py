#!/usr/bin/env python
"""
Engineering design optimisation: Pressure Vessel Design.
=========================================================

A classic constrained engineering optimisation benchmark.

**Problem** — Design a cylindrical pressure vessel capped by hemispherical
heads at both ends to minimise total fabrication cost.

Design variables::

    x1 = Ts  — shell thickness         (continuous, in multiples of 0.0625 in)
    x2 = Th  — head thickness          (continuous, in multiples of 0.0625 in)
    x3 = R   — inner radius            (continuous, 10–200 in)
    x4 = L   — length without heads    (continuous, 10–200 in)

Objective (fabrication cost)::

    f(x) = 0.6224·x1·x3·x4 + 1.7781·x2·x3² + 3.1661·x1²·x4 + 19.84·x1²·x3

Subject to::

    g1(x) = -x1 + 0.0193·x3           <= 0
    g2(x) = -x2 + 0.00954·x3          <= 0
    g3(x) = -π·x3²·x4 - (4/3)π·x3³ + 1296000  <= 0
    g4(x) = x4 - 240                   <= 0

Known best feasible solution ≈ $5868.76 (from literature).

Usage::

    python -m genetic_opt.sga.examples.engineering_design

Reference:
    Kannan & Kramer (1994), Coello (2000), various GA textbooks.
"""

import math
import random
from typing import List

from genetic_opt import minimize


# ──────────────────────────────────────────────────────────────
# Problem definition
# ──────────────────────────────────────────────────────────────

def cost(x: List[float]) -> float:
    """Total fabrication cost of the pressure vessel."""
    x1, x2, x3, x4 = x
    return (
        0.6224 * x1 * x3 * x4
        + 1.7781 * x2 * x3 ** 2
        + 3.1661 * x1 ** 2 * x4
        + 19.84 * x1 ** 2 * x3
    )


def g1(x: List[float]) -> float:
    """Shell thickness constraint."""
    return -x[0] + 0.0193 * x[2]


def g2(x: List[float]) -> float:
    """Head thickness constraint."""
    return -x[1] + 0.00954 * x[2]


def g3(x: List[float]) -> float:
    """Minimum volume constraint (must hold at least 1 296 000 in³)."""
    return -math.pi * x[2] ** 2 * x[3] - (4 / 3) * math.pi * x[2] ** 3 + 1_296_000


def g4(x: List[float]) -> float:
    """Maximum length constraint."""
    return x[3] - 240.0


# ──────────────────────────────────────────────────────────────
# Solve
# ──────────────────────────────────────────────────────────────

def main():
    random.seed(42)

    print("Pressure Vessel Design Optimisation")
    print("=" * 50)
    print()
    print("Variables: Ts (shell), Th (head), R (radius), L (length)")
    print("Objective: minimise fabrication cost")
    print("Constraints: 4 inequality constraints")
    print()

    bounds = [
        (0.0625, 6.1875),   # x1: shell thickness
        (0.0625, 6.1875),   # x2: head thickness
        (10.0, 200.0),       # x3: inner radius
        (10.0, 200.0),       # x4: cylinder length
    ]

    constraints = [g1, g2, g3, g4]

    # --- Run 1: default operators ---
    random.seed(42)
    r1 = minimize(
        cost,
        bounds,
        n_generations=500,
        population_size=200,
        constraints=constraints,
        penalty_weight=1e6,
        crossover_type="single_point",
        convergence_threshold=1e-4,
        convergence_generations=50,
    )

    # --- Run 2: SBX + adaptive mutation ---
    random.seed(42)
    r2 = minimize(
        cost,
        bounds,
        n_generations=500,
        population_size=200,
        constraints=constraints,
        penalty_weight=1e6,
        crossover_type="sbx",
        adaptive_mutation=True,
        convergence_threshold=1e-4,
        convergence_generations=50,
    )

    # --- Run 3: BLX-alpha + rank selection + seeding ---
    # Seed with a known feasible point
    seed = [[1.0, 0.5, 50.0, 150.0]]
    random.seed(42)
    r3 = minimize(
        cost,
        bounds,
        n_generations=500,
        population_size=200,
        constraints=constraints,
        penalty_weight=1e6,
        crossover_type="blx_alpha",
        selection_type="rank",
        adaptive_mutation=True,
        seed_population=seed,
        convergence_threshold=1e-4,
        convergence_generations=50,
    )

    print(f"{'Configuration':<38s} {'Cost ($)':>10} {'Gens':>6}  Feasible?")
    print("-" * 70)

    for label, r in [
        ("Single-point + tournament", r1),
        ("SBX + adaptive mutation", r2),
        ("BLX-alpha + rank + seeded", r3),
    ]:
        x = r.x
        feasible = all(g(x) <= 1e-3 for g in constraints)
        print(
            f"  {label:<36s} {r.fun:>10.2f} {r.n_generations:>6}  "
            f"{'Yes' if feasible else 'No'}"
        )

    # Print best solution details
    best = min([r1, r2, r3], key=lambda r: r.fun if all(g(r.x) <= 1e-3 for g in constraints) else float("inf"))
    x = best.x
    print()
    print("Best feasible solution:")
    print(f"  Ts (shell thickness) = {x[0]:.4f} in")
    print(f"  Th (head thickness)  = {x[1]:.4f} in")
    print(f"  R  (inner radius)    = {x[2]:.4f} in")
    print(f"  L  (cylinder length) = {x[3]:.4f} in")
    print(f"  Cost                 = ${best.fun:,.2f}")
    print()
    print("Constraint satisfaction:")
    for i, (name, g) in enumerate(
        [("g1 (shell)", g1), ("g2 (head)", g2), ("g3 (volume)", g3), ("g4 (length)", g4)]
    ):
        val = g(x)
        print(f"  {name}: {val:>10.4f}  {'OK' if val <= 1e-3 else '*** VIOLATED ***'}")

    print()
    print("(Literature best ≈ $5,868.76)")


if __name__ == "__main__":
    main()
