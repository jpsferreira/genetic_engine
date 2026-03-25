#!/usr/bin/env python
"""
Quick-start tutorial for genetic_opt.
======================================

This script walks through the library from the simplest possible usage
to advanced features.  Run it directly or step through the numbered
sections in an interactive session.

Usage::

    python -m genetic_opt.sga.examples.quickstart
"""

import random
import math
from typing import List


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# ──────────────────────────────────────────────────────────────
# 1.  One-liner with minimize()
# ──────────────────────────────────────────────────────────────

def tutorial_1_one_liner():
    """The simplest way to run an optimisation."""
    section("1. One-liner minimisation")

    from genetic_opt import minimize

    # Minimise the sphere function in 5 dimensions
    result = minimize(
        func=lambda x: sum(xi ** 2 for xi in x),
        bounds=[(-5, 5)] * 5,
    )

    print(f"Best solution : {[round(v, 4) for v in result.x]}")
    print(f"Best value    : {result.fun:.6f}")
    print(f"Generations   : {result.n_generations}")
    print(f"Converged     : {result.converged}")

    # The result object also carries the full metrics history
    print(f"Final avg fit : {result.metrics['avg_fitness'][-1]:.6f}")


# ──────────────────────────────────────────────────────────────
# 2.  Maximisation
# ──────────────────────────────────────────────────────────────

def tutorial_2_maximize():
    """Maximise a function instead of minimising it."""
    section("2. Maximisation")

    from genetic_opt import maximize

    # Find the peak of a 2D Gaussian
    def neg_gaussian(x: List[float]) -> float:
        return math.exp(-(x[0] - 3) ** 2 - (x[1] + 1) ** 2)

    result = maximize(neg_gaussian, bounds=[(-10, 10)] * 2)

    print(f"Peak location : ({result.x[0]:.4f}, {result.x[1]:.4f})")
    print(f"Peak value    : {result.fun:.6f}")
    print(f"(Expected ≈ (3, -1) with value ≈ 1.0)")


# ──────────────────────────────────────────────────────────────
# 3.  Using the class API for more control
# ──────────────────────────────────────────────────────────────

def tutorial_3_class_api():
    """Use SimpleGeneticAlgorithm directly for full control."""
    section("3. Class API with verbose output")

    from genetic_opt import SimpleGeneticAlgorithm

    def rastrigin(x: List[float]) -> float:
        n = len(x)
        return 10 * n + sum(xi ** 2 - 10 * math.cos(2 * math.pi * xi) for xi in x)

    ga = SimpleGeneticAlgorithm(
        fitness_function=rastrigin,
        population_size=120,
        sense="minimize",
        mutation_rate=0.15,
        crossover_type="sbx",          # Simulated Binary Crossover
        selection_type="tournament",
        tournament_size=4,
        elite_size=12,
        verbose=True,                   # Print a table each generation
        export_data=False,
    )

    best_x, best_f = ga.optimize(
        n_generations=30,               # Short run for the tutorial
        chromosome_length=3,
        bounds=[(-5.12, 5.12)] * 3,
    )

    print(f"\nBest f(x) = {best_f:.6f}")
    print(f"x = {[round(v, 4) for v in best_x]}")


# ──────────────────────────────────────────────────────────────
# 4.  Choosing crossover and selection operators
# ──────────────────────────────────────────────────────────────

def tutorial_4_operators():
    """Swap crossover and selection strategies."""
    section("4. Trying different operators")

    from genetic_opt import minimize

    def rosenbrock(x: List[float]) -> float:
        return sum(
            100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
            for i in range(len(x) - 1)
        )

    configs = {
        "single_point + tournament": dict(crossover_type="single_point", selection_type="tournament"),
        "uniform + rank":           dict(crossover_type="uniform",      selection_type="rank"),
        "sbx + roulette":           dict(crossover_type="sbx",          selection_type="roulette"),
        "blx_alpha + tournament":   dict(crossover_type="blx_alpha",    selection_type="tournament"),
    }

    bounds = [(-5, 10)] * 4

    for label, kw in configs.items():
        random.seed(42)
        result = minimize(rosenbrock, bounds, n_generations=200, **kw)
        print(f"  {label:<35s}  f(x) = {result.fun:12.4f}")

    print("\n(Lower is better — Rosenbrock optimum = 0.0)")


# ──────────────────────────────────────────────────────────────
# 5.  Adaptive mutation
# ──────────────────────────────────────────────────────────────

def tutorial_5_adaptive_mutation():
    """Let the GA auto-tune its mutation rate."""
    section("5. Adaptive mutation")

    from genetic_opt import minimize
    from genetic_opt.sga.benchmarks import ackley

    # Fixed mutation
    random.seed(42)
    r_fixed = minimize(
        ackley,
        bounds=[(-5, 5)] * 10,
        n_generations=300,
        adaptive_mutation=False,
    )

    # Adaptive mutation
    random.seed(42)
    r_adaptive = minimize(
        ackley,
        bounds=[(-5, 5)] * 10,
        n_generations=300,
        adaptive_mutation=True,
    )

    print(f"  Fixed mutation    : f(x) = {r_fixed.fun:.6f}  ({r_fixed.n_generations} gens)")
    print(f"  Adaptive mutation : f(x) = {r_adaptive.fun:.6f}  ({r_adaptive.n_generations} gens)")
    print(f"\n  (Ackley optimum = 0.0)")


# ──────────────────────────────────────────────────────────────
# 6.  Constraints
# ──────────────────────────────────────────────────────────────

def tutorial_6_constraints():
    """Add inequality constraints to the problem."""
    section("6. Constrained optimisation")

    from genetic_opt import minimize

    # Minimise x0 + x1 subject to x0 >= 2 and x0^2 + x1^2 <= 25
    def objective(x):
        return x[0] + x[1]

    constraints = [
        lambda x: 2.0 - x[0],            # x0 >= 2
        lambda x: x[0]**2 + x[1]**2 - 25,  # x0^2 + x1^2 <= 25
    ]

    random.seed(42)
    result = minimize(
        objective,
        bounds=[(-6, 6)] * 2,
        n_generations=200,
        constraints=constraints,
        penalty_weight=1000.0,
    )

    g1 = 2.0 - result.x[0]
    g2 = result.x[0]**2 + result.x[1]**2 - 25
    print(f"  x = ({result.x[0]:.4f}, {result.x[1]:.4f})")
    print(f"  f(x) = {result.fun:.4f}")
    print(f"  g1 = {g1:.4f}  {'OK' if g1 <= 1e-3 else 'VIOLATED'}")
    print(f"  g2 = {g2:.4f}  {'OK' if g2 <= 1e-3 else 'VIOLATED'}")


# ──────────────────────────────────────────────────────────────
# 7.  Seeding the initial population
# ──────────────────────────────────────────────────────────────

def tutorial_7_seeding():
    """Inject known-good solutions to warm-start the GA."""
    section("7. Population seeding")

    from genetic_opt import minimize
    from genetic_opt.sga.benchmarks import sphere

    # Without seed — starts from scratch
    random.seed(42)
    r_cold = minimize(sphere, bounds=[(-5, 5)] * 5, n_generations=100)

    # With seed — we already know the answer is near the origin
    random.seed(42)
    r_warm = minimize(
        sphere,
        bounds=[(-5, 5)] * 5,
        n_generations=100,
        seed_population=[
            [0.5, -0.3, 0.1, -0.2, 0.4],
            [-0.1, 0.2, -0.1, 0.3, -0.2],
        ],
    )

    print(f"  Cold start : f(x) = {r_cold.fun:.6f}")
    print(f"  Warm start : f(x) = {r_warm.fun:.6f}")
    print(f"\n  Seeding helps the GA converge faster when you have prior knowledge.")


# ──────────────────────────────────────────────────────────────
# 8.  Callbacks for custom logging
# ──────────────────────────────────────────────────────────────

def tutorial_8_callbacks():
    """Use callbacks to monitor progress programmatically."""
    section("8. Callbacks")

    from genetic_opt import minimize
    from genetic_opt.sga.benchmarks import sphere

    best_per_gen = []

    def track_best(generation, population, fitness_scores, metrics):
        best_per_gen.append(min(fitness_scores))

    random.seed(42)
    result = minimize(
        sphere,
        bounds=[(-5, 5)] * 5,
        n_generations=50,
        callbacks=[track_best],
    )

    # Show improvement over first 10 generations
    print("  Generation | Best fitness")
    print("  -----------+-------------")
    for i in range(min(10, len(best_per_gen))):
        print(f"  {i:>9}  | {best_per_gen[i]:.6f}")
    print(f"  {'...':>9}  |")
    print(f"  {len(best_per_gen)-1:>9}  | {best_per_gen[-1]:.6f}")


# ──────────────────────────────────────────────────────────────
# 9.  Early stopping
# ──────────────────────────────────────────────────────────────

def tutorial_9_early_stopping():
    """Stop automatically when the fitness plateaus."""
    section("9. Early stopping")

    from genetic_opt import minimize
    from genetic_opt.sga.benchmarks import sphere

    random.seed(42)
    result = minimize(
        sphere,
        bounds=[(-5, 5)] * 5,
        n_generations=1000,  # Would be overkill, but early stopping kicks in
        convergence_threshold=1e-6,
        convergence_generations=20,
    )

    print(f"  Requested 1000 generations")
    print(f"  Actually ran {result.n_generations} generations")
    print(f"  Converged: {result.converged}")
    print(f"  f(x) = {result.fun:.8f}")


# ──────────────────────────────────────────────────────────────
# 10. Using benchmark functions
# ──────────────────────────────────────────────────────────────

def tutorial_10_benchmarks():
    """Use the built-in benchmark suite to evaluate configurations."""
    section("10. Built-in benchmarks")

    from genetic_opt import minimize, get_benchmark, BENCHMARKS

    print(f"  Available benchmarks: {list(BENCHMARKS.keys())}\n")

    for name in ["sphere", "rastrigin", "ackley"]:
        bench = get_benchmark(name, ndim=5)
        random.seed(42)
        result = minimize(
            bench["func"],
            bench["bounds"],
            n_generations=200,
            crossover_type="sbx",
            adaptive_mutation=True,
        )
        error = abs(result.fun - bench["optimum"])
        print(f"  {bench['name']:<12s}  f(x) = {result.fun:10.4f}  "
              f"error = {error:.4f}  (optimum = {bench['optimum']})")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    print("╔════════════════════════════════════════════════════════╗")
    print("║       genetic_opt  —  Quick-Start Tutorial            ║")
    print("╚════════════════════════════════════════════════════════╝")

    tutorial_1_one_liner()
    tutorial_2_maximize()
    tutorial_3_class_api()
    tutorial_4_operators()
    tutorial_5_adaptive_mutation()
    tutorial_6_constraints()
    tutorial_7_seeding()
    tutorial_8_callbacks()
    tutorial_9_early_stopping()
    tutorial_10_benchmarks()

    section("Done!")
    print("For more examples see:")
    print("  python -m genetic_opt.sga.examples.benchmark_suite")
    print("  python -m genetic_opt.sga.examples.constrained_optimization")
    print("  python -m genetic_opt.sga.examples.high_dimensional")
    print("  python -m genetic_opt.sga.examples.engineering_design")
    print("  python -m genetic_opt.sga.examples.parameter_tuning")
    print("  python -m genetic_opt.sga.examples.operator_comparison")


if __name__ == "__main__":
    main()
