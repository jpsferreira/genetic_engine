#!/usr/bin/env python
"""
Multi-start optimisation for robust global search.
====================================================

When the landscape has many local minima, a single GA run may get
trapped.  Running several independent restarts and keeping the best
result is a simple but effective strategy.

This example wraps ``minimize()`` in a multi-start loop and
demonstrates the improvement on the Rastrigin function (many local
minima) vs. a single long run.

Usage::

    python -m genetic_opt.sga.examples.multi_start
"""

import random
import time
from typing import List, Optional

from genetic_opt.sga.api import minimize, OptimizationResult
from genetic_opt.sga.benchmarks import get_benchmark


def multi_start_minimize(
    func,
    bounds,
    n_restarts: int = 5,
    n_generations: int = 200,
    seed: int = 42,
    verbose: bool = True,
    **kwargs,
) -> OptimizationResult:
    """Run ``minimize`` multiple times and return the best result.

    Args:
        func: Objective function to minimise.
        bounds: ``[(lo, hi), ...]`` per dimension.
        n_restarts: Number of independent runs.
        n_generations: Generations per run.
        seed: Base random seed (incremented per restart).
        verbose: Print per-run progress.
        **kwargs: Forwarded to ``minimize()``.

    Returns:
        The best :class:`OptimizationResult` across all restarts.
    """
    best: Optional[OptimizationResult] = None

    for i in range(n_restarts):
        random.seed(seed + i)
        result = minimize(func, bounds, n_generations=n_generations, **kwargs)

        if verbose:
            tag = " <-- new best" if (best is None or result.fun < best.fun) else ""
            print(
                f"  Restart {i + 1:>2}/{n_restarts}  "
                f"f(x) = {result.fun:12.6f}  "
                f"gens = {result.n_generations:>4}{tag}"
            )

        if best is None or result.fun < best.fun:
            best = result

    return best


# ──────────────────────────────────────────────────────────────


def main():
    print("Multi-Start Optimisation")
    print("=" * 55)

    ndim = 10
    bench = get_benchmark("rastrigin", ndim=ndim)
    bounds = bench["bounds"]

    print(f"\nProblem: Rastrigin  |  Dimensions: {ndim}  |  Optimum: {bench['optimum']}")

    # --- Strategy A: single long run ---
    print(f"\n--- Strategy A: single run, 1000 generations ---")
    random.seed(42)
    t0 = time.time()
    r_single = minimize(
        bench["func"], bounds,
        n_generations=1000,
        population_size=150,
        crossover_type="sbx",
        adaptive_mutation=True,
        convergence_threshold=1e-8,
        convergence_generations=50,
    )
    t_single = time.time() - t0
    print(f"  f(x) = {r_single.fun:.6f}  |  gens = {r_single.n_generations}  |  time = {t_single:.2f}s")

    # --- Strategy B: 5 restarts × 200 generations each ---
    print(f"\n--- Strategy B: 5 restarts × 200 generations ---")
    t0 = time.time()
    r_multi = multi_start_minimize(
        bench["func"], bounds,
        n_restarts=5,
        n_generations=200,
        population_size=150,
        crossover_type="sbx",
        adaptive_mutation=True,
    )
    t_multi = time.time() - t0
    print(f"\n  Best f(x) = {r_multi.fun:.6f}  |  time = {t_multi:.2f}s")

    # --- Strategy C: 10 restarts × 100 generations each ---
    print(f"\n--- Strategy C: 10 restarts × 100 generations ---")
    t0 = time.time()
    r_many = multi_start_minimize(
        bench["func"], bounds,
        n_restarts=10,
        n_generations=100,
        population_size=150,
        crossover_type="sbx",
        adaptive_mutation=True,
    )
    t_many = time.time() - t0
    print(f"\n  Best f(x) = {r_many.fun:.6f}  |  time = {t_many:.2f}s")

    # --- Summary ---
    print(f"\n{'─' * 55}")
    print(f"  {'Strategy':<30s} {'f(x)':>12}  {'Time':>8}")
    print(f"  {'─' * 52}")
    print(f"  {'A: 1×1000 gens':<30s} {r_single.fun:>12.6f}  {t_single:>7.2f}s")
    print(f"  {'B: 5×200 gens':<30s} {r_multi.fun:>12.6f}  {t_multi:>7.2f}s")
    print(f"  {'C: 10×100 gens':<30s} {r_many.fun:>12.6f}  {t_many:>7.2f}s")
    print(f"  {'─' * 52}")
    print()
    print("  Multi-start trades depth for breadth — useful when the")
    print("  landscape has many local minima (like Rastrigin).")


if __name__ == "__main__":
    main()
