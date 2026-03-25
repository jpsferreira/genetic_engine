"""High-dimensional optimization with adaptive mutation.

Demonstrates that the GA can handle 20+ dimensional problems when
adaptive mutation and SBX crossover are enabled.

Usage::

    python -m genetic_opt.sga.examples.high_dimensional
"""

import random
import time
from typing import List

from genetic_opt.sga.api import minimize
from genetic_opt.sga.benchmarks import get_benchmark


def main():
    random.seed(42)

    ndim = 20
    print("High-Dimensional Optimization Example")
    print("=" * 55)
    print(f"Dimensions: {ndim}")
    print()

    configs = [
        {
            "label": "Default (single-point, fixed mutation)",
            "kwargs": {},
        },
        {
            "label": "SBX crossover + adaptive mutation",
            "kwargs": {"crossover_type": "sbx", "adaptive_mutation": True},
        },
        {
            "label": "BLX-alpha crossover + rank selection",
            "kwargs": {
                "crossover_type": "blx_alpha",
                "selection_type": "rank",
                "adaptive_mutation": True,
            },
        },
    ]

    for bench_name in ["sphere", "rastrigin", "rosenbrock"]:
        bench = get_benchmark(bench_name, ndim=ndim)
        print(f"\n--- {bench['name']} (optimum = {bench['optimum']}) ---")
        print(f"  {'Configuration':<42} {'Best f(x)':>12} {'Gens':>6} {'Time':>8}")

        for cfg in configs:
            random.seed(42)
            t0 = time.time()
            result = minimize(
                bench["func"],
                bench["bounds"],
                n_generations=500,
                population_size=max(100, 15 * ndim),
                convergence_threshold=1e-8,
                convergence_generations=40,
                **cfg["kwargs"],
            )
            elapsed = time.time() - t0
            print(
                f"  {cfg['label']:<42} {result.fun:>12.4f} "
                f"{result.n_generations:>6} {elapsed:>7.2f}s"
            )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
