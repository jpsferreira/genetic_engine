#!/usr/bin/env python
"""
Operator comparison: which crossover × selection works best?
=============================================================

Runs a grid of crossover and selection operators on several benchmark
problems and prints a performance matrix.  Useful for choosing the
right configuration for your problem type.

Usage::

    python -m genetic_opt.sga.examples.operator_comparison
    python -m genetic_opt.sga.examples.operator_comparison --ndim 10 --gens 300
"""

import argparse
import random
import time
from typing import Dict, List, Tuple

from genetic_opt.sga.api import minimize
from genetic_opt.sga.benchmarks import get_benchmark


CROSSOVER_TYPES = ["single_point", "uniform", "blx_alpha", "sbx"]
SELECTION_TYPES = ["tournament", "roulette", "rank"]
PROBLEMS = ["sphere", "rastrigin", "rosenbrock", "ackley"]


def run_comparison(
    ndim: int = 5,
    n_generations: int = 200,
    population_size: int = 100,
    n_runs: int = 3,
) -> Dict[str, Dict[Tuple[str, str], float]]:
    """Run the full grid and return results.

    Returns:
        ``{problem_name: {(crossover, selection): best_fitness, ...}, ...}``
    """
    results: Dict[str, Dict[Tuple[str, str], float]] = {}

    for prob_name in PROBLEMS:
        bench = get_benchmark(prob_name, ndim=ndim)
        results[prob_name] = {}

        for cx in CROSSOVER_TYPES:
            for sel in SELECTION_TYPES:
                best_f = float("inf")
                for run in range(n_runs):
                    random.seed(42 + run)
                    r = minimize(
                        bench["func"],
                        bench["bounds"],
                        n_generations=n_generations,
                        population_size=population_size,
                        crossover_type=cx,
                        selection_type=sel,
                    )
                    best_f = min(best_f, r.fun)
                results[prob_name][(cx, sel)] = best_f

    return results


def print_table(results: Dict[str, Dict[Tuple[str, str], float]]) -> None:
    """Pretty-print the results as one table per problem."""
    for prob_name, data in results.items():
        print(f"\n{'─' * 70}")
        print(f"  {prob_name.upper()}")
        print(f"{'─' * 70}")

        # Header
        sel_header = "".join(f"{sel:>14s}" for sel in SELECTION_TYPES)
        print(f"  {'crossover \\ selection':<20s}{sel_header}")
        print(f"  {'─' * 62}")

        # Find the overall best for highlighting
        best_val = min(data.values())

        for cx in CROSSOVER_TYPES:
            row = f"  {cx:<20s}"
            for sel in SELECTION_TYPES:
                val = data[(cx, sel)]
                marker = " *" if val == best_val else "  "
                row += f"{val:>12.4f}{marker}"
            print(row)

    print(f"\n{'─' * 70}")
    print("  * = best configuration for that problem")
    print(f"{'─' * 70}")


def print_recommendations(results: Dict[str, Dict[Tuple[str, str], float]]) -> None:
    """Print a recommendation per problem."""
    print("\nRecommendations:")
    for prob_name, data in results.items():
        best_key = min(data, key=data.get)
        cx, sel = best_key
        print(f"  {prob_name:<14s} → crossover={cx}, selection={sel}  "
              f"(f={data[best_key]:.4f})")

    # Overall winner: which config appears as best most often?
    from collections import Counter

    winners = Counter()
    for data in results.values():
        best_key = min(data, key=data.get)
        winners[best_key] += 1

    overall = winners.most_common(1)[0]
    cx, sel = overall[0]
    print(f"\n  Overall most-winning config: crossover={cx}, selection={sel} "
          f"(won {overall[1]}/{len(results)} problems)")


def main():
    parser = argparse.ArgumentParser(
        description="Compare GA operator configurations"
    )
    parser.add_argument("--ndim", type=int, default=5)
    parser.add_argument("--gens", type=int, default=200)
    parser.add_argument("--pop", type=int, default=100)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    print("GA Operator Comparison")
    print("=" * 50)
    print(f"Dimensions: {args.ndim}  |  Generations: {args.gens}  |  "
          f"Pop: {args.pop}  |  Runs: {args.runs}")
    print(f"Crossover types: {CROSSOVER_TYPES}")
    print(f"Selection types: {SELECTION_TYPES}")
    print(f"Problems: {PROBLEMS}")

    t0 = time.time()
    results = run_comparison(args.ndim, args.gens, args.pop, args.runs)
    elapsed = time.time() - t0

    print_table(results)
    print_recommendations(results)
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
