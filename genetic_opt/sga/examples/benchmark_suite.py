"""Run the full benchmark suite and print a comparison table.

Usage::

    python -m genetic_opt.sga.examples.benchmark_suite
    python scripts/run_benchmark_suite.py
"""

import random
import time
from typing import List

from genetic_opt.sga.api import minimize
from genetic_opt.sga.benchmarks import BENCHMARKS, get_benchmark


def run_benchmark_suite(
    ndim: int = 5,
    n_generations: int = 300,
    population_size: int = 150,
    n_runs: int = 3,
    verbose: bool = False,
):
    """Run every benchmark problem and print a summary table.

    Args:
        ndim: Number of dimensions for each problem.
        n_generations: Generations per run.
        population_size: Population size per run.
        n_runs: How many independent runs per benchmark (best is reported).
        verbose: Print per-run details.
    """
    header = (
        f"{'Problem':<14} {'Best Found':>12} {'Known Opt':>12} "
        f"{'Error':>12} {'Gens':>6} {'Time (s)':>10}"
    )
    print("=" * len(header))
    print("Genetic Algorithm Benchmark Suite")
    print(f"Dimensions: {ndim}  |  Generations: {n_generations}  |  "
          f"Pop size: {population_size}  |  Runs: {n_runs}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for key in BENCHMARKS:
        bench = get_benchmark(key, ndim=ndim)
        best_result = None

        for run in range(n_runs):
            random.seed(42 + run)
            t0 = time.time()
            result = minimize(
                bench["func"],
                bench["bounds"],
                n_generations=n_generations,
                population_size=population_size,
                crossover_type="sbx",
                adaptive_mutation=True,
                convergence_threshold=1e-8,
                convergence_generations=30,
            )
            elapsed = time.time() - t0

            if verbose:
                print(
                    f"  [{key} run {run+1}] best={result.fun:.6f} "
                    f"gens={result.n_generations} time={elapsed:.2f}s"
                )

            if best_result is None or result.fun < best_result.fun:
                best_result = result
                best_time = elapsed

        error = abs(best_result.fun - bench["optimum"])
        print(
            f"{bench['name']:<14} {best_result.fun:>12.6f} "
            f"{bench['optimum']:>12.6f} {error:>12.6f} "
            f"{best_result.n_generations:>6} {best_time:>10.2f}"
        )

    print("-" * len(header))
    print("Done.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run GA benchmark suite")
    parser.add_argument("--ndim", type=int, default=5, help="Number of dimensions")
    parser.add_argument("--gens", type=int, default=300, help="Generations per run")
    parser.add_argument("--pop", type=int, default=150, help="Population size")
    parser.add_argument("--runs", type=int, default=3, help="Independent runs per benchmark")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    run_benchmark_suite(
        ndim=args.ndim,
        n_generations=args.gens,
        population_size=args.pop,
        n_runs=args.runs,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
