#!/usr/bin/env python
"""
Generate *The Evolution Chronicle* — an interactive, biologically-inspired
visualisation of how a genetic algorithm evolves.

This script either:
  1. Uses a population CSV from a previous run, or
  2. Runs a fresh optimisation on a benchmark and then visualises it.

Usage::

    # Run a demo on Rastrigin (generates data + visualisation)
    python -m genetic_opt.sga.examples.evolution_chronicle

    # Visualise an existing population CSV
    python -m genetic_opt.sga.examples.evolution_chronicle \\
        --population results/.../population.csv \\
        --metrics results/.../metrics.csv

    # Choose benchmark and dimensions
    python -m genetic_opt.sga.examples.evolution_chronicle \\
        --benchmark rastrigin --ndim 5 --gens 150
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np

from genetic_opt.sga.api import minimize
from genetic_opt.sga.benchmarks import BENCHMARKS, get_benchmark
from genetic_opt.sga.utils.bio_visualization import create_evolution_chronicle


def run_demo(benchmark: str, ndim: int, n_generations: int) -> dict:
    """Run a GA on a benchmark problem and return file paths."""
    random.seed(42)
    np.random.seed(42)

    bench = get_benchmark(benchmark, ndim=ndim)

    print(f"Running {bench['name']} ({ndim}D) for {n_generations} generations...")
    result = minimize(
        bench["func"],
        bench["bounds"],
        n_generations=n_generations,
        population_size=max(80, 12 * ndim),
        crossover_type="sbx",
        selection_type="tournament",
        adaptive_mutation=True,
        track_history=True,
        export_data=True,
        verbose=True,
    )

    print(f"\nBest f(x) = {result.fun:.6f}  (optimum = {bench['optimum']})")

    opt = result.optimizer
    return {
        "population_file": opt.export_paths.get("population_history"),
        "metrics_file": opt.export_paths.get("metrics"),
        "fitness_function": bench["func"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate The Evolution Chronicle visualisation"
    )
    parser.add_argument(
        "--population", type=str, default=None,
        help="Path to population history CSV from a previous run",
    )
    parser.add_argument(
        "--metrics", type=str, default=None,
        help="Path to metrics CSV from a previous run",
    )
    parser.add_argument(
        "--benchmark", type=str, default="rastrigin",
        choices=list(BENCHMARKS.keys()),
        help="Benchmark to run if no population file given (default: rastrigin)",
    )
    parser.add_argument("--ndim", type=int, default=5)
    parser.add_argument("--gens", type=int, default=120)
    parser.add_argument(
        "-o", "--output", type=str, default="evolution_chronicle.html",
        help="Output HTML file",
    )
    args = parser.parse_args()

    if args.population:
        # Use existing data
        pop_file = args.population
        met_file = args.metrics
        # Try to match a benchmark for fitness colouring
        bench = get_benchmark(args.benchmark)
        fitness_fn = bench["func"]
        print(f"Using existing data: {pop_file}")
    else:
        # Run a fresh demo
        paths = run_demo(args.benchmark, args.ndim, args.gens)
        pop_file = paths["population_file"]
        met_file = paths["metrics_file"]
        fitness_fn = paths["fitness_function"]

    if not pop_file or not Path(pop_file).exists():
        print("Error: no population file found. Run with --population or let the demo generate one.")
        sys.exit(1)

    print(f"\nGenerating The Evolution Chronicle...")
    output = create_evolution_chronicle(
        population_file=pop_file,
        output_file=args.output,
        metrics_file=met_file,
        fitness_function=fitness_fn,
    )
    print(f"Saved to: {output}")
    print("Open the HTML file in a browser to explore the interactive visualisation.")


if __name__ == "__main__":
    main()
