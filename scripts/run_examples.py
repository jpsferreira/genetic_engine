#!/usr/bin/env python
"""
Main entry point for running genetic optimization examples.

Lists available examples and directs users to the specific entry points.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    print("Genetic Optimization Examples")
    print("=============================")
    print("Available examples:")
    print()
    print("1. Polynomial Fitting")
    print("   Fit a polynomial curve using genetic optimization.")
    print("   Run: python scripts/run_polynomial_example.py [options]")
    print("     --no-monitor     Disable live terminal monitor")
    print("     --no-track       Disable population tracking")
    print("     --no-analysis    Disable migration analysis")
    print("     --enhanced-viz   Enable fitness landscapes, 3D migration, etc.")
    print()
    print("2. Benchmark Suite")
    print("   Run standard test functions (Sphere, Rastrigin, Rosenbrock, ...).")
    print("   Run: python -m genetic_opt.sga.examples.benchmark_suite [options]")
    print("     --ndim N         Number of dimensions (default: 5)")
    print("     --gens N         Generations per run (default: 300)")
    print("     --runs N         Independent runs per benchmark (default: 3)")
    print("     -v               Verbose per-run output")
    print()
    print("3. Constrained Optimization")
    print("   Minimise Himmelblau's function with inequality constraints.")
    print("   Run: python -m genetic_opt.sga.examples.constrained_optimization")
    print()
    print("4. High-Dimensional Optimization")
    print("   Compare operator configurations on 20-dimensional problems.")
    print("   Run: python -m genetic_opt.sga.examples.high_dimensional")
    print()
    print("Quick start (from Python):")
    print("  from genetic_opt import minimize")
    print("  result = minimize(lambda x: sum(xi**2 for xi in x),")
    print("                    bounds=[(-5, 5)] * 10)")
    print("  print(result.x, result.fun)")


if __name__ == "__main__":
    main()
