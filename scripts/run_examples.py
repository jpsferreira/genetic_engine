#!/usr/bin/env python
"""
Main entry point for running genetic optimization examples.

Lists available examples and directs users to the specific entry points.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    print("Genetic Optimization Examples & Tutorials")
    print("==========================================")
    print()
    print("TUTORIALS")
    print("---------")
    print()
    print("  Quick-start tutorial (start here!)")
    print("    Walks through every feature: one-liner API, maximisation,")
    print("    operators, adaptive mutation, constraints, seeding,")
    print("    callbacks, early stopping, and benchmarks.")
    print("    Run: python -m genetic_opt.sga.examples.quickstart")
    print()
    print()
    print("EXAMPLES")
    print("--------")
    print()
    print("  1. Polynomial Fitting")
    print("     Fit a polynomial curve to noisy data with live visualisation.")
    print("     Run: python scripts/run_polynomial_example.py [options]")
    print("       --no-monitor     Disable live terminal monitor")
    print("       --enhanced-viz   Fitness landscapes, 3D migration, etc.")
    print()
    print("  2. Engineering Design (Pressure Vessel)")
    print("     Minimise fabrication cost subject to structural constraints.")
    print("     Run: python -m genetic_opt.sga.examples.engineering_design")
    print()
    print("  3. PID Controller Tuning")
    print("     Find optimal PID gains for a simulated second-order plant.")
    print("     Run: python -m genetic_opt.sga.examples.parameter_tuning")
    print()
    print("  4. Constrained Optimisation")
    print("     Minimise Himmelblau's function with inequality constraints.")
    print("     Run: python -m genetic_opt.sga.examples.constrained_optimization")
    print()
    print("  5. Multi-Start Optimisation")
    print("     Multiple restarts vs. single long run on Rastrigin.")
    print("     Run: python -m genetic_opt.sga.examples.multi_start")
    print()
    print()
    print("ANALYSIS TOOLS")
    print("--------------")
    print()
    print("  6. Benchmark Suite")
    print("     Run Sphere, Rastrigin, Rosenbrock, Ackley, Griewank, Schwefel.")
    print("     Run: python -m genetic_opt.sga.examples.benchmark_suite")
    print("       --ndim N    Dimensions (default 5)")
    print("       --gens N    Generations (default 300)")
    print("       --runs N    Runs per benchmark (default 3)")
    print()
    print("  7. Operator Comparison")
    print("     Grid search: crossover × selection on multiple benchmarks.")
    print("     Run: python -m genetic_opt.sga.examples.operator_comparison")
    print()
    print("  8. High-Dimensional Optimisation")
    print("     Compare configs on 20-dimensional Sphere/Rastrigin/Rosenbrock.")
    print("     Run: python -m genetic_opt.sga.examples.high_dimensional")
    print()
    print()
    print("QUICK START (from Python)")
    print("-------------------------")
    print()
    print("  from genetic_opt import minimize")
    print("  result = minimize(lambda x: sum(xi**2 for xi in x),")
    print("                    bounds=[(-5, 5)] * 10)")
    print("  print(result.x, result.fun)")


if __name__ == "__main__":
    main()
