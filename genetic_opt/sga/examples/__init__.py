"""Example applications and tutorials for genetic_opt.

Available examples (run with ``python -m genetic_opt.sga.examples.<name>``):

quickstart
    Progressive tutorial: one-liner → class API → operators → constraints →
    callbacks → early stopping → benchmarks.

polynomial_fit
    Fit a polynomial curve to noisy data.

benchmark_suite
    Run standard test functions and compare results in a table.

constrained_optimization
    Minimise Himmelblau's function with inequality constraints.

high_dimensional
    Compare operator configurations on 20-dimensional problems.

engineering_design
    Pressure vessel design — a real-world constrained engineering problem.

parameter_tuning
    Tune PID controller gains for a second-order plant via simulation.

operator_comparison
    Grid search over crossover × selection operators on multiple benchmarks.

multi_start
    Multiple independent restarts for robust global search on Rastrigin.
"""

from genetic_opt.sga.examples.polynomial_fit import polynomial_fit_example

__all__ = [
    "polynomial_fit_example",
]
