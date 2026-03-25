"""Genetic optimization algorithms."""

__version__ = "0.2.0"

from genetic_opt.sga.optimizer import GeneticOptimizer, SimpleGeneticAlgorithm
from genetic_opt.sga.api import minimize, maximize, OptimizationResult
from genetic_opt.sga.benchmarks import BENCHMARKS, get_benchmark
from genetic_opt.sga.operators import (
    CROSSOVER_OPERATORS,
    SELECTION_OPERATORS,
)

__all__ = [
    "GeneticOptimizer",
    "SimpleGeneticAlgorithm",
    "minimize",
    "maximize",
    "OptimizationResult",
    "BENCHMARKS",
    "get_benchmark",
    "CROSSOVER_OPERATORS",
    "SELECTION_OPERATORS",
]
