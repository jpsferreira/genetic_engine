"""Genetic optimization library."""

from genetic_opt.sga.optimizer import GeneticOptimizer, SimpleGeneticAlgorithm
from genetic_opt.sga.api import minimize, maximize, OptimizationResult
from genetic_opt.sga.benchmarks import BENCHMARKS, get_benchmark

__all__ = [
    "GeneticOptimizer",
    "SimpleGeneticAlgorithm",
    "minimize",
    "maximize",
    "OptimizationResult",
    "BENCHMARKS",
    "get_benchmark",
]
