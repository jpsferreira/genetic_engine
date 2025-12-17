"""Genetic optimization algorithms."""

__version__ = "0.1.0"

# Import main classes for easier access
from genetic_opt.sga.optimizer import GeneticOptimizer, SimpleGeneticAlgorithm

__all__ = [
    "GeneticOptimizer",
    "SimpleGeneticAlgorithm",
]
