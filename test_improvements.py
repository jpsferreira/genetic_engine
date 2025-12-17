#!/usr/bin/env python
"""Quick test to verify all improvements work correctly."""

import sys
import random
import numpy as np

# Add current directory to path
sys.path.insert(0, ".")

from genetic_opt.sga.optimizer import SimpleGeneticAlgorithm


def test_basic_functionality():
    """Test basic functionality with improved code."""
    print("Testing basic functionality...")

    def simple_fitness(x):
        """Simple quadratic fitness function."""
        return -sum(xi**2 for xi in x)

    # Test with default parameters
    optimizer = SimpleGeneticAlgorithm(
        fitness_function=simple_fitness,
        population_size=20,
        mutation_rate=0.1,
        elite_size=2,
        verbose=False,
    )

    best_solution, best_fitness = optimizer.optimize(
        n_generations=10, chromosome_length=3, bounds=[(-5, 5)]
    )

    print(f"✓ Basic functionality works")
    print(f"  Best fitness: {best_fitness:.4f}")
    return True


def test_configurable_tournament_size():
    """Test configurable tournament size."""
    print("\nTesting configurable tournament size...")

    def simple_fitness(x):
        return -sum(xi**2 for xi in x)

    # Test with custom tournament size
    optimizer = SimpleGeneticAlgorithm(
        fitness_function=simple_fitness,
        population_size=20,
        tournament_size=5,  # Custom tournament size
        verbose=False,
    )

    assert optimizer.tournament_size == 5
    print(f"✓ Tournament size configurable: {optimizer.tournament_size}")

    best_solution, best_fitness = optimizer.optimize(
        n_generations=5, chromosome_length=2, bounds=[(-5, 5)]
    )

    print(f"✓ Optimization with custom tournament size works")
    return True


def test_parallel_evaluation():
    """Test parallel evaluation feature."""
    print("\nTesting parallel evaluation...")

    def simple_fitness(x):
        return -sum(xi**2 for xi in x)

    # Test with parallel evaluation enabled
    optimizer = SimpleGeneticAlgorithm(
        fitness_function=simple_fitness,
        population_size=20,
        parallel_evaluation=True,
        n_workers=2,
        verbose=False,
    )

    assert optimizer.parallel_evaluation is True
    assert optimizer.n_workers == 2
    print(f"✓ Parallel evaluation configured")

    best_solution, best_fitness = optimizer.optimize(
        n_generations=5, chromosome_length=2, bounds=[(-5, 5)]
    )

    print(f"✓ Parallel evaluation works")
    print(f"  Best fitness: {best_fitness:.4f}")
    return True


def test_bounds_helper():
    """Test bounds helper method."""
    print("\nTesting bounds helper method...")

    def simple_fitness(x):
        return -sum(xi**2 for xi in x)

    optimizer = SimpleGeneticAlgorithm(
        fitness_function=simple_fitness,
        population_size=10,
        verbose=False,
    )

    # Test _get_bounds with single bound
    bounds = [(-5, 5)]
    min_val, max_val = optimizer._get_bounds(0, bounds)
    assert min_val == -5 and max_val == 5

    # Test with multiple bounds
    bounds = [(-1, 1), (-10, 10), (-5, 5)]
    min_val, max_val = optimizer._get_bounds(4, bounds)  # Index 4 wraps to index 1
    assert min_val == -10 and max_val == 10

    print(f"✓ Bounds helper method works correctly")
    return True


def test_docstrings():
    """Test that all methods have docstrings."""
    print("\nTesting docstrings...")

    optimizer = SimpleGeneticAlgorithm(
        fitness_function=lambda x: sum(x),
        population_size=10,
        verbose=False,
    )

    methods_to_check = [
        "_tournament_selection",
        "_crossover",
        "_mutate",
        "_get_bounds",
        "_evaluate_population",
    ]

    for method_name in methods_to_check:
        method = getattr(optimizer, method_name)
        assert method.__doc__ is not None, f"{method_name} missing docstring"
        print(f"✓ {method_name} has docstring")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Genetic Optimizer Improvements")
    print("=" * 60)

    try:
        test_basic_functionality()
        test_configurable_tournament_size()
        test_parallel_evaluation()
        test_bounds_helper()
        test_docstrings()

        print("\n" + "=" * 60)
        print("✓ All improvements verified successfully!")
        print("=" * 60)
        sys.exit(0)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
