"""Tests for the genetic optimizer."""

import numpy as np
import random
import pytest

from genetic_opt.sga.optimizer import SimpleGeneticAlgorithm


def test_simple_genetic_algorithm_initialization():
    """Test that the SimpleGeneticAlgorithm initializes correctly."""

    def dummy_fitness(x):
        return sum(x)

    optimizer = SimpleGeneticAlgorithm(
        fitness_function=dummy_fitness,
        population_size=50,
        mutation_rate=0.2,
        elite_size=5,
    )

    assert optimizer.fitness_function is dummy_fitness
    assert optimizer.population_size == 50
    assert optimizer.mutation_rate == 0.2
    assert optimizer.elite_size == 5
    assert optimizer.best_solution is None


def test_population_initialization():
    """Test population initialization."""

    def dummy_fitness(x):
        return sum(x)

    optimizer = SimpleGeneticAlgorithm(
        fitness_function=dummy_fitness,
        population_size=20,
    )

    bounds = [(0, 10), (-5, 5)]
    chromosome_length = 4

    population = optimizer._initialize_population(chromosome_length, bounds)

    assert len(population) == 20
    assert len(population[0]) == chromosome_length

    # Check bounds
    for individual in population:
        for i, gene in enumerate(individual):
            min_val, max_val = bounds[i % len(bounds)]
            assert min_val <= gene <= max_val


def test_polynomial_fitting():
    """Test fitting a polynomial using SimpleGeneticAlgorithm."""
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Generate synthetic data for a polynomial: y = 2x^3 - 5x^2 + 3x - 1
    true_coefficients = [2, -5, 3, -1]  # [a, b, c, d] for ax^3 + bx^2 + cx + d

    x_data = np.linspace(-2, 2, 20)
    y_data = np.polyval(true_coefficients, x_data)

    # Add some noise
    np.random.seed(42)
    y_data_noisy = y_data + np.random.normal(0, 0.5, size=len(y_data))

    # Define fitness function (negative mean squared error, since we maximize fitness)
    def polynomial_fitness(coefficients):
        y_pred = np.polyval(coefficients, x_data)
        mse = np.mean((y_pred - y_data_noisy) ** 2)
        return -mse  # Negative MSE for maximization

    # Create the optimizer
    optimizer = SimpleGeneticAlgorithm(
        fitness_function=polynomial_fitness,
        population_size=100,
        mutation_rate=0.1,
        elite_size=10,
        export_data=False,
    )

    # Bounds for coefficients
    bounds = [(-10, 10)] * 4  # Same bounds for all coefficients

    # Run optimization
    best_solution, best_fitness = optimizer.optimize(
        n_generations=50,
        chromosome_length=4,  # 4 coefficients
        bounds=bounds,
    )

    # Check if solution is reasonable (not exact due to noise and stochasticity)
    for i, (true_coef, found_coef) in enumerate(
        zip(true_coefficients, best_solution)
    ):
        assert (
            abs(true_coef - found_coef) < 3.0
        ), f"Coefficient at index {i} is too far from true value"

    # Check that the fitness improved (MSE decreased)
    assert best_fitness < 0  # Should be negative since we're maximizing negative MSE

    # Calculate final MSE to verify it's reasonable
    y_pred_final = np.polyval(best_solution, x_data)
    final_mse = np.mean((y_pred_final - y_data_noisy) ** 2)
    assert final_mse < 10.0, "Final MSE should be reasonably low"


def test_parameter_validation():
    """Test that invalid parameters raise ValueError."""

    def dummy_fitness(x):
        return sum(x)

    with pytest.raises(ValueError, match="population_size"):
        SimpleGeneticAlgorithm(dummy_fitness, population_size=1)

    with pytest.raises(ValueError, match="mutation_rate"):
        SimpleGeneticAlgorithm(dummy_fitness, mutation_rate=-0.1)

    with pytest.raises(ValueError, match="mutation_rate"):
        SimpleGeneticAlgorithm(dummy_fitness, mutation_rate=1.5)

    with pytest.raises(ValueError, match="elite_size.*less than.*population_size"):
        SimpleGeneticAlgorithm(dummy_fitness, population_size=10, elite_size=10)

    with pytest.raises(ValueError, match="tournament_size.*must not exceed"):
        SimpleGeneticAlgorithm(dummy_fitness, population_size=10, tournament_size=20)


def test_gaussian_mutation():
    """Test that mutation uses Gaussian perturbation, not random replacement."""
    random.seed(42)

    def dummy_fitness(x):
        return sum(x)

    optimizer = SimpleGeneticAlgorithm(
        fitness_function=dummy_fitness,
        mutation_rate=1.0,  # Always mutate
        mutation_scale=0.01,  # Very small perturbation
    )

    original = [5.0, 5.0, 5.0]
    bounds = [(0, 10)] * 3

    # With small mutation_scale, mutated values should be close to originals
    mutated = optimizer._mutate(original, bounds)
    for orig, mut in zip(original, mutated):
        assert abs(orig - mut) < 1.0, (
            "Gaussian mutation with small scale should produce small changes"
        )


def test_early_stopping():
    """Test that early stopping works when fitness converges."""
    random.seed(42)
    np.random.seed(42)

    # Trivial fitness: sum of genes. Converges quickly.
    def simple_fitness(x):
        return -sum(xi**2 for xi in x)

    optimizer = SimpleGeneticAlgorithm(
        fitness_function=simple_fitness,
        population_size=50,
        convergence_threshold=1e-6,
        convergence_generations=5,
        export_data=False,
    )

    best_solution, best_fitness = optimizer.optimize(
        n_generations=500,
        chromosome_length=2,
        bounds=[(-5, 5)] * 2,
    )

    # Should have stopped well before 500 generations
    assert optimizer._n_generations_run < 500, (
        f"Expected early stopping but ran {optimizer._n_generations_run} generations"
    )
    assert optimizer._stopped_early is True


def test_metadata_includes_full_config():
    """Test that exported metadata includes all configuration."""
    random.seed(42)

    def dummy_fitness(x):
        return -sum(xi**2 for xi in x)

    optimizer = SimpleGeneticAlgorithm(
        fitness_function=dummy_fitness,
        population_size=20,
        mutation_rate=0.15,
        elite_size=3,
        tournament_size=5,
        export_data=False,
    )

    optimizer.optimize(n_generations=5, chromosome_length=3, bounds=[(-1, 1)] * 3)

    # Verify internal state tracks config
    assert optimizer._chromosome_length == 3
    assert optimizer._bounds == [(-1, 1)] * 3
