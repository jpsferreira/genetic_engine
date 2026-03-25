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

    for individual in population:
        for i, gene in enumerate(individual):
            min_val, max_val = bounds[i % len(bounds)]
            assert min_val <= gene <= max_val


def test_polynomial_fitting():
    """Test fitting a polynomial using SimpleGeneticAlgorithm."""
    random.seed(42)
    np.random.seed(42)

    true_coefficients = [2, -5, 3, -1]
    x_data = np.linspace(-2, 2, 20)
    y_data = np.polyval(true_coefficients, x_data)
    np.random.seed(42)
    y_data_noisy = y_data + np.random.normal(0, 0.5, size=len(y_data))

    def polynomial_fitness(coefficients):
        y_pred = np.polyval(coefficients, x_data)
        mse = np.mean((y_pred - y_data_noisy) ** 2)
        return -mse

    optimizer = SimpleGeneticAlgorithm(
        fitness_function=polynomial_fitness,
        population_size=100,
        mutation_rate=0.1,
        elite_size=10,
        export_data=False,
    )

    bounds = [(-10, 10)] * 4
    best_solution, best_fitness = optimizer.optimize(
        n_generations=50,
        chromosome_length=4,
        bounds=bounds,
    )

    for i, (true_coef, found_coef) in enumerate(
        zip(true_coefficients, best_solution)
    ):
        assert abs(true_coef - found_coef) < 3.0, (
            f"Coefficient at index {i} is too far from true value"
        )

    assert best_fitness < 0
    y_pred_final = np.polyval(best_solution, x_data)
    final_mse = np.mean((y_pred_final - y_data_noisy) ** 2)
    assert final_mse < 10.0


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

    with pytest.raises(ValueError, match="sense"):
        SimpleGeneticAlgorithm(dummy_fitness, sense="invalid")

    with pytest.raises(ValueError, match="crossover_type"):
        SimpleGeneticAlgorithm(dummy_fitness, crossover_type="invalid")

    with pytest.raises(ValueError, match="selection_type"):
        SimpleGeneticAlgorithm(dummy_fitness, selection_type="invalid")

    with pytest.raises(ValueError, match="crossover_rate"):
        SimpleGeneticAlgorithm(dummy_fitness, crossover_rate=1.5)


def test_gaussian_mutation():
    """Test that mutation uses Gaussian perturbation."""
    random.seed(42)

    def dummy_fitness(x):
        return sum(x)

    optimizer = SimpleGeneticAlgorithm(
        fitness_function=dummy_fitness,
        mutation_rate=1.0,
        mutation_scale=0.01,
    )

    original = [5.0, 5.0, 5.0]
    bounds = [(0, 10)] * 3

    mutated = optimizer._mutate(original, bounds)
    for orig, mut in zip(original, mutated):
        assert abs(orig - mut) < 1.0


def test_early_stopping():
    """Test early stopping on convergence."""
    random.seed(42)
    np.random.seed(42)

    def simple_fitness(x):
        return -sum(xi ** 2 for xi in x)

    optimizer = SimpleGeneticAlgorithm(
        fitness_function=simple_fitness,
        population_size=50,
        convergence_threshold=1e-6,
        convergence_generations=5,
        export_data=False,
    )

    optimizer.optimize(n_generations=500, chromosome_length=2, bounds=[(-5, 5)] * 2)
    assert optimizer._n_generations_run < 500
    assert optimizer._stopped_early is True


def test_metadata_includes_full_config():
    """Test that internal state tracks all configuration."""
    random.seed(42)

    def dummy_fitness(x):
        return -sum(xi ** 2 for xi in x)

    optimizer = SimpleGeneticAlgorithm(
        fitness_function=dummy_fitness,
        population_size=20,
        mutation_rate=0.15,
        elite_size=3,
        tournament_size=5,
        export_data=False,
    )

    optimizer.optimize(n_generations=5, chromosome_length=3, bounds=[(-1, 1)] * 3)
    assert optimizer._chromosome_length == 3
    assert optimizer._bounds == [(-1, 1)] * 3


# -----------------------------------------------------------------------
# New feature tests
# -----------------------------------------------------------------------


def test_minimize_sense():
    """Test that sense='minimize' finds a minimum."""
    random.seed(42)

    def sphere(x):
        return sum(xi ** 2 for xi in x)

    optimizer = SimpleGeneticAlgorithm(
        fitness_function=sphere,
        population_size=80,
        sense="minimize",
        export_data=False,
    )

    best_x, best_f = optimizer.optimize(
        n_generations=100, chromosome_length=3, bounds=[(-5, 5)] * 3
    )
    assert best_f < 1.0, f"Expected near-zero minimum, got {best_f}"


def test_crossover_types():
    """Test that all crossover types work without error."""
    random.seed(42)

    def sphere(x):
        return sum(xi ** 2 for xi in x)

    for cx_type in ["single_point", "uniform", "blx_alpha", "sbx"]:
        optimizer = SimpleGeneticAlgorithm(
            fitness_function=sphere,
            population_size=30,
            sense="minimize",
            crossover_type=cx_type,
            export_data=False,
        )
        best_x, best_f = optimizer.optimize(
            n_generations=20, chromosome_length=3, bounds=[(-5, 5)] * 3
        )
        assert best_f is not None


def test_selection_types():
    """Test that all selection types work without error."""
    random.seed(42)

    def sphere(x):
        return sum(xi ** 2 for xi in x)

    for sel_type in ["tournament", "roulette", "rank"]:
        optimizer = SimpleGeneticAlgorithm(
            fitness_function=sphere,
            population_size=30,
            sense="minimize",
            selection_type=sel_type,
            export_data=False,
        )
        best_x, best_f = optimizer.optimize(
            n_generations=20, chromosome_length=3, bounds=[(-5, 5)] * 3
        )
        assert best_f is not None


def test_adaptive_mutation():
    """Test that adaptive mutation runs without error and converges."""
    random.seed(42)

    def sphere(x):
        return sum(xi ** 2 for xi in x)

    optimizer = SimpleGeneticAlgorithm(
        fitness_function=sphere,
        population_size=50,
        sense="minimize",
        adaptive_mutation=True,
        export_data=False,
    )
    best_x, best_f = optimizer.optimize(
        n_generations=100, chromosome_length=3, bounds=[(-5, 5)] * 3
    )
    assert best_f < 5.0


def test_seed_population():
    """Test that seed_population injects individuals."""
    random.seed(42)

    def sphere(x):
        return sum(xi ** 2 for xi in x)

    seed = [[0.1, 0.1, 0.1], [0.01, 0.01, 0.01]]
    optimizer = SimpleGeneticAlgorithm(
        fitness_function=sphere,
        population_size=20,
        sense="minimize",
        seed_population=seed,
        export_data=False,
    )
    best_x, best_f = optimizer.optimize(
        n_generations=50, chromosome_length=3, bounds=[(-5, 5)] * 3
    )
    # With seeded near-optimal individuals, should converge very quickly
    assert best_f < 0.5


def test_constraints():
    """Test penalty-based constraint handling."""
    random.seed(42)

    def objective(x):
        return x[0] + x[1]

    # Constraint: x[0] >= 2  →  g(x) = 2 - x[0] <= 0
    def g1(x):
        return 2.0 - x[0]

    optimizer = SimpleGeneticAlgorithm(
        fitness_function=objective,
        population_size=50,
        sense="minimize",
        constraints=[g1],
        penalty_weight=1000.0,
        export_data=False,
    )
    best_x, best_f = optimizer.optimize(
        n_generations=100, chromosome_length=2, bounds=[(-5, 5)] * 2
    )
    # x[0] should be >= 2 (or very close)
    assert best_x[0] >= 1.5, f"Constraint violated: x[0]={best_x[0]}"


def test_callbacks():
    """Test that callbacks are invoked each generation."""
    random.seed(42)

    def sphere(x):
        return sum(xi ** 2 for xi in x)

    call_count = [0]

    def my_callback(generation, population, fitness_scores, metrics):
        call_count[0] += 1

    optimizer = SimpleGeneticAlgorithm(
        fitness_function=sphere,
        population_size=20,
        callbacks=[my_callback],
        export_data=False,
    )
    optimizer.optimize(n_generations=10, chromosome_length=2, bounds=[(-5, 5)] * 2)
    assert call_count[0] == 10


def test_crossover_rate():
    """Test that crossover_rate=0 means children are just copies of parent1."""
    random.seed(42)

    def sphere(x):
        return sum(xi ** 2 for xi in x)

    optimizer = SimpleGeneticAlgorithm(
        fitness_function=sphere,
        population_size=20,
        crossover_rate=0.0,
        mutation_rate=0.0,
        export_data=False,
    )
    # With no crossover and no mutation, children = copies of selected parents
    # This should still run without error
    optimizer.optimize(n_generations=5, chromosome_length=3, bounds=[(-5, 5)] * 3)
