"""Tests for the convenience API and benchmark functions."""

import random
import pytest

from genetic_opt.sga.api import minimize, maximize, OptimizationResult
from genetic_opt.sga.benchmarks import (
    sphere, rastrigin, rosenbrock, ackley, griewank, schwefel,
    BENCHMARKS, get_benchmark,
)
from genetic_opt.sga.operators import (
    single_point_crossover, uniform_crossover,
    blx_alpha_crossover, sbx_crossover,
    tournament_selection, roulette_wheel_selection, rank_based_selection,
)


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------

class TestMinimize:
    def test_sphere(self):
        random.seed(42)
        result = minimize(sphere, bounds=[(-5, 5)] * 3, n_generations=100)
        assert isinstance(result, OptimizationResult)
        assert result.fun < 2.0
        assert len(result.x) == 3

    def test_returns_result_object(self):
        random.seed(42)
        result = minimize(sphere, bounds=[(-5, 5)] * 2, n_generations=10)
        assert hasattr(result, "x")
        assert hasattr(result, "fun")
        assert hasattr(result, "n_generations")
        assert hasattr(result, "converged")
        assert hasattr(result, "metrics")
        assert hasattr(result, "optimizer")


class TestMaximize:
    def test_negative_sphere(self):
        random.seed(42)
        result = maximize(
            lambda x: -sphere(x),
            bounds=[(-5, 5)] * 3,
            n_generations=100,
        )
        assert result.fun > -2.0

    def test_auto_population_size(self):
        """Population size should scale with dimensionality when not given."""
        random.seed(42)
        result = maximize(
            lambda x: -sphere(x),
            bounds=[(-1, 1)] * 20,
            n_generations=5,
        )
        assert result.optimizer.population_size == max(50, 10 * 20)


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------

class TestBenchmarks:
    def test_sphere_at_optimum(self):
        assert sphere([0.0, 0.0, 0.0]) == 0.0

    def test_rastrigin_at_optimum(self):
        assert abs(rastrigin([0.0, 0.0])) < 1e-10

    def test_rosenbrock_at_optimum(self):
        assert abs(rosenbrock([1.0, 1.0, 1.0])) < 1e-10

    def test_ackley_at_optimum(self):
        assert abs(ackley([0.0, 0.0])) < 1e-10

    def test_griewank_at_optimum(self):
        assert abs(griewank([0.0, 0.0])) < 1e-10

    def test_registry(self):
        assert "sphere" in BENCHMARKS
        assert "rastrigin" in BENCHMARKS
        assert "rosenbrock" in BENCHMARKS
        assert "ackley" in BENCHMARKS

    def test_get_benchmark(self):
        b = get_benchmark("sphere", ndim=5)
        assert b["ndim"] == 5
        assert len(b["bounds"]) == 5
        assert b["optimum"] == 0.0


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class TestCrossoverOperators:
    def setup_method(self):
        random.seed(42)
        self.p1 = [1.0, 2.0, 3.0, 4.0]
        self.p2 = [5.0, 6.0, 7.0, 8.0]
        self.bounds = [(-10, 10)] * 4

    def test_single_point(self):
        child = single_point_crossover(self.p1, self.p2)
        assert len(child) == 4

    def test_uniform(self):
        child = uniform_crossover(self.p1, self.p2)
        assert len(child) == 4

    def test_blx_alpha(self):
        child = blx_alpha_crossover(self.p1, self.p2, bounds=self.bounds)
        assert len(child) == 4
        for i, v in enumerate(child):
            assert self.bounds[i][0] <= v <= self.bounds[i][1]

    def test_sbx(self):
        child = sbx_crossover(self.p1, self.p2, bounds=self.bounds)
        assert len(child) == 4
        for i, v in enumerate(child):
            assert self.bounds[i][0] <= v <= self.bounds[i][1]


class TestSelectionOperators:
    def setup_method(self):
        random.seed(42)
        self.pop = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        self.fit = [10.0, 20.0, 30.0, 40.0, 50.0]

    def test_tournament(self):
        winner = tournament_selection(
            self.pop, self.fit, maximize=True, tournament_size=3
        )
        assert winner in self.pop

    def test_roulette(self):
        winner = roulette_wheel_selection(self.pop, self.fit, maximize=True)
        assert winner in self.pop

    def test_rank(self):
        winner = rank_based_selection(self.pop, self.fit, maximize=True)
        assert winner in self.pop

    def test_roulette_minimize(self):
        winner = roulette_wheel_selection(self.pop, self.fit, maximize=False)
        assert winner in self.pop

    def test_rank_minimize(self):
        winner = rank_based_selection(self.pop, self.fit, maximize=False)
        assert winner in self.pop
