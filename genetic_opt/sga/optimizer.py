"""Genetic optimization implementation."""

import abc
import random
import time
import psutil
import statistics
import atexit
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, List, Optional, Tuple, Dict, Union

from pathlib import Path

from genetic_opt.sga.utils.monitor import OptimizationMonitor
from genetic_opt.sga.utils.export import (
    export_metrics_to_csv,
    export_population_history_to_csv,
    export_run_metadata,
)
from genetic_opt.sga.operators import (
    CROSSOVER_OPERATORS,
    SELECTION_OPERATORS,
)


class GeneticOptimizer(abc.ABC):
    """Abstract base class for genetic optimization algorithms."""

    def __init__(
        self,
        fitness_function: Callable[[List[float]], float],
        population_size: int = 100,
        mutation_rate: float = 0.1,
        elite_size: int = 10,
        sense: str = "maximize",
        verbose: bool = False,
        live_monitor: bool = False,
        track_history: bool = False,
        export_data: bool = True,
        parallel_evaluation: bool = False,
        n_workers: Optional[int] = None,
        convergence_threshold: Optional[float] = None,
        convergence_generations: int = 20,
        constraints: Optional[List[Callable[[List[float]], float]]] = None,
        penalty_weight: float = 1000.0,
        callbacks: Optional[List[Callable]] = None,
        seed_population: Optional[List[List[float]]] = None,
    ):
        """Initialize the genetic optimizer.

        Args:
            fitness_function: Function to evaluate solutions.
            population_size: Number of individuals in the population.
            mutation_rate: Probability of mutation (0.0 to 1.0).
            elite_size: Number of top individuals to keep unchanged.
            sense: ``"maximize"`` or ``"minimize"``.
            verbose: Whether to print progress information.
            live_monitor: Whether to show live monitoring interface.
            track_history: Whether to track population history for each generation.
            export_data: Whether to export run data to files on completion.
            parallel_evaluation: Whether to evaluate fitness in parallel.
            n_workers: Number of worker processes for parallel evaluation
                (``None`` = CPU count).
            convergence_threshold: Stop early if best fitness improves less
                than this over *convergence_generations*.  ``None`` disables.
            convergence_generations: Number of generations to look back for
                the convergence check.
            constraints: List of inequality constraint functions ``g(x)``.
                A solution is feasible when ``g(x) <= 0`` for all constraints.
            penalty_weight: Multiplier applied to constraint violations.
            callbacks: List of callables invoked at the end of every
                generation with signature
                ``callback(generation, population, fitness_scores, metrics)``.
            seed_population: Optional list of individuals to inject into the
                initial population.  Remaining slots are filled randomly.

        Raises:
            ValueError: If parameters are invalid.
        """
        if population_size < 2:
            raise ValueError("population_size must be at least 2")
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be between 0.0 and 1.0")
        if elite_size < 0:
            raise ValueError("elite_size must be non-negative")
        if elite_size >= population_size:
            raise ValueError(
                f"elite_size ({elite_size}) must be less than "
                f"population_size ({population_size})"
            )
        if sense not in ("maximize", "minimize"):
            raise ValueError("sense must be 'maximize' or 'minimize'")
        if convergence_generations < 1:
            raise ValueError("convergence_generations must be at least 1")

        self.fitness_function = fitness_function
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.sense = sense
        self.verbose = verbose
        self.live_monitor = live_monitor
        self.track_history = track_history
        self.export_data = export_data
        self.parallel_evaluation = parallel_evaluation
        self.n_workers = n_workers
        self.convergence_threshold = convergence_threshold
        self.convergence_generations = convergence_generations
        self.constraints = constraints or []
        self.penalty_weight = penalty_weight
        self.callbacks = callbacks or []
        self.seed_population = seed_population
        self.best_solution: Optional[Tuple[List[float], float]] = None
        self.metrics: Dict[str, List] = {
            "best_fitness": [],
            "avg_fitness": [],
            "std_fitness": [],
            "generation_time": [],
            "memory_usage_mb": [],
        }
        self.population_history: List[List[List[float]]] = []
        self.fitness_history: List[List[float]] = []
        self.monitor = OptimizationMonitor() if live_monitor else None
        self.export_paths: Dict[str, str] = {}
        self._stopped_early = False
        self._n_generations_run = 0
        self._chromosome_length: Optional[int] = None
        self._bounds: Optional[List[Tuple[float, float]]] = None
        self._maximize = sense == "maximize"

        if self.live_monitor:
            atexit.register(self._cleanup_monitor)

    # ------------------------------------------------------------------
    # Comparison helpers (sense-aware)
    # ------------------------------------------------------------------

    def _is_better(self, a: float, b: float) -> bool:
        """Return True if fitness *a* is better than *b*."""
        return a > b if self._maximize else a < b

    def _best_of(self, values: List[float]) -> float:
        return max(values) if self._maximize else min(values)

    def _best_index(self, values: List[float]) -> int:
        fn = max if self._maximize else min
        best = fn(values)
        return values.index(best)

    # ------------------------------------------------------------------
    # Constraint handling
    # ------------------------------------------------------------------

    def _apply_penalty(self, raw_fitness: float, individual: List[float]) -> float:
        """Adjust *raw_fitness* by adding a penalty for constraint violations."""
        if not self.constraints:
            return raw_fitness
        violation = sum(max(0.0, g(individual)) for g in self.constraints)
        if self._maximize:
            return raw_fitness - self.penalty_weight * violation
        else:
            return raw_fitness + self.penalty_weight * violation

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def optimize(
        self,
        n_generations: int,
        chromosome_length: int,
        bounds: List[Tuple[float, float]],
    ) -> Tuple[List[float], float]:
        """Run the genetic optimization algorithm.

        Args:
            n_generations: Number of generations to run.
            chromosome_length: Length of each solution vector.
            bounds: List of (min, max) bounds for each parameter.

        Returns:
            Tuple of ``(best_solution, best_fitness)``.
        """
        if n_generations < 1:
            raise ValueError("n_generations must be at least 1")
        if chromosome_length < 1:
            raise ValueError("chromosome_length must be at least 1")
        if not bounds:
            raise ValueError("bounds must not be empty")

        self._chromosome_length = chromosome_length
        self._bounds = bounds

        start_time = time.time()
        population = self._initialize_population(chromosome_length, bounds)

        # Evaluate initial population
        initial_fitness = self._evaluate_population(population)

        if self.track_history:
            self.population_history.append([ind.copy() for ind in population])
            self.fitness_history.append(initial_fitness.copy())

        if self.live_monitor:
            self.monitor.start(n_generations, self.metrics)
        elif self.verbose:
            self._print_header()

        self._stopped_early = False

        for generation in range(n_generations):
            gen_start_time = time.time()

            fitness_scores = self._evaluate_population(population)

            # Statistics
            best_idx = self._best_index(fitness_scores)
            best_fitness = fitness_scores[best_idx]
            avg_fitness = statistics.mean(fitness_scores)
            std_fitness = (
                statistics.stdev(fitness_scores) if len(fitness_scores) > 1 else 0
            )

            # Update best solution
            current_best = (population[best_idx].copy(), best_fitness)
            if self.best_solution is None or self._is_better(
                current_best[1], self.best_solution[1]
            ):
                self.best_solution = current_best

            # Next generation
            population = self._create_next_generation(
                population, fitness_scores, bounds
            )

            if self.track_history:
                new_fitness = self._evaluate_population(population)
                self.population_history.append([ind.copy() for ind in population])
                self.fitness_history.append(new_fitness.copy())

            gen_time = time.time() - gen_start_time
            memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)

            self.metrics["best_fitness"].append(best_fitness)
            self.metrics["avg_fitness"].append(avg_fitness)
            self.metrics["std_fitness"].append(std_fitness)
            self.metrics["generation_time"].append(gen_time)
            self.metrics["memory_usage_mb"].append(memory_usage)

            # Progress display
            if self.live_monitor:
                if not self.monitor.started:
                    self.live_monitor = False
                    self.verbose = True
                    self._print_header()
                else:
                    self.monitor.update(generation)
            elif self.verbose:
                self._print_row(generation, best_fitness, avg_fitness, std_fitness,
                                gen_time, memory_usage)

            # Callbacks
            for cb in self.callbacks:
                cb(generation, population, fitness_scores, self.metrics)

            self._n_generations_run = generation + 1

            if self._check_convergence():
                self._stopped_early = True
                if self.verbose and not self.live_monitor:
                    print(
                        f"\nEarly stopping: fitness converged after "
                        f"{generation + 1} generations"
                    )
                break

        if self.live_monitor and self.monitor and self.monitor.started:
            self.monitor.stop()

        total_time = time.time() - start_time
        if self.verbose and not self.live_monitor:
            print("-" * 85)
            print(f"Total optimization time: {total_time:.2f} seconds")
            print(f"Final best fitness: {self.best_solution[1]:.6f}")

        if self.export_data:
            self.export_run_data()

        return self.best_solution

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _print_header():
        header = (
            f"{'Generation':^10} | {'Best Fitness':^15} | "
            f"{'Avg Fitness':^15} | {'Std Fitness':^15} | "
            f"{'Time (s)':^10} | {'Memory (MB)':^12}"
        )
        print(header)
        print("-" * 85)

    @staticmethod
    def _print_row(generation, best, avg, std, gen_time, mem):
        print(
            f"{generation:^10} | {best:^15.6f} | "
            f"{avg:^15.6f} | {std:^15.6f} | "
            f"{gen_time:^10.4f} | {mem:^12.2f}"
        )

    # ------------------------------------------------------------------
    # Convergence
    # ------------------------------------------------------------------

    def _check_convergence(self) -> bool:
        if self.convergence_threshold is None:
            return False
        history = self.metrics["best_fitness"]
        if len(history) < self.convergence_generations:
            return False
        recent = history[-self.convergence_generations:]
        return abs(recent[-1] - recent[0]) < self.convergence_threshold

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_run_data(
        self, directory: str = "results", base_filename: Optional[str] = None
    ) -> Dict[str, str]:
        """Export all run data to files."""
        timestamp = Path(time.strftime("%Y%m%d_%H%M%S"))
        if base_filename is None:
            base_filename = f"optimization_run_{timestamp}"

        run_dir = Path(directory) / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        metrics_file = export_metrics_to_csv(
            self.metrics,
            filename=f"{base_filename}_metrics.csv",
            directory=str(run_dir),
        )
        self.export_paths["metrics"] = metrics_file

        if self.track_history and self.population_history:
            history_file = export_population_history_to_csv(
                self.population_history,
                filename=f"{base_filename}_population.csv",
                directory=str(run_dir),
            )
            self.export_paths["population_history"] = history_file

        config = {
            "population_size": self.population_size,
            "mutation_rate": self.mutation_rate,
            "elite_size": self.elite_size,
            "sense": self.sense,
            "generations_requested": len(self.metrics["best_fitness"]),
            "generations_run": self._n_generations_run,
            "stopped_early": self._stopped_early,
            "track_history": self.track_history,
            "parallel_evaluation": self.parallel_evaluation,
            "convergence_threshold": self.convergence_threshold,
            "convergence_generations": self.convergence_generations,
            "n_constraints": len(self.constraints),
        }
        if self._chromosome_length is not None:
            config["chromosome_length"] = self._chromosome_length
        if self._bounds is not None:
            config["bounds"] = self._bounds

        results = {}
        if self.best_solution:
            results["best_solution"] = self.best_solution[0]
            results["best_fitness"] = self.best_solution[1]
            results["total_evaluations"] = self.population_size * len(
                self.metrics["best_fitness"]
            )
            if self.metrics["generation_time"]:
                results["total_time"] = sum(self.metrics["generation_time"])

        metadata_file = export_run_metadata(
            config, results,
            filename=f"{base_filename}_metadata.json",
            directory=str(run_dir),
        )
        self.export_paths["metadata"] = metadata_file
        self.export_paths["run_directory"] = str(run_dir)
        return self.export_paths

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cleanup_monitor(self) -> None:
        if self.monitor and self.monitor.started:
            self.monitor.stop()

    def _get_bounds(
        self, gene_index: int, bounds: List[Tuple[float, float]]
    ) -> Tuple[float, float]:
        return bounds[gene_index % len(bounds)]

    def _evaluate_population(self, population: List[List[float]]) -> List[float]:
        if self.parallel_evaluation:
            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                raw = list(executor.map(self.fitness_function, population))
        else:
            raw = [self.fitness_function(ind) for ind in population]

        if self.constraints:
            return [
                self._apply_penalty(f, ind)
                for f, ind in zip(raw, population)
            ]
        return raw

    @abc.abstractmethod
    def _initialize_population(
        self, chromosome_length: int, bounds: List[Tuple[float, float]]
    ) -> List[List[float]]:
        pass

    @abc.abstractmethod
    def _create_next_generation(
        self,
        population: List[List[float]],
        fitness_scores: List[float],
        bounds: List[Tuple[float, float]],
    ) -> List[List[float]]:
        pass


class SimpleGeneticAlgorithm(GeneticOptimizer):
    """Configurable genetic algorithm with pluggable operators.

    Supports multiple crossover strategies, selection strategies,
    adaptive mutation, population seeding, constraint handling, and
    both minimization and maximization.
    """

    def __init__(
        self,
        fitness_function: Callable[[List[float]], float],
        population_size: int = 100,
        mutation_rate: float = 0.1,
        elite_size: int = 10,
        tournament_size: int = 3,
        sense: str = "maximize",
        crossover_type: str = "single_point",
        crossover_rate: float = 0.8,
        selection_type: str = "tournament",
        adaptive_mutation: bool = False,
        mutation_scale: float = 0.1,
        verbose: bool = False,
        live_monitor: bool = False,
        track_history: bool = False,
        export_data: bool = True,
        parallel_evaluation: bool = False,
        n_workers: Optional[int] = None,
        convergence_threshold: Optional[float] = None,
        convergence_generations: int = 20,
        constraints: Optional[List[Callable[[List[float]], float]]] = None,
        penalty_weight: float = 1000.0,
        callbacks: Optional[List[Callable]] = None,
        seed_population: Optional[List[List[float]]] = None,
    ):
        """Initialize the genetic algorithm.

        Args:
            fitness_function: Function to evaluate solutions.
            population_size: Number of individuals in the population.
            mutation_rate: Base probability of mutation per gene.
            elite_size: Number of top individuals kept unchanged.
            tournament_size: Tournament size (only used when
                ``selection_type="tournament"``).
            sense: ``"maximize"`` or ``"minimize"``.
            crossover_type: One of ``"single_point"``, ``"uniform"``,
                ``"blx_alpha"``, ``"sbx"``.
            crossover_rate: Probability that crossover is applied.  When
                crossover is skipped the first parent is copied directly.
            selection_type: One of ``"tournament"``, ``"roulette"``, ``"rank"``.
            adaptive_mutation: When ``True`` the mutation rate is adjusted
                each generation based on population diversity.
            mutation_scale: Std dev of Gaussian mutation as a fraction of
                the gene range (0 < scale <= 1).
            verbose: Print progress information.
            live_monitor: Show live curses monitor.
            track_history: Record every population for analysis.
            export_data: Write results to disk on completion.
            parallel_evaluation: Evaluate fitness in parallel.
            n_workers: Worker processes for parallel evaluation.
            convergence_threshold: Early-stopping threshold.
            convergence_generations: Lookback window for convergence.
            constraints: Inequality constraints ``g(x) <= 0``.
            penalty_weight: Penalty multiplier for constraint violations.
            callbacks: Per-generation callbacks.
            seed_population: Known-good solutions to inject.
        """
        if tournament_size < 1:
            raise ValueError("tournament_size must be at least 1")
        if tournament_size > population_size:
            raise ValueError(
                f"tournament_size ({tournament_size}) must not exceed "
                f"population_size ({population_size})"
            )
        if not 0.0 < mutation_scale <= 1.0:
            raise ValueError("mutation_scale must be between 0.0 (exclusive) and 1.0")
        if crossover_type not in CROSSOVER_OPERATORS:
            raise ValueError(
                f"crossover_type must be one of {list(CROSSOVER_OPERATORS)}, "
                f"got '{crossover_type}'"
            )
        if selection_type not in SELECTION_OPERATORS:
            raise ValueError(
                f"selection_type must be one of {list(SELECTION_OPERATORS)}, "
                f"got '{selection_type}'"
            )
        if not 0.0 <= crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be between 0.0 and 1.0")

        super().__init__(
            fitness_function=fitness_function,
            population_size=population_size,
            mutation_rate=mutation_rate,
            elite_size=elite_size,
            sense=sense,
            verbose=verbose,
            live_monitor=live_monitor,
            track_history=track_history,
            export_data=export_data,
            parallel_evaluation=parallel_evaluation,
            n_workers=n_workers,
            convergence_threshold=convergence_threshold,
            convergence_generations=convergence_generations,
            constraints=constraints,
            penalty_weight=penalty_weight,
            callbacks=callbacks,
            seed_population=seed_population,
        )
        self.tournament_size = tournament_size
        self.crossover_type = crossover_type
        self.crossover_rate = crossover_rate
        self.selection_type = selection_type
        self.adaptive_mutation = adaptive_mutation
        self.mutation_scale = mutation_scale
        self._effective_mutation_rate = mutation_rate

    # ------------------------------------------------------------------
    # Population initialisation
    # ------------------------------------------------------------------

    def _initialize_population(
        self, chromosome_length: int, bounds: List[Tuple[float, float]]
    ) -> List[List[float]]:
        """Initialise population, injecting seed individuals first."""
        population: List[List[float]] = []

        # Inject seed individuals
        if self.seed_population:
            for ind in self.seed_population:
                if len(ind) != chromosome_length:
                    raise ValueError(
                        f"Seed individual has length {len(ind)}, "
                        f"expected {chromosome_length}"
                    )
                population.append(list(ind))  # copy
            if len(population) > self.population_size:
                population = population[: self.population_size]

        # Fill remaining slots randomly
        while len(population) < self.population_size:
            individual = [
                random.uniform(*self._get_bounds(i, bounds))
                for i in range(chromosome_length)
            ]
            population.append(individual)

        return population

    # ------------------------------------------------------------------
    # Generation loop
    # ------------------------------------------------------------------

    def _create_next_generation(
        self,
        population: List[List[float]],
        fitness_scores: List[float],
        bounds: List[Tuple[float, float]],
    ) -> List[List[float]]:
        # Adaptive mutation: adjust rate based on diversity
        if self.adaptive_mutation:
            self._adapt_mutation_rate(fitness_scores)

        # Sort by fitness (best first)
        sorted_indices = sorted(
            range(len(fitness_scores)),
            key=lambda i: fitness_scores[i],
            reverse=self._maximize,
        )
        sorted_population = [population[i] for i in sorted_indices]

        # Elitism
        new_population = [ind.copy() for ind in sorted_population[: self.elite_size]]

        # Build the rest
        crossover_fn = CROSSOVER_OPERATORS[self.crossover_type]
        selection_fn = SELECTION_OPERATORS[self.selection_type]
        sel_kwargs = {
            "maximize": self._maximize,
            "tournament_size": self.tournament_size,
        }

        while len(new_population) < self.population_size:
            parent1 = selection_fn(population, fitness_scores, **sel_kwargs)
            parent2 = selection_fn(population, fitness_scores, **sel_kwargs)

            if random.random() < self.crossover_rate:
                child = crossover_fn(parent1, parent2, bounds=bounds)
            else:
                child = parent1.copy()

            child = self._mutate(child, bounds)
            new_population.append(child)

        return new_population

    # ------------------------------------------------------------------
    # Adaptive mutation
    # ------------------------------------------------------------------

    def _adapt_mutation_rate(self, fitness_scores: List[float]) -> None:
        """Adjust mutation rate based on population diversity.

        When diversity is low the mutation rate is increased to promote
        exploration; when diversity is high it is decreased.
        """
        if len(fitness_scores) < 2:
            return
        std = statistics.stdev(fitness_scores)
        mean = statistics.mean(fitness_scores)
        cv = std / abs(mean) if abs(mean) > 1e-12 else std

        # Low diversity → increase; high diversity → decrease
        if cv < 0.01:
            self._effective_mutation_rate = min(1.0, self.mutation_rate * 3.0)
        elif cv < 0.05:
            self._effective_mutation_rate = min(1.0, self.mutation_rate * 1.5)
        elif cv > 0.5:
            self._effective_mutation_rate = max(0.01, self.mutation_rate * 0.5)
        else:
            self._effective_mutation_rate = self.mutation_rate

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def _mutate(
        self, individual: List[float], bounds: List[Tuple[float, float]]
    ) -> List[float]:
        """Apply Gaussian mutation to genes, clamped to bounds."""
        rate = (
            self._effective_mutation_rate
            if self.adaptive_mutation
            else self.mutation_rate
        )
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < rate:
                min_val, max_val = self._get_bounds(i, bounds)
                sigma = (max_val - min_val) * self.mutation_scale
                mutated[i] = max(min_val, min(max_val, mutated[i] + random.gauss(0, sigma)))
        return mutated
