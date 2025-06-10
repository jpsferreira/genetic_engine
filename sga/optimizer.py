"""Genetic optimization implementation."""

import abc
import random
import time
import psutil
import statistics
import atexit
import os
from typing import Callable, List, Optional, Tuple, Any, Dict, Union
from pathlib import Path

from genetic_opt.utils.monitor import OptimizationMonitor
from genetic_opt.utils.export import (
    export_metrics_to_csv,
    export_population_history_to_csv,
    export_run_metadata,
)


class GeneticOptimizer(abc.ABC):
    """Abstract base class for genetic optimization algorithms."""

    def __init__(
        self,
        fitness_function: Callable[[List[float]], float],
        population_size: int = 100,
        mutation_rate: float = 0.1,
        elite_size: int = 10,
        verbose: bool = False,
        live_monitor: bool = False,
        track_history: bool = False,
        export_data: bool = True,
    ):
        """Initialize the genetic optimizer.
        
        Args:
            fitness_function: Function to evaluate solutions
            population_size: Number of individuals in the population
            mutation_rate: Probability of mutation
            elite_size: Number of top individuals to keep unchanged
            verbose: Whether to print progress information
            live_monitor: Whether to show live monitoring interface
            track_history: Whether to track population history for each generation
            export_data: Whether to export run data to files on completion
        """
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.verbose = verbose
        self.live_monitor = live_monitor
        self.track_history = track_history
        self.export_data = export_data
        self.best_solution: Optional[Tuple[List[float], float]] = None
        self.metrics: Dict[str, List] = {
            "best_fitness": [],
            "avg_fitness": [],
            "std_fitness": [],
            "generation_time": [],
            "memory_usage_mb": [],
        }
        self.population_history: List[List[List[float]]] = []
        self.monitor = OptimizationMonitor() if live_monitor else None
        self.export_paths: Dict[str, str] = {}
        
        # Register a cleanup function to ensure curses is properly closed
        if self.live_monitor:
            atexit.register(self._cleanup_monitor)
        
    def optimize(
        self, 
        n_generations: int, 
        chromosome_length: int,
        bounds: List[Tuple[float, float]],
    ) -> Tuple[List[float], float]:
        """Run the genetic optimization algorithm.
        
        Args:
            n_generations: Number of generations to run
            chromosome_length: Length of each solution vector
            bounds: List of (min, max) bounds for each parameter
            
        Returns:
            Tuple of (best_solution, best_fitness)
        """
        # Initialize population
        start_time = time.time()
        population = self._initialize_population(chromosome_length, bounds)
        
        # Save the initial population if tracking history
        if self.track_history:
            self.population_history.append([ind.copy() for ind in population])
        
        if self.live_monitor:
            # Start the monitor
            self.monitor.start(n_generations, self.metrics)
        elif self.verbose:
            print(f"{'Generation':^10} | {'Best Fitness':^15} | {'Avg Fitness':^15} | {'Std Fitness':^15} | {'Time (s)':^10} | {'Memory (MB)':^12}")
            print("-" * 85)
        
        # Run generations
        for generation in range(n_generations):
            gen_start_time = time.time()
            
            # Evaluate fitness
            fitness_scores = [self.fitness_function(ind) for ind in population]
            
            # Calculate statistics
            best_idx = fitness_scores.index(max(fitness_scores))
            best_fitness = fitness_scores[best_idx]
            avg_fitness = statistics.mean(fitness_scores)
            std_fitness = statistics.stdev(fitness_scores) if len(fitness_scores) > 1 else 0
            
            # Update best solution
            current_best = (population[best_idx].copy(), best_fitness)
            
            if self.best_solution is None or current_best[1] > self.best_solution[1]:
                self.best_solution = current_best
            
            # Create next generation
            population = self._create_next_generation(population, fitness_scores, bounds)
            
            # Save the population if tracking history
            if self.track_history:
                self.population_history.append([ind.copy() for ind in population])
            
            # Calculate metrics
            gen_time = time.time() - gen_start_time
            memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            
            # Store metrics
            self.metrics["best_fitness"].append(best_fitness)
            self.metrics["avg_fitness"].append(avg_fitness)
            self.metrics["std_fitness"].append(std_fitness)
            self.metrics["generation_time"].append(gen_time)
            self.metrics["memory_usage_mb"].append(memory_usage)
            
            # Show progress
            if self.live_monitor:
                self.monitor.update(generation)
            elif self.verbose:
                print(f"{generation:^10} | {best_fitness:^15.6f} | {avg_fitness:^15.6f} | {std_fitness:^15.6f} | {gen_time:^10.4f} | {memory_usage:^12.2f}")
        
        # Cleanup monitor if used
        if self.live_monitor:
            self.monitor.stop()
        
        total_time = time.time() - start_time
        if self.verbose and not self.live_monitor:
            print("-" * 85)
            print(f"Total optimization time: {total_time:.2f} seconds")
            print(f"Final best fitness: {self.best_solution[1]:.6f}")
        
        # Export data if requested
        if self.export_data:
            self.export_run_data()
        
        return self.best_solution
    
    def export_run_data(
        self, directory: str = "results", base_filename: Optional[str] = None
    ) -> Dict[str, str]:
        """Export all run data to files.
        
        Args:
            directory: Base directory to save files to
            base_filename: Base name for the files (without extension)
            
        Returns:
            Dictionary of file types to file paths
        """
        # Generate timestamp for this run
        timestamp = Path(time.strftime("%Y%m%d_%H%M%S"))
        
        # Generate base filename if not provided
        if base_filename is None:
            base_filename = f"optimization_run_{timestamp}"
        
        # Create run directory within the base directory
        run_dir = Path(directory) / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Export metrics
        metrics_file = export_metrics_to_csv(
            self.metrics, 
            filename=f"{base_filename}_metrics.csv", 
            directory=str(run_dir)
        )
        self.export_paths["metrics"] = metrics_file
        
        # Export population history if available
        if self.track_history and self.population_history:
            history_file = export_population_history_to_csv(
                self.population_history,
                filename=f"{base_filename}_population.csv",
                directory=str(run_dir)
            )
            self.export_paths["population_history"] = history_file
        
        # Export run metadata
        config = {
            "population_size": self.population_size,
            "mutation_rate": self.mutation_rate,
            "elite_size": self.elite_size,
            "generations": len(self.metrics["best_fitness"]),
            "track_history": self.track_history,
        }
        
        results = {}
        if self.best_solution:
            results["best_solution"] = self.best_solution[0]
            results["best_fitness"] = self.best_solution[1]
            results["total_evaluations"] = self.population_size * len(self.metrics["best_fitness"])
            if len(self.metrics["generation_time"]) > 0:
                results["total_time"] = sum(self.metrics["generation_time"])
        
        metadata_file = export_run_metadata(
            config, 
            results,
            filename=f"{base_filename}_metadata.json",
            directory=str(run_dir)
        )
        self.export_paths["metadata"] = metadata_file
        self.export_paths["run_directory"] = str(run_dir)
        
        return self.export_paths
    
    def _cleanup_monitor(self):
        """Ensure the monitor is properly stopped."""
        if self.monitor and self.monitor.started:
            self.monitor.stop()
    
    @abc.abstractmethod
    def _initialize_population(
        self, chromosome_length: int, bounds: List[Tuple[float, float]]
    ) -> List[List[float]]:
        """Initialize the population.
        
        Args:
            chromosome_length: Length of each chromosome
            bounds: List of (min, max) bounds for each parameter
            
        Returns:
            List of individuals (each a list of float values)
        """
        pass
    
    @abc.abstractmethod
    def _create_next_generation(
        self, 
        population: List[List[float]], 
        fitness_scores: List[float],
        bounds: List[Tuple[float, float]],
    ) -> List[List[float]]:
        """Create the next generation of individuals.
        
        Args:
            population: Current population
            fitness_scores: Fitness scores for current population
            bounds: List of (min, max) bounds for each parameter
            
        Returns:
            New population
        """
        pass


class SimpleGeneticAlgorithm(GeneticOptimizer):
    """Simple genetic algorithm implementation."""
    
    def _initialize_population(
        self, chromosome_length: int, bounds: List[Tuple[float, float]]
    ) -> List[List[float]]:
        """Initialize random population within bounds."""
        population = []
        for _ in range(self.population_size):
            individual = []
            for i in range(chromosome_length):
                min_val, max_val = bounds[i % len(bounds)]
                gene = random.uniform(min_val, max_val)
                individual.append(gene)
            population.append(individual)
        return population
    
    def _create_next_generation(
        self, 
        population: List[List[float]], 
        fitness_scores: List[float],
        bounds: List[Tuple[float, float]],
    ) -> List[List[float]]:
        """Create next generation using selection, crossover and mutation."""
        # Sort population by fitness
        sorted_indices = sorted(range(len(fitness_scores)), 
                               key=lambda i: fitness_scores[i], 
                               reverse=True)
        sorted_population = [population[i] for i in sorted_indices]
        
        # Keep elite individuals
        new_population = [ind.copy() for ind in sorted_population[:self.elite_size]]
        
        # Create rest of population
        while len(new_population) < self.population_size:
            # Selection - tournament selection
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)
            
            # Crossover
            child = self._crossover(parent1, parent2)
            
            # Mutation
            child = self._mutate(child, bounds)
            
            new_population.append(child)
            
        return new_population
    
    def _tournament_selection(
        self, population: List[List[float]], fitness_scores: List[float], tournament_size: int = 3
    ) -> List[float]:
        """Select individual using tournament selection."""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
        return population[winner_idx]
    
    def _crossover(self, parent1: List[float], parent2: List[float]) -> List[float]:
        """Single point crossover between two parents."""
        if len(parent1) <= 1:
            return parent1.copy()
            
        crossover_point = random.randint(1, len(parent1) - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child
    
    def _mutate(self, individual: List[float], bounds: List[Tuple[float, float]]) -> List[float]:
        """Randomly mutate genes in the individual."""
        mutated = individual.copy()
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                min_val, max_val = bounds[i % len(bounds)]
                mutated[i] = random.uniform(min_val, max_val)
        return mutated