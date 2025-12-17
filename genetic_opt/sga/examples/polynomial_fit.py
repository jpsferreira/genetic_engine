"""Example of using SimpleGeneticAlgorithm for polynomial curve fitting."""

import numpy as np
import matplotlib.pyplot as plt
import random
import time
import argparse
from pathlib import Path
from typing import List

from genetic_opt.sga.optimizer import SimpleGeneticAlgorithm
from genetic_opt.sga.utils import analyze_population_migration


def polynomial_fit_example(
    use_live_monitor: bool = True,
    track_population: bool = True,
    analyze_migration: bool = True
):
    """Demonstrate fitting a polynomial curve using genetic optimization.
    
    Args:
        use_live_monitor: Whether to use the live monitor or verbose output
        track_population: Whether to track population history for migration analysis
        analyze_migration: Whether to analyze population migration after optimization
        
    Returns:
        The optimizer object with the optimization results
    """
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Define the true polynomial: 2x^3 - 5x^2 + 3x - 1
    true_coefficients = [2, -5, 3, -1]  # [a, b, c, d] for ax^3 + bx^2 + cx + d
    
    # Generate data points
    x_data = np.linspace(-2, 2, 50)
    y_true = np.polyval(true_coefficients, x_data)
    
    # Add noise to create synthetic observations
    y_noisy = y_true + np.random.normal(0, 0.5, size=len(y_true))
    
    # Define fitness function (negative mean squared error)
    def polynomial_fitness(coefficients: List[float]) -> float:
        """Calculate fitness for a set of polynomial coefficients."""
        y_pred = np.polyval(coefficients, x_data)
        mse = np.mean((y_pred - y_noisy) ** 2)
        return -mse  # Negative MSE for maximization
    
    # Configure and run the genetic algorithm
    optimizer = SimpleGeneticAlgorithm(
        fitness_function=polynomial_fitness,
        population_size=200,
        mutation_rate=0.15,
        elite_size=20,
        verbose=not use_live_monitor,  # Only use verbose if not using live monitor
        live_monitor=use_live_monitor,  # Use live monitor as specified
        track_history=track_population,  # Track population for migration analysis
        export_data=True,  # Export data to CSV files
    )
    
    # Parameter bounds
    bounds = [(-10, 10)] * 4  # Same bounds for all coefficients
    
    if not use_live_monitor:
        print("\nStarting optimization for polynomial fitting...\n")
    
    # Run optimization
    best_solution, best_fitness = optimizer.optimize(
        n_generations=100,
        chromosome_length=4,
        bounds=bounds,
    )
    
    # If we were using the live monitor, add a small delay to make sure
    # the terminal has been restored correctly
    if use_live_monitor:
        time.sleep(0.5)
    
    # Results
    print("\nOptimization Results:")
    print(f"True coefficients: {true_coefficients}")
    print(f"Best solution found: {best_solution}")
    print(f"Best fitness (negative MSE): {best_fitness}")
    print(f"Final MSE: {-best_fitness}")
    
    # Print file paths where data was exported
    print("\nData exported to:")
    for data_type, filepath in optimizer.export_paths.items():
        print(f"- {data_type}: {filepath}")
    
    # Analyze population migration if requested
    if analyze_migration and track_population and "population_history" in optimizer.export_paths:
        print("\nAnalyzing population migration patterns...")
        
        # Get the run directory path from the optimizer's export paths
        run_dir = optimizer.export_paths.get("run_directory")
        
        analysis_results = analyze_population_migration(
            population_file=optimizer.export_paths["population_history"],
            output_dir="analysis",
            run_dir=run_dir,  # Pass the run directory to organize by timestamp
            create_animation=True
        )
        
        print("\nAnalysis results saved to:")
        print(f"- Statistics: {analysis_results['statistics_plot']}")
        
        print("- Density plots:")
        for name, path in analysis_results["density_plots"].items():
            print(f"  - {name}: {path}")
        
        print("- Animations:")
        for name, path in analysis_results["animations"].items():
            print(f"  - {name}: {path}")
    
    # Get the run directory to save plots in the same structure
    run_dir = Path(optimizer.export_paths.get("run_directory", "results"))
    
    # Ensure run directory exists
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualization of the polynomial fit
    plt.figure(figsize=(10, 6))
    
    # Plot the original data points
    plt.scatter(x_data, y_noisy, alpha=0.6, label='Noisy data points')
    
    # Plot the true function
    plt.plot(x_data, y_true, 'g-', label='True function')
    
    # Plot the best fit function
    y_pred = np.polyval(best_solution, x_data)
    plt.plot(x_data, y_pred, 'r--', label='Genetic algorithm fit')
    
    plt.legend()
    plt.title('Polynomial Curve Fitting with Genetic Algorithm')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)
    
    # Save plot in run directory
    polynomial_fit_plot = run_dir / 'polynomial_fit_results.png'
    plt.savefig(polynomial_fit_plot)
    
    # Visualization of the optimization metrics
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot fitness evolution
    axs[0, 0].plot(optimizer.metrics["best_fitness"], 'b-', label='Best Fitness')
    axs[0, 0].plot(optimizer.metrics["avg_fitness"], 'g--', label='Avg Fitness')
    axs[0, 0].set_title('Fitness Evolution')
    axs[0, 0].set_xlabel('Generation')
    axs[0, 0].set_ylabel('Fitness (negative MSE)')
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)
    
    # Plot standard deviation of fitness
    axs[0, 1].plot(optimizer.metrics["std_fitness"], 'r-')
    axs[0, 1].set_title('Population Diversity')
    axs[0, 1].set_xlabel('Generation')
    axs[0, 1].set_ylabel('Std Dev of Fitness')
    axs[0, 1].grid(True, alpha=0.3)
    
    # Plot generation time
    axs[1, 0].plot(optimizer.metrics["generation_time"], 'g-')
    axs[1, 0].set_title('Computation Time per Generation')
    axs[1, 0].set_xlabel('Generation')
    axs[1, 0].set_ylabel('Time (seconds)')
    axs[1, 0].grid(True, alpha=0.3)
    
    # Plot memory usage
    axs[1, 1].plot(optimizer.metrics["memory_usage_mb"], 'm-')
    axs[1, 1].set_title('Memory Usage')
    axs[1, 1].set_xlabel('Generation')
    axs[1, 1].set_ylabel('Memory (MB)')
    axs[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot in run directory
    metrics_plot = run_dir / 'optimization_metrics.png'
    plt.savefig(metrics_plot)
    
    # Show both plots
    plt.show()
    
    # Return the optimizer for further analysis
    return optimizer


def main():
    """Entry point function for running the polynomial fit example."""
    parser = argparse.ArgumentParser(description="Run polynomial fit example with genetic algorithm")
    parser.add_argument("--no-monitor", action="store_true", help="Disable live monitor")
    parser.add_argument("--no-track", action="store_true", help="Disable population tracking")
    parser.add_argument("--no-analysis", action="store_true", help="Disable migration analysis")
    
    args = parser.parse_args()
    
    polynomial_fit_example(
        use_live_monitor=not args.no_monitor,
        track_population=not args.no_track,
        analyze_migration=not args.no_analysis
    )


if __name__ == "__main__":
    main() 