#!/usr/bin/env python
"""
Entry point script for the polynomial fitting example.

This script avoids the RuntimeWarning that occurs when running modules directly
with the -m flag by providing a clean entry point separate from the module structure.
"""

import sys
import argparse
import numpy as np
from pathlib import Path

# Add the parent directory to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

# Import the example function from the module
from genetic_opt.examples.polynomial_fit import polynomial_fit_example


def main():
    """Parse arguments and run the polynomial fit example."""
    parser = argparse.ArgumentParser(description="Run polynomial fit example with genetic algorithm")
    parser.add_argument("--no-monitor", action="store_true", help="Disable live monitor")
    parser.add_argument("--no-track", action="store_true", help="Disable population tracking")
    parser.add_argument("--no-analysis", action="store_true", help="Disable migration analysis")
    parser.add_argument("--enhanced-viz", action="store_true", help="Enable enhanced visualizations")
    parser.add_argument("--dim-reduction", choices=["none", "pca", "tsne"], default="pca",
                      help="Dimensionality reduction method (default: pca)")
    parser.add_argument("--no-dim-reduction", action="store_true", help="Disable dimensionality reduction visualization")
    
    args = parser.parse_args()
    
    # If enhanced visualizations are requested, use fitness landscape visualization
    if args.enhanced_viz and not args.no_track and not args.no_analysis:
        # Define polynomial fitness function for visualization
        def run_with_enhanced_viz():
            """Run the example with enhanced visualization for fitness landscape."""
            import random
            from genetic_opt.optimizer import SimpleGeneticAlgorithm
            from genetic_opt.utils import analyze_population_migration
            
            # Same random seeds as in the main example
            random.seed(42)
            np.random.seed(42)
            
            # Define the true polynomial: 2x^3 - 5x^2 + 3x - 1
            true_coefficients = [2, -5, 3, -1]
            
            # Generate data points
            x_data = np.linspace(-2, 2, 50)
            y_true = np.polyval(true_coefficients, x_data)
            
            # Add noise to create synthetic observations
            y_noisy = y_true + np.random.normal(0, 0.5, size=len(y_true))
            
            # Define fitness function (negative mean squared error)
            def polynomial_fitness(coefficients):
                """Calculate fitness for a set of polynomial coefficients."""
                y_pred = np.polyval(coefficients, x_data)
                mse = np.mean((y_pred - y_noisy) ** 2)
                return -mse  # Negative MSE for maximization
            
            # Run the example first
            optimizer = polynomial_fit_example(
                use_live_monitor=not args.no_monitor,
                track_population=True,  # Must be True for enhanced visualization
                analyze_migration=False  # We'll handle this separately with the fitness function
            )
            
            # Check if optimizer is None (shouldn't happen after our fix)
            if optimizer is None or not hasattr(optimizer, 'export_paths'):
                print("\nError: Could not access optimizer export paths. Make sure population history is being tracked.")
                return
                
            if 'population_history' not in optimizer.export_paths:
                print("\nError: No population history was exported. Please run again with population tracking enabled.")
                return
            
            # Determine dimensionality reduction settings
            include_dim_reduction = not args.no_dim_reduction
            dim_reduction_method = args.dim_reduction if args.dim_reduction != "none" else "pca"
            
            # Extract the run directory from optimizer paths
            run_dir = Path(optimizer.export_paths.get("run_directory", ""))
                
            # Now run the enhanced analysis with the fitness function
            print("\nGenerating enhanced visualizations (fitness landscapes, 3D migration, etc.)...")
            analysis_results = analyze_population_migration(
                population_file=optimizer.export_paths["population_history"],
                output_dir="analysis",
                run_dir=str(run_dir) if run_dir else None,
                create_animation=True,
                fitness_function=polynomial_fitness,
                include_3d=True,
                include_correlations=True,
                include_dim_reduction=include_dim_reduction,
                dim_reduction_method=dim_reduction_method
            )
            
            # Print analysis results
            print("\nEnhanced visualization results saved to:")
            print(f"- Statistics: {analysis_results['statistics_plot']}")
            
            print("- Density plots:")
            for name, path in analysis_results["density_plots"].items():
                print(f"  - {name}: {path}")
            
            print("- Animations:")
            for name, path in analysis_results["animations"].items():
                print(f"  - {name}: {path}")
            
            if analysis_results.get("fitness_landscapes"):
                print("- Fitness landscapes:")
                for name, path in analysis_results["fitness_landscapes"].items():
                    print(f"  - {name}: {path}")
            
            if analysis_results.get("fitness_animations"):
                print("- Fitness landscape animations:")
                for name, path in analysis_results["fitness_animations"].items():
                    print(f"  - {name}: {path}")
            
            if analysis_results.get("3d_visualization"):
                print(f"- 3D migration visualization: {analysis_results['3d_visualization']}")
            
            if analysis_results.get("correlation_plots"):
                print(f"- Gene correlation analysis: {analysis_results['correlation_plots']}")
                
            if include_dim_reduction:
                if analysis_results.get("reduced_space_plots"):
                    for key, path in analysis_results["reduced_space_plots"].items():
                        if key == 'reduced_space_plot':
                            print(f"- Dimensionality reduction ({dim_reduction_method.upper()}) visualization: {path}")
                
                if analysis_results.get("reduced_space_plots") and 'reduced_space_animation' in analysis_results['reduced_space_plots']:
                    print(f"- Dimensionality reduction animation: {analysis_results['reduced_space_plots']['reduced_space_animation']}")
        
        # Run with enhanced visualizations
        run_with_enhanced_viz()
    else:
        # Run with standard visualizations
        polynomial_fit_example(
            use_live_monitor=not args.no_monitor,
            track_population=not args.no_track,
            analyze_migration=not args.no_analysis
        )


if __name__ == "__main__":
    main() 