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

from genetic_opt.sga.examples.polynomial_fit import polynomial_fit_example
from genetic_opt.sga.utils import analyze_population_migration


def main():
    """Parse arguments and run the polynomial fit example."""
    parser = argparse.ArgumentParser(
        description="Run polynomial fit example with genetic algorithm"
    )
    parser.add_argument(
        "--no-monitor", action="store_true", help="Disable live monitor"
    )
    parser.add_argument(
        "--no-track", action="store_true", help="Disable population tracking"
    )
    parser.add_argument(
        "--no-analysis", action="store_true", help="Disable migration analysis"
    )
    parser.add_argument(
        "--enhanced-viz", action="store_true", help="Enable enhanced visualizations"
    )
    parser.add_argument(
        "--dim-reduction",
        choices=["none", "pca", "tsne"],
        default="pca",
        help="Dimensionality reduction method (default: pca)",
    )
    parser.add_argument(
        "--no-dim-reduction",
        action="store_true",
        help="Disable dimensionality reduction visualization",
    )

    args = parser.parse_args()

    track_population = not args.no_track

    # Run the base example (skip its built-in analysis if we'll do enhanced)
    optimizer = polynomial_fit_example(
        use_live_monitor=not args.no_monitor,
        track_population=track_population,
        analyze_migration=not args.no_analysis and not args.enhanced_viz,
    )

    # If enhanced visualizations are requested, run the full analysis suite
    if args.enhanced_viz and track_population and not args.no_analysis:
        if optimizer is None or not hasattr(optimizer, "export_paths"):
            print(
                "\nError: Could not access optimizer export paths. "
                "Make sure population history is being tracked."
            )
            return

        if "population_history" not in optimizer.export_paths:
            print(
                "\nError: No population history was exported. "
                "Please run again with population tracking enabled."
            )
            return

        # Reconstruct the fitness function from the example's data
        import random

        random.seed(42)
        np.random.seed(42)
        true_coefficients = [2, -5, 3, -1]
        x_data = np.linspace(-2, 2, 50)
        y_true = np.polyval(true_coefficients, x_data)
        y_noisy = y_true + np.random.normal(0, 0.5, size=len(y_true))

        def polynomial_fitness(coefficients):
            y_pred = np.polyval(coefficients, x_data)
            mse = np.mean((y_pred - y_noisy) ** 2)
            return -mse

        include_dim_reduction = not args.no_dim_reduction
        dim_reduction_method = (
            args.dim_reduction if args.dim_reduction != "none" else "pca"
        )
        run_dir = Path(optimizer.export_paths.get("run_directory", ""))

        print(
            "\nGenerating enhanced visualizations "
            "(fitness landscapes, 3D migration, etc.)..."
        )
        analysis_results = analyze_population_migration(
            population_file=optimizer.export_paths["population_history"],
            output_dir="analysis",
            run_dir=str(run_dir) if run_dir else None,
            create_animation=True,
            fitness_function=polynomial_fitness,
            include_3d=True,
            include_correlations=True,
            include_dim_reduction=include_dim_reduction,
            dim_reduction_method=dim_reduction_method,
        )

        print("\nEnhanced visualization results saved to:")
        print(f"- Statistics: {analysis_results['statistics_plot']}")

        for section, label in [
            ("density_plots", "Density plots"),
            ("animations", "Animations"),
            ("fitness_landscapes", "Fitness landscapes"),
            ("fitness_animations", "Fitness landscape animations"),
        ]:
            if analysis_results.get(section):
                print(f"- {label}:")
                for name, path in analysis_results[section].items():
                    print(f"  - {name}: {path}")

        if analysis_results.get("3d_visualization"):
            print(
                f"- 3D migration visualization: "
                f"{analysis_results['3d_visualization']}"
            )

        if analysis_results.get("correlation_plots"):
            print(
                f"- Gene correlation analysis: "
                f"{analysis_results['correlation_plots']}"
            )

        if include_dim_reduction and analysis_results.get("reduced_space_plots"):
            for key, path in analysis_results["reduced_space_plots"].items():
                label = "visualization" if "plot" in key else "animation"
                print(
                    f"- Dimensionality reduction "
                    f"({dim_reduction_method.upper()}) {label}: {path}"
                )


if __name__ == "__main__":
    main()
