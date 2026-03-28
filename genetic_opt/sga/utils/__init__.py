"""Utility modules for genetic optimization."""

from genetic_opt.sga.utils.monitor import OptimizationMonitor
from genetic_opt.sga.utils.export import (
    export_metrics_to_csv,
    export_population_history_to_csv,
    export_run_metadata,
)
from genetic_opt.sga.utils.visualization import (
    plot_metrics,
    plot_population_density,
    create_population_migration_animation,
    plot_population_statistics,
    analyze_population_migration,
    plot_fitness_landscape,
    create_fitness_landscape_animation,
    plot_3d_migration_trajectory,
    plot_pairwise_correlations,
    plot_reduced_space_migration,
)

# Interactive Plotly visualization (optional dependency)
try:
    from genetic_opt.sga.utils.bio_visualization import create_evolution_chronicle
    _has_plotly = True
except ImportError:
    _has_plotly = False

__all__ = [
    # Monitor
    "OptimizationMonitor",
    # Export
    "export_metrics_to_csv",
    "export_population_history_to_csv",
    "export_run_metadata",
    # Visualization (matplotlib)
    "plot_metrics",
    "plot_population_density",
    "create_population_migration_animation",
    "plot_population_statistics",
    "analyze_population_migration",
    "plot_fitness_landscape",
    "create_fitness_landscape_animation",
    "plot_3d_migration_trajectory",
    "plot_pairwise_correlations",
    "plot_reduced_space_migration",
    # Interactive visualization (plotly)
    *(["create_evolution_chronicle"] if _has_plotly else []),
]
