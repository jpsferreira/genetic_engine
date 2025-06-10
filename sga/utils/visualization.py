"""Visualization utilities for genetic optimization analysis."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Tuple, Dict, Optional, Union, Any
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


def plot_metrics(
    metrics_file: str, 
    output_file: Optional[str] = None, 
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """Plot optimization metrics from a CSV file.
    
    Args:
        metrics_file: Path to the metrics CSV file
        output_file: Optional path to save the figure
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure
    """
    # Load metrics data
    metrics_df = pd.read_csv(metrics_file)
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Optimization Metrics', fontsize=16)
    
    # Plot fitness evolution
    if 'best_fitness' in metrics_df.columns and 'avg_fitness' in metrics_df.columns:
        axs[0, 0].plot(metrics_df['generation'], metrics_df['best_fitness'], 'b-', label='Best Fitness')
        axs[0, 0].plot(metrics_df['generation'], metrics_df['avg_fitness'], 'g--', label='Avg Fitness')
        axs[0, 0].set_title('Fitness Evolution')
        axs[0, 0].set_xlabel('Generation')
        axs[0, 0].set_ylabel('Fitness')
        axs[0, 0].legend()
        axs[0, 0].grid(True, alpha=0.3)
    
    # Plot standard deviation of fitness
    if 'std_fitness' in metrics_df.columns:
        axs[0, 1].plot(metrics_df['generation'], metrics_df['std_fitness'], 'r-')
        axs[0, 1].set_title('Population Diversity')
        axs[0, 1].set_xlabel('Generation')
        axs[0, 1].set_ylabel('Std Dev of Fitness')
        axs[0, 1].grid(True, alpha=0.3)
    
    # Plot generation time
    if 'generation_time' in metrics_df.columns:
        axs[1, 0].plot(metrics_df['generation'], metrics_df['generation_time'], 'g-')
        axs[1, 0].set_title('Computation Time per Generation')
        axs[1, 0].set_xlabel('Generation')
        axs[1, 0].set_ylabel('Time (seconds)')
        axs[1, 0].grid(True, alpha=0.3)
    
    # Plot memory usage
    if 'memory_usage_mb' in metrics_df.columns:
        axs[1, 1].plot(metrics_df['generation'], metrics_df['memory_usage_mb'], 'm-')
        axs[1, 1].set_title('Memory Usage')
        axs[1, 1].set_xlabel('Generation')
        axs[1, 1].set_ylabel('Memory (MB)')
        axs[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300)
    
    return fig


def plot_population_density(
    population_file: str,
    gene_indices: List[int] = [0, 1],
    generations: Optional[List[int]] = None,
    bins: int = 20,
    figsize: Tuple[int, int] = (12, 10),
    output_file: Optional[str] = None,
    cmap: str = 'viridis',
) -> plt.Figure:
    """Plot the population density across generations.
    
    Args:
        population_file: Path to the population history CSV file
        gene_indices: Indices of genes to plot (2D)
        generations: Specific generations to plot (if None, selects evenly spaced generations)
        bins: Number of bins for the 2D histogram
        figsize: Figure size (width, height)
        output_file: Optional path to save the figure
        cmap: Colormap for the density plot
        
    Returns:
        Matplotlib figure
    """
    # Load population data
    pop_df = pd.read_csv(population_file)
    
    # Verify gene columns exist
    gene_cols = [f'gene_{i+1}' for i in gene_indices]
    for col in gene_cols:
        if col not in pop_df.columns:
            raise ValueError(f"Column {col} not found in population data")
    
    # Get unique generations
    unique_gens = sorted(pop_df['generation'].unique())
    
    # Select generations to plot
    if generations is None:
        # Choose at most 6 evenly spaced generations
        n_plots = min(6, len(unique_gens))
        if n_plots > 1:
            selected_gens = [unique_gens[i] for i in 
                             np.linspace(0, len(unique_gens)-1, n_plots).astype(int)]
        else:
            selected_gens = unique_gens
    else:
        # Use provided generations if they exist in the data
        selected_gens = [g for g in generations if g in unique_gens]
    
    # Calculate subplot layout
    n_plots = len(selected_gens)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    # Create figure and axes
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    fig.suptitle(f'Population Density Evolution (Genes {gene_indices[0]+1} & {gene_indices[1]+1})', 
                 fontsize=16)
    
    # Get data ranges for consistent plots
    x_min = pop_df[gene_cols[0]].min()
    x_max = pop_df[gene_cols[0]].max()
    y_min = pop_df[gene_cols[1]].min()
    y_max = pop_df[gene_cols[1]].max()
    
    # Add some padding to ranges
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad
    
    # Plot each generation
    for i, gen in enumerate(selected_gens):
        row = i // n_cols
        col = i % n_cols
        ax = axs[row, col]
        
        # Get data for this generation
        gen_data = pop_df[pop_df['generation'] == gen]
        
        # Create 2D histogram
        h, xedges, yedges, img = ax.hist2d(
            gen_data[gene_cols[0]], 
            gen_data[gene_cols[1]], 
            bins=bins, 
            range=[[x_min, x_max], [y_min, y_max]],
            cmap=cmap
        )
        
        ax.set_title(f'Generation {gen}')
        ax.set_xlabel(f'Gene {gene_indices[0]+1}')
        ax.set_ylabel(f'Gene {gene_indices[1]+1}')
        fig.colorbar(img, ax=ax, label='Population Count')
    
    # Hide unused subplots
    for i in range(n_plots, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axs[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save figure if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300)
    
    return fig


def create_population_migration_animation(
    population_file: str,
    gene_indices: List[int] = [0, 1],
    fps: int = 5,
    bins: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    output_file: Optional[str] = None,
    cmap: str = 'viridis',
    skip_frames: int = 1,
) -> FuncAnimation:
    """Create an animation of population migration across generations.
    
    Args:
        population_file: Path to the population history CSV file
        gene_indices: Indices of genes to plot (2D)
        fps: Frames per second for the animation
        bins: Number of bins for the 2D histogram
        figsize: Figure size (width, height)
        output_file: Optional path to save the animation (as .mp4 or .gif)
        cmap: Colormap for the density plot
        skip_frames: Number of generations to skip between frames
        
    Returns:
        Animation object
    """
    # Load population data
    pop_df = pd.read_csv(population_file)
    
    # Verify gene columns exist
    gene_cols = [f'gene_{i+1}' for i in gene_indices]
    for col in gene_cols:
        if col not in pop_df.columns:
            raise ValueError(f"Column {col} not found in population data")
    
    # Get unique generations
    unique_gens = sorted(pop_df['generation'].unique())
    
    # Skip frames if requested
    if skip_frames > 1:
        unique_gens = unique_gens[::skip_frames]
    
    # Get data ranges for consistent plots
    x_min = pop_df[gene_cols[0]].min()
    x_max = pop_df[gene_cols[0]].max()
    y_min = pop_df[gene_cols[1]].min()
    y_max = pop_df[gene_cols[1]].max()
    
    # Add some padding to ranges
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel(f'Gene {gene_indices[0]+1}')
    ax.set_ylabel(f'Gene {gene_indices[1]+1}')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Initialize histogram (returns the histogram itself, bin edges, and quad mesh object)
    h, xedges, yedges, quad = ax.hist2d(
        pop_df[pop_df['generation'] == unique_gens[0]][gene_cols[0]],
        pop_df[pop_df['generation'] == unique_gens[0]][gene_cols[1]],
        bins=bins,
        range=[[x_min, x_max], [y_min, y_max]],
        cmap=cmap
    )
    
    # Add colorbar
    cbar = fig.colorbar(quad, ax=ax, label='Population Count')
    
    # Add generation text
    gen_text = ax.text(
        0.02, 0.98, f'Generation: {unique_gens[0]}',
        transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )
    
    # Update function for animation
    def update(frame):
        gen = unique_gens[frame]
        gen_data = pop_df[pop_df['generation'] == gen]
        
        # Clear previous data
        ax.cla()
        
        # Plot new data
        h, xedges, yedges, quad = ax.hist2d(
            gen_data[gene_cols[0]], 
            gen_data[gene_cols[1]], 
            bins=bins, 
            range=[[x_min, x_max], [y_min, y_max]],
            cmap=cmap
        )
        
        # Update axes labels and limits
        ax.set_xlabel(f'Gene {gene_indices[0]+1}')
        ax.set_ylabel(f'Gene {gene_indices[1]+1}')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Update generation text
        gen_text = ax.text(
            0.02, 0.98, f'Generation: {gen}',
            transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
        
        return quad, gen_text
    
    # Create animation
    anim = FuncAnimation(
        fig, update, frames=len(unique_gens), blit=False, interval=1000/fps
    )
    
    # Save animation if output file is specified
    if output_file:
        if output_file.endswith('.gif'):
            anim.save(output_file, writer='pillow', fps=fps, dpi=100)
        elif output_file.endswith('.mp4'):
            anim.save(output_file, writer='ffmpeg', fps=fps, dpi=200)
        else:
            raise ValueError("Output file must be .gif or .mp4")
    
    return anim


def plot_population_statistics(
    population_file: str,
    metrics_file: Optional[str] = None,
    output_file: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10)
) -> plt.Figure:
    """Plot various population statistics across generations.
    
    Args:
        population_file: Path to the population history CSV file
        metrics_file: Optional path to metrics CSV file for additional statistics
        output_file: Optional path to save the figure
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure
    """
    # Load population data
    pop_df = pd.read_csv(population_file)
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Population Statistics', fontsize=16)
    
    # Get gene columns
    gene_cols = [col for col in pop_df.columns if col.startswith('gene_')]
    
    # Calculate gene statistics
    stats_by_gen = []
    for gen in sorted(pop_df['generation'].unique()):
        gen_data = pop_df[pop_df['generation'] == gen]
        
        # Calculate statistics for each gene
        gene_means = [gen_data[col].mean() for col in gene_cols]
        gene_stds = [gen_data[col].std() for col in gene_cols]
        
        stats_by_gen.append({
            'generation': gen,
            'gene_means': gene_means,
            'gene_stds': gene_stds,
        })
    
    # Convert to DataFrame
    stats_df = pd.DataFrame(stats_by_gen)
    
    # Plot mean values for each gene
    ax = axs[0, 0]
    for i, col in enumerate(gene_cols):
        gene_means = [stats['gene_means'][i] for stats in stats_by_gen]
        ax.plot(stats_df['generation'], gene_means, label=f'Gene {i+1}')
    
    ax.set_title('Gene Mean Values')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Mean Value')
    if len(gene_cols) <= 10:  # Only show legend if not too many genes
        ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot standard deviations for each gene
    ax = axs[0, 1]
    for i, col in enumerate(gene_cols):
        gene_stds = [stats['gene_stds'][i] for stats in stats_by_gen]
        ax.plot(stats_df['generation'], gene_stds, label=f'Gene {i+1}')
    
    ax.set_title('Gene Standard Deviations')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Standard Deviation')
    if len(gene_cols) <= 10:  # Only show legend if not too many genes
        ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Load metrics data if provided
    if metrics_file and Path(metrics_file).exists():
        metrics_df = pd.read_csv(metrics_file)
        
        # Plot fitness evolution if available
        if 'best_fitness' in metrics_df.columns:
            ax = axs[1, 0]
            ax.plot(metrics_df['generation'], metrics_df['best_fitness'], 'b-', label='Best Fitness')
            if 'avg_fitness' in metrics_df.columns:
                ax.plot(metrics_df['generation'], metrics_df['avg_fitness'], 'g--', label='Avg Fitness')
            ax.set_title('Fitness Evolution')
            ax.set_xlabel('Generation')
            ax.set_ylabel('Fitness')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot population diversity if available
        if 'std_fitness' in metrics_df.columns:
            ax = axs[1, 1]
            ax.plot(metrics_df['generation'], metrics_df['std_fitness'], 'r-')
            ax.set_title('Population Diversity')
            ax.set_xlabel('Generation')
            ax.set_ylabel('Std Dev of Fitness')
            ax.grid(True, alpha=0.3)
    else:
        # Compute gene diversity (sum of standard deviations)
        axs[1, 0].plot(stats_df['generation'], 
                    [sum(stats['gene_stds']) for stats in stats_by_gen])
        axs[1, 0].set_title('Gene Diversity (Sum of Std Devs)')
        axs[1, 0].set_xlabel('Generation')
        axs[1, 0].set_ylabel('Total Standard Deviation')
        axs[1, 0].grid(True, alpha=0.3)
        
        # Turn off the unused subplot
        axs[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Save figure if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300)
    
    return fig


def analyze_population_migration(
    population_file: str,
    output_dir: str = "analysis",
    run_dir: Optional[str] = None,
    gene_pairs: Optional[List[Tuple[int, int]]] = None,
    create_animation: bool = True,
    fitness_function: Optional[callable] = None,
    include_3d: bool = True,
    include_correlations: bool = True,
    include_dim_reduction: bool = True,
    dim_reduction_method: str = 'pca'
) -> Dict[str, Any]:
    """Analyze population migration and generate visualizations.
    
    Args:
        population_file: Path to the population history CSV file
        output_dir: Base directory to save results
        run_dir: Optional path to the run directory (contains timestamp), will be used to organize outputs
        gene_pairs: List of gene pairs to analyze (default: all combinations of first 3 genes)
        create_animation: Whether to create animations
        fitness_function: Optional fitness function for landscape visualization
        include_3d: Whether to include 3D migration trajectory visualization
        include_correlations: Whether to include gene correlation analysis
        include_dim_reduction: Whether to include dimensionality reduction visualization
        dim_reduction_method: Method for dimensionality reduction
        
    Returns:
        Dictionary with analysis results
    """
    # Load population data
    pop_df = pd.read_csv(population_file)
    
    # Get gene columns
    gene_cols = [col for col in pop_df.columns if col.startswith('gene_')]
    n_genes = len(gene_cols)
    
    # Determine gene pairs to analyze
    if gene_pairs is None:
        # Use all combinations of first 3 genes (or all if less than 3)
        n = min(3, n_genes)
        gene_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Extract base filename and timestamp (if available)
    population_path = Path(population_file)
    base_filename = population_path.stem
    
    # Determine analysis directory
    if run_dir:
        # Use the provided run directory to organize by timestamp
        run_path = Path(run_dir)
        timestamp = run_path.name  # Extract timestamp from the run directory
        analysis_dir = Path(output_dir) / timestamp
    else:
        # If no run directory is provided, try to extract timestamp from the filename
        timestamp_parts = base_filename.split('_')
        if len(timestamp_parts) >= 3 and timestamp_parts[-2].isdigit() and timestamp_parts[-1].isdigit():
            # Assuming format "optimization_run_YYYYMMDD_HHMMSS"
            timestamp = f"{timestamp_parts[-2]}_{timestamp_parts[-1]}"
            analysis_dir = Path(output_dir) / timestamp
        else:
            # No timestamp available, use base directory
            analysis_dir = Path(output_dir)
    
    # Create analysis directory
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate statistics plot
    metrics_file_path = population_path.parent / Path(base_filename.replace('_population', '_metrics')).with_suffix('.csv')
    metrics_file = str(metrics_file_path) if metrics_file_path.exists() else None
    
    stats_file = analysis_dir / f"{base_filename}_statistics.png"
    stats_fig = plot_population_statistics(
        population_file,
        metrics_file=metrics_file,
        output_file=stats_file,
        figsize=(12, 10)
    )
    
    # Generate density plots and animations for each gene pair
    results = {
        "statistics_plot": str(stats_file),
        "density_plots": {},
        "animations": {},
        "fitness_landscapes": {},
        "fitness_animations": {},
        "3d_visualization": None,
        "correlation_plots": None,
        "reduced_space_plots": {},
        "reduced_space_animations": {}
    }
    
    for gene_pair in gene_pairs:
        i, j = gene_pair
        
        # Generate density plot
        density_file = analysis_dir / f"{base_filename}_density_genes_{i+1}_{j+1}.png"
        density_fig = plot_population_density(
            population_file,
            gene_indices=[i, j],
            output_file=density_file
        )
        results["density_plots"][f"genes_{i+1}_{j+1}"] = str(density_file)
        
        # Generate animation if requested
        if create_animation:
            anim_file = analysis_dir / f"{base_filename}_animation_genes_{i+1}_{j+1}.mp4"
            anim = create_population_migration_animation(
                population_file,
                gene_indices=[i, j],
                output_file=anim_file
            )
            results["animations"][f"genes_{i+1}_{j+1}"] = str(anim_file)
            
        # Generate fitness landscape if fitness function is provided
        if fitness_function is not None:
            # Static landscape with population overlay
            landscape_file = analysis_dir / f"{base_filename}_fitness_landscape_genes_{i+1}_{j+1}.png"
            landscape_fig = plot_fitness_landscape(
                population_file,
                fitness_function=fitness_function,
                gene_indices=[i, j],
                output_file=landscape_file
            )
            results["fitness_landscapes"][f"genes_{i+1}_{j+1}"] = str(landscape_file)
            
            # Animated landscape if animations are enabled
            if create_animation:
                landscape_anim_file = analysis_dir / f"{base_filename}_fitness_landscape_animation_genes_{i+1}_{j+1}.mp4"
                landscape_anim = create_fitness_landscape_animation(
                    population_file,
                    fitness_function=fitness_function,
                    gene_indices=[i, j],
                    output_file=landscape_anim_file
                )
                results["fitness_animations"][f"genes_{i+1}_{j+1}"] = str(landscape_anim_file)
    
    # Generate 3D migration trajectory if requested and if we have at least 3 genes
    if include_3d and n_genes >= 3:
        # Get first three genes
        threed_file = analysis_dir / f"{base_filename}_3d_migration.png"
        threed_fig = plot_3d_migration_trajectory(
            population_file,
            gene_indices=[0, 1, 2],  # First three genes
            output_file=threed_file
        )
        results["3d_visualization"] = str(threed_file)
    
    # Generate correlation analysis if requested
    if include_correlations:
        corr_file = analysis_dir / f"{base_filename}_gene_correlations.png"
        corr_fig = plot_pairwise_correlations(
            population_file,
            output_file=corr_file
        )
        results["correlation_plots"] = str(corr_file)
    
    # Add dimensionality reduction visualization if requested
    if include_dim_reduction and len(gene_cols) >= 2:
        reduced_space_paths = plot_reduced_space_migration(
            pop_df,
            str(analysis_dir),
            base_filename,
            method=dim_reduction_method,
            n_components=min(3, len(gene_cols)),
            create_animation=create_animation
        )
        results["reduced_space_plots"].update(reduced_space_paths)
    
    # Close all figures to save memory
    plt.close('all')
    
    return results


def plot_fitness_landscape(
    population_file: str,
    fitness_function: Optional[callable] = None,
    gene_indices: List[int] = [0, 1],
    generations: Optional[List[int]] = None,
    resolution: int = 50,
    figsize: Tuple[int, int] = (12, 10),
    output_file: Optional[str] = None,
    cmap: str = 'viridis',
    population_color: str = 'red',
    alpha: float = 0.5,
) -> plt.Figure:
    """Plot the fitness landscape with population individuals overlaid.
    
    Args:
        population_file: Path to the population history CSV file
        fitness_function: Function to calculate fitness for landscape (required)
        gene_indices: Indices of genes to plot (2D)
        generations: Specific generations to plot (if None, selects evenly spaced generations)
        resolution: Number of points for the fitness landscape grid
        figsize: Figure size (width, height)
        output_file: Optional path to save the figure
        cmap: Colormap for the fitness landscape
        population_color: Color for population scatter points
        alpha: Transparency of population points
        
    Returns:
        Matplotlib figure
    """
    # Load population data
    pop_df = pd.read_csv(population_file)
    
    # Verify gene columns exist
    gene_cols = [f'gene_{i+1}' for i in gene_indices]
    for col in gene_cols:
        if col not in pop_df.columns:
            raise ValueError(f"Column {col} not found in population data")
    
    # Get unique generations
    unique_gens = sorted(pop_df['generation'].unique())
    
    # Select generations to plot
    if generations is None:
        # Choose at most 6 evenly spaced generations
        n_plots = min(6, len(unique_gens))
        if n_plots > 1:
            selected_gens = [unique_gens[i] for i in 
                           np.linspace(0, len(unique_gens)-1, n_plots).astype(int)]
        else:
            selected_gens = unique_gens
    else:
        # Use provided generations if they exist in the data
        selected_gens = [g for g in generations if g in unique_gens]
    
    # Calculate subplot layout
    n_plots = len(selected_gens)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    # Create figure and axes
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    fig.suptitle(f'Fitness Landscape and Population Evolution (Genes {gene_indices[0]+1} & {gene_indices[1]+1})', 
                 fontsize=16)
    
    # Get data ranges for plots
    x_min = pop_df[gene_cols[0]].min()
    x_max = pop_df[gene_cols[0]].max()
    y_min = pop_df[gene_cols[1]].min()
    y_max = pop_df[gene_cols[1]].max()
    
    # Add some padding to ranges
    x_pad = (x_max - x_min) * 0.1
    y_pad = (y_max - y_min) * 0.1
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad
    
    # Check if fitness function is provided
    if fitness_function is None:
        raise ValueError("A fitness function must be provided to generate the landscape")
    
    # Generate fitness landscape grid
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Calculate fitness values
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            # Create a chromosome with default values to evaluate fitness
            chromosome = [0.0] * max(gene_indices) + [0.0]  # +1 to ensure enough elements
            chromosome[gene_indices[0]] = X[i, j]
            chromosome[gene_indices[1]] = Y[i, j]
            Z[i, j] = fitness_function(chromosome)  # Calculate fitness
    
    # Plot each generation
    for i, gen in enumerate(selected_gens):
        row = i // n_cols
        col = i % n_cols
        ax = axs[row, col]
        
        # Plot fitness landscape contour
        contour = ax.contourf(X, Y, Z, levels=50, cmap=cmap, alpha=0.7)
        
        # Get data for this generation
        gen_data = pop_df[pop_df['generation'] == gen]
        
        # Plot individuals as scatter points
        scatter = ax.scatter(
            gen_data[gene_cols[0]], 
            gen_data[gene_cols[1]], 
            c=population_color,
            s=20,
            alpha=alpha,
            edgecolor='white',
            linewidth=0.5
        )
        
        ax.set_title(f'Generation {gen}')
        ax.set_xlabel(f'Gene {gene_indices[0]+1}')
        ax.set_ylabel(f'Gene {gene_indices[1]+1}')
    
    # Add a single colorbar for the fitness landscape
    cbar = fig.colorbar(contour, ax=axs.ravel().tolist(), label='Fitness')
    
    # Hide unused subplots
    for i in range(n_plots, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axs[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save figure if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300)
    
    return fig


def create_fitness_landscape_animation(
    population_file: str,
    fitness_function: callable,
    gene_indices: List[int] = [0, 1],
    fps: int = 5,
    resolution: int = 50,
    figsize: Tuple[int, int] = (10, 8),
    output_file: Optional[str] = None,
    cmap: str = 'viridis',
    population_color: str = 'red',
    alpha: float = 0.7,
    skip_frames: int = 1,
    show_best: bool = True,
) -> FuncAnimation:
    """Create an animation showing population evolution over a fitness landscape.
    
    Args:
        population_file: Path to the population history CSV file
        fitness_function: Function to calculate fitness for landscape
        gene_indices: Indices of genes to plot (2D)
        fps: Frames per second for the animation
        resolution: Number of points for the fitness landscape grid
        figsize: Figure size (width, height)
        output_file: Optional path to save the animation (as .mp4 or .gif)
        cmap: Colormap for the fitness landscape
        population_color: Color for population scatter points
        alpha: Transparency of population points
        skip_frames: Number of generations to skip between frames
        show_best: Whether to highlight the best individual
        
    Returns:
        Animation object
    """
    # Load population data
    pop_df = pd.read_csv(population_file)
    
    # Verify gene columns exist
    gene_cols = [f'gene_{i+1}' for i in gene_indices]
    for col in gene_cols:
        if col not in pop_df.columns:
            raise ValueError(f"Column {col} not found in population data")
    
    # Get unique generations
    unique_gens = sorted(pop_df['generation'].unique())
    
    # Skip frames if requested
    if skip_frames > 1:
        unique_gens = unique_gens[::skip_frames]
    
    # Get data ranges for consistent plots
    x_min = pop_df[gene_cols[0]].min()
    x_max = pop_df[gene_cols[0]].max()
    y_min = pop_df[gene_cols[1]].min()
    y_max = pop_df[gene_cols[1]].max()
    
    # Add some padding to ranges
    x_pad = (x_max - x_min) * 0.1
    y_pad = (y_max - y_min) * 0.1
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad
    
    # Generate fitness landscape grid
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Calculate fitness values
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            # Create a chromosome with default values to evaluate fitness
            chromosome = [0.0] * max(gene_indices) + [0.0]  # +1 to ensure enough elements
            chromosome[gene_indices[0]] = X[i, j]
            chromosome[gene_indices[1]] = Y[i, j]
            Z[i, j] = fitness_function(chromosome)  # Calculate fitness
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the fitness landscape
    contour = ax.contourf(X, Y, Z, levels=50, cmap=cmap, alpha=0.7)
    
    # Initialize empty scatter plot for population
    scatter = ax.scatter([], [], c=population_color, s=20, alpha=alpha, edgecolor='white', linewidth=0.5)
    
    # Initialize empty scatter for best individual
    best_scatter = ax.scatter([], [], c='yellow', s=100, alpha=1.0, marker='*', edgecolor='black', linewidth=1.0)
    
    # Add labels
    ax.set_xlabel(f'Gene {gene_indices[0]+1}')
    ax.set_ylabel(f'Gene {gene_indices[1]+1}')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Add colorbar
    cbar = fig.colorbar(contour, ax=ax, label='Fitness')
    
    # Add generation text
    gen_text = ax.text(
        0.02, 0.98, f'Generation: {unique_gens[0]}',
        transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
    )
    
    # Update function for animation
    def update(frame):
        gen = unique_gens[frame]
        gen_data = pop_df[pop_df['generation'] == gen]
        
        # Update scatter plot data
        scatter.set_offsets(np.column_stack((gen_data[gene_cols[0]], gen_data[gene_cols[1]])))
        
        # Update best individual if requested
        if show_best:
            # Find the best individual for this generation
            if 'fitness' in gen_data.columns:
                # If fitness is directly available in the data
                best_idx = gen_data['fitness'].idxmax()
                best_x = gen_data.loc[best_idx, gene_cols[0]]
                best_y = gen_data.loc[best_idx, gene_cols[1]]
            else:
                # Calculate fitness for each individual
                best_fitness = -float('inf')
                best_x, best_y = 0, 0
                
                for _, row in gen_data.iterrows():
                    # Create a chromosome with default values to evaluate fitness
                    chromosome = [0.0] * max(gene_indices) + [0.0]  # +1 to ensure enough elements
                    for i, col in enumerate(gene_cols):
                        if i < len(chromosome):
                            chromosome[i] = row[col]
                    
                    # Calculate fitness
                    fitness = fitness_function(chromosome)
                    
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_x = row[gene_cols[0]]
                        best_y = row[gene_cols[1]]
            
            best_scatter.set_offsets(np.array([[best_x, best_y]]))
        
        # Update generation text
        gen_text.set_text(f'Generation: {gen}')
        
        # Return all objects that were updated
        if show_best:
            return scatter, best_scatter, gen_text
        else:
            return scatter, gen_text
    
    # Create animation
    anim = FuncAnimation(
        fig, update, frames=len(unique_gens), blit=True, interval=1000/fps
    )
    
    # Save animation if output file is specified
    if output_file:
        if output_file.endswith('.gif'):
            anim.save(output_file, writer='pillow', fps=fps, dpi=100)
        elif output_file.endswith('.mp4'):
            anim.save(output_file, writer='ffmpeg', fps=fps, dpi=200)
        else:
            raise ValueError("Output file must be .gif or .mp4")
    
    return anim 


def plot_3d_migration_trajectory(
    population_file: str,
    gene_indices: List[int] = [0, 1, 2],
    color_by: str = 'generation',
    figsize: Tuple[int, int] = (12, 10),
    output_file: Optional[str] = None,
    marker_size: int = 15,
    density_plot: bool = True,
) -> plt.Figure:
    """Create a 3D visualization of population migration across generations.
    
    Args:
        population_file: Path to the population history CSV file
        gene_indices: Indices of three genes to plot (3D)
        color_by: How to color points ('generation' or 'density')
        figsize: Figure size (width, height)
        output_file: Optional path to save the figure
        marker_size: Size of markers in the scatter plot
        density_plot: Whether to add 2D density plots on each plane
        
    Returns:
        Matplotlib figure
    """
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.colors import LinearSegmentedColormap
    
    # Load population data
    pop_df = pd.read_csv(population_file)
    
    # Verify gene columns exist
    if len(gene_indices) != 3:
        raise ValueError("Exactly three gene indices must be provided for 3D visualization")
        
    gene_cols = [f'gene_{i+1}' for i in gene_indices]
    for col in gene_cols:
        if col not in pop_df.columns:
            raise ValueError(f"Column {col} not found in population data")
    
    # Create figure and 3D axes
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Get data for each gene
    x = pop_df[gene_cols[0]].values
    y = pop_df[gene_cols[1]].values
    z = pop_df[gene_cols[2]].values
    
    # Color by generation or density
    if color_by == 'generation':
        # Color by generation (from blue to red)
        generations = pop_df['generation'].values
        unique_gens = sorted(pop_df['generation'].unique())
        norm = plt.Normalize(min(unique_gens), max(unique_gens))
        colors = plt.cm.viridis(norm(generations))
        
        # Plot population scatter
        scatter = ax.scatter(x, y, z, c=generations, cmap='viridis', s=marker_size, alpha=0.6)
        
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax, label='Generation')
        
    elif color_by == 'density':
        # Color by point density
        from scipy.stats import gaussian_kde
        
        # Calculate point density
        xyz = np.vstack([x, y, z])
        density = gaussian_kde(xyz)(xyz)
        
        # Sort points by density
        idx = density.argsort()
        x, y, z, density = x[idx], y[idx], z[idx], density[idx]
        
        # Create custom colormap from blue to yellow to red
        colors = plt.cm.plasma(plt.Normalize()(density))
        
        # Plot population scatter
        scatter = ax.scatter(x, y, z, c=density, cmap='plasma', s=marker_size, alpha=0.6)
        
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax, label='Point Density')
    
    else:
        raise ValueError("color_by must be 'generation' or 'density'")
    
    # Add density plots on each plane if requested
    if density_plot:
        # Add 2D density plots on each face of the 3D plot
        # XY plane (bottom)
        try:
            from scipy.stats import gaussian_kde
            
            # Get min and max for each dimension
            x_min, x_max = min(x), max(x)
            y_min, y_max = min(y), max(y)
            z_min, z_max = min(z), max(z)
            
            # Add some padding
            x_pad = (x_max - x_min) * 0.05
            y_pad = (y_max - y_min) * 0.05
            z_pad = (z_max - z_min) * 0.05
            
            x_min -= x_pad; x_max += x_pad
            y_min -= y_pad; y_max += y_pad
            z_min -= z_pad; z_max += z_pad
            
            # Create grids for each plane
            resolution = 50
            xx_xy, yy_xy = np.mgrid[x_min:x_max:resolution*1j, y_min:y_max:resolution*1j]
            xx_xz, zz_xz = np.mgrid[x_min:x_max:resolution*1j, z_min:z_max:resolution*1j]
            yy_yz, zz_yz = np.mgrid[y_min:y_max:resolution*1j, z_min:z_max:resolution*1j]
            
            # Calculate densities for each plane
            xy_positions = np.vstack([xx_xy.ravel(), yy_xy.ravel()])
            xz_positions = np.vstack([xx_xz.ravel(), zz_xz.ravel()])
            yz_positions = np.vstack([yy_yz.ravel(), zz_yz.ravel()])
            
            xy_kernel = gaussian_kde(np.vstack([x, y]))
            xz_kernel = gaussian_kde(np.vstack([x, z]))
            yz_kernel = gaussian_kde(np.vstack([y, z]))
            
            xy_density = np.reshape(xy_kernel(xy_positions), xx_xy.shape)
            xz_density = np.reshape(xz_kernel(xz_positions), xx_xz.shape)
            yz_density = np.reshape(yz_kernel(yz_positions), yy_yz.shape)
            
            # Plot densities using appropriate alpha to not obscure points
            alpha = 0.3
            
            # XY plane (bottom)
            ax.contourf(xx_xy, yy_xy, xy_density, zdir='z', offset=z_min, 
                       cmap='Blues', alpha=alpha, levels=10)
            
            # XZ plane (back)
            ax.contourf(xx_xz, xz_density, zz_xz, zdir='y', offset=y_max, 
                       cmap='Greens', alpha=alpha, levels=10)
            
            # YZ plane (right side)
            ax.contourf(yz_density, yy_yz, zz_yz, zdir='x', offset=x_max, 
                       cmap='Reds', alpha=alpha, levels=10)
        except:
            # If density plot fails, just skip it
            pass
    
    # Add labels and title
    ax.set_xlabel(f'Gene {gene_indices[0]+1}')
    ax.set_ylabel(f'Gene {gene_indices[1]+1}')
    ax.set_zlabel(f'Gene {gene_indices[2]+1}')
    ax.set_title('3D Population Migration Trajectory')
    
    # Save figure if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return fig


def plot_pairwise_correlations(
    population_file: str,
    generations: Optional[List[int]] = None,
    output_file: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    cmap: str = 'coolwarm',
) -> plt.Figure:
    """Create a correlation matrix visualization between genes across generations.
    
    Args:
        population_file: Path to the population history CSV file
        generations: Specific generations to analyze (if None, selects evenly spaced generations)
        output_file: Optional path to save the figure
        figsize: Figure size (width, height), auto-calculated if None
        cmap: Colormap for correlation heatmap
        
    Returns:
        Matplotlib figure
    """
    # Load population data
    pop_df = pd.read_csv(population_file)
    
    # Get gene columns
    gene_cols = [col for col in pop_df.columns if col.startswith('gene_')]
    n_genes = len(gene_cols)
    
    # Get unique generations
    unique_gens = sorted(pop_df['generation'].unique())
    
    # Select generations to plot
    if generations is None:
        # Choose at most 4 evenly spaced generations
        n_plots = min(4, len(unique_gens))
        if n_plots > 1:
            selected_gens = [unique_gens[i] for i in 
                           np.linspace(0, len(unique_gens)-1, n_plots).astype(int)]
        else:
            selected_gens = unique_gens
    else:
        # Use provided generations if they exist in the data
        selected_gens = [g for g in generations if g in unique_gens]
    
    # Calculate auto figsize if not provided
    if figsize is None:
        figsize = (3 * len(selected_gens), 3 * n_genes // 2)
    
    # Create figure with subplots for each generation
    fig, axs = plt.subplots(1, len(selected_gens), figsize=figsize)
    
    # Handle single subplot case
    if len(selected_gens) == 1:
        axs = [axs]
    
    # Calculate correlation matrices for each generation
    for i, gen in enumerate(selected_gens):
        gen_data = pop_df[pop_df['generation'] == gen][gene_cols]
        corr_matrix = gen_data.corr()
        
        # Plot correlation matrix
        im = axs[i].imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1)
        
        # Add correlation values
        for ii in range(n_genes):
            for jj in range(n_genes):
                color = 'white' if abs(corr_matrix.iloc[ii, jj]) > 0.6 else 'black'
                axs[i].text(jj, ii, f'{corr_matrix.iloc[ii, jj]:.2f}', 
                           ha='center', va='center', color=color, fontsize=8)
        
        # Set title and labels
        axs[i].set_title(f'Generation {gen}')
        axs[i].set_xticks(range(n_genes))
        axs[i].set_yticks(range(n_genes))
        axs[i].set_xticklabels([f'G{i+1}' for i in range(n_genes)], rotation=45)
        axs[i].set_yticklabels([f'G{i+1}' for i in range(n_genes)])
    
    # Add colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Correlation')
    
    # Add overall title
    fig.suptitle('Gene Correlation Evolution', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    # Save figure if output file is specified
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return fig 


def plot_reduced_space_migration(population_data, output_dir, output_prefix, method='pca', n_components=2, create_animation=True):
    """
    Visualize population migration in a reduced dimensional space using dimensionality reduction techniques.
    
    Args:
        population_data (DataFrame): The population data with generations and genetic information.
        output_dir (str): Directory to save visualization outputs.
        output_prefix (str): Prefix for output filenames.
        method (str): Dimensionality reduction method, either 'pca' or 'tsne'.
        n_components (int): Number of components for the reduced space (2 or 3).
        create_animation (bool): Whether to create an animation of the population evolution.
        
    Returns:
        dict: Dictionary with paths to the generated visualizations.
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check if we have enough genetic information for the method
    gene_columns = [col for col in population_data.columns if col.startswith('gene_')]
    if len(gene_columns) < n_components:
        print(f"Warning: Not enough gene dimensions ({len(gene_columns)}) for {n_components} components.")
        return {}
    
    # Get unique generations for analysis
    generations = sorted(population_data['generation'].unique())
    
    # Extract genetic information for dimensionality reduction
    X = population_data[gene_columns].values
    
    # Apply dimensionality reduction
    if method.lower() == 'pca':
        reducer = PCA(n_components=n_components)
        X_reduced = reducer.fit_transform(X)
        method_name = "PCA"
        explained_variance = reducer.explained_variance_ratio_
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42)
        X_reduced = reducer.fit_transform(X)
        method_name = "t-SNE"
        explained_variance = None
    else:
        raise ValueError(f"Unsupported method: {method}. Use 'pca' or 'tsne'.")
    
    # Add reduced dimensions to the dataframe
    for i in range(n_components):
        component_name = f"{method.lower()}_component_{i+1}"
        population_data[component_name] = X_reduced[:, i]
    
    # Apply K-means clustering to identify population clusters
    n_clusters = min(5, len(population_data) // 10)  # Adjust number of clusters based on data size
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    population_data['cluster'] = kmeans.fit_predict(X_reduced)
    
    # Create visualization results dictionary
    result_paths = {}
    
    # Static visualization of the reduced space
    plt.figure(figsize=(12, 10))
    
    # Set up axes based on number of components
    if n_components == 2:
        ax = plt.subplot(111)
        scatter = ax.scatter(
            population_data[f"{method.lower()}_component_1"],
            population_data[f"{method.lower()}_component_2"],
            c=population_data['generation'],
            cmap='viridis',
            alpha=0.7,
            s=50
        )
        
        # Add cluster centers
        centers = kmeans.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.8, marker='X', label='Cluster Centers')
        
        plt.xlabel(f"{method_name} Component 1")
        plt.ylabel(f"{method_name} Component 2")
        
        if explained_variance is not None:
            plt.xlabel(f"{method_name} Component 1 ({explained_variance[0]:.2%} variance)")
            plt.ylabel(f"{method_name} Component 2 ({explained_variance[1]:.2%} variance)")
    else:
        ax = plt.subplot(111, projection='3d')
        scatter = ax.scatter(
            population_data[f"{method.lower()}_component_1"],
            population_data[f"{method.lower()}_component_2"],
            population_data[f"{method.lower()}_component_3"],
            c=population_data['generation'],
            cmap='viridis',
            alpha=0.7,
            s=50
        )
        
        # Add cluster centers
        centers = kmeans.cluster_centers_
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='red', s=200, alpha=0.8, marker='X', label='Cluster Centers')
        
        ax.set_xlabel(f"{method_name} Component 1")
        ax.set_ylabel(f"{method_name} Component 2")
        ax.set_zlabel(f"{method_name} Component 3")
        
        if explained_variance is not None:
            ax.set_xlabel(f"{method_name} Component 1 ({explained_variance[0]:.2%} variance)")
            ax.set_ylabel(f"{method_name} Component 2 ({explained_variance[1]:.2%} variance)")
            ax.set_zlabel(f"{method_name} Component 3 ({explained_variance[2]:.2%} variance)")
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Generation')
    
    plt.title(f"Population Migration in {method_name} Space with Clustering")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save static visualization
    static_path = output_path / f"{output_prefix}_{method.lower()}_space.png"
    plt.savefig(static_path)
    result_paths['reduced_space_plot'] = str(static_path)
    
    # Create animation if requested
    if create_animation:
        # Setup figure for animation
        fig = plt.figure(figsize=(12, 10))
        
        if n_components == 2:
            ax = plt.subplot(111)
            
            # Initialize scatter plot for animation
            scatter = ax.scatter([], [], c=[], cmap='viridis', alpha=0.7, s=50)
            
            # Add cluster centers
            centers_scatter = ax.scatter([], [], c='red', s=200, alpha=0.8, marker='X')
            
            # Set axis labels
            if explained_variance is not None:
                ax.set_xlabel(f"{method_name} Component 1 ({explained_variance[0]:.2%} variance)")
                ax.set_ylabel(f"{method_name} Component 2 ({explained_variance[1]:.2%} variance)")
            else:
                ax.set_xlabel(f"{method_name} Component 1")
                ax.set_ylabel(f"{method_name} Component 2")
                
            # Set plot limits based on all data
            margin = 0.1
            x_min, x_max = population_data[f"{method.lower()}_component_1"].min(), population_data[f"{method.lower()}_component_1"].max()
            y_min, y_max = population_data[f"{method.lower()}_component_2"].min(), population_data[f"{method.lower()}_component_2"].max()
            x_range = x_max - x_min
            y_range = y_max - y_min
            ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
            ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)
            
            # Text annotations for generation number
            generation_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12, 
                                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # Function to initialize animation
            def init():
                scatter.set_offsets(np.empty((0, 2)))
                centers_scatter.set_offsets(np.empty((0, 2)))
                generation_text.set_text('')
                return scatter, centers_scatter, generation_text
            
            # Function to update animation for each frame (generation)
            def update(frame):
                # Filter data for current generation
                gen_data = population_data[population_data['generation'] == generations[frame]]
                
                # Update scatter data
                scatter.set_offsets(gen_data[[f"{method.lower()}_component_1", f"{method.lower()}_component_2"]].values)
                
                # Update colors based on clusters within this generation
                if len(gen_data) > n_clusters:
                    kmeans_gen = KMeans(n_clusters=min(n_clusters, len(gen_data) // 2), random_state=42, n_init=10)
                    gen_clusters = kmeans_gen.fit_predict(gen_data[[f"{method.lower()}_component_1", f"{method.lower()}_component_2"]].values)
                    scatter.set_array(gen_clusters)
                    
                    # Update cluster centers
                    centers_gen = kmeans_gen.cluster_centers_
                    centers_scatter.set_offsets(centers_gen)
                else:
                    scatter.set_array(np.zeros(len(gen_data)))
                    centers_scatter.set_offsets(np.empty((0, 2)))
                
                # Update generation text
                generation_text.set_text(f'Generation: {generations[frame]}')
                
                return scatter, centers_scatter, generation_text
            
        else:  # 3D animation
            ax = plt.subplot(111, projection='3d')
            
            # Initialize scatter plots
            scatter = ax.scatter([], [], [], c=[], cmap='viridis', alpha=0.7, s=50)
            centers_scatter = ax.scatter([], [], [], c='red', s=200, alpha=0.8, marker='X')
            
            # Set axis labels
            if explained_variance is not None:
                ax.set_xlabel(f"{method_name} Component 1 ({explained_variance[0]:.2%} variance)")
                ax.set_ylabel(f"{method_name} Component 2 ({explained_variance[1]:.2%} variance)")
                ax.set_zlabel(f"{method_name} Component 3 ({explained_variance[2]:.2%} variance)")
            else:
                ax.set_xlabel(f"{method_name} Component 1")
                ax.set_ylabel(f"{method_name} Component 2")
                ax.set_zlabel(f"{method_name} Component 3")
                
            # Set plot limits
            margin = 0.1
            x_min, x_max = population_data[f"{method.lower()}_component_1"].min(), population_data[f"{method.lower()}_component_1"].max()
            y_min, y_max = population_data[f"{method.lower()}_component_2"].min(), population_data[f"{method.lower()}_component_2"].max()
            z_min, z_max = population_data[f"{method.lower()}_component_3"].min(), population_data[f"{method.lower()}_component_3"].max()
            x_range = x_max - x_min
            y_range = y_max - y_min
            z_range = z_max - z_min
            ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
            ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)
            ax.set_zlim(z_min - margin * z_range, z_max + margin * z_range)
            
            # Text annotations for generation number
            generation_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes, fontsize=12, 
                                      verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            # Function to initialize animation
            def init():
                scatter._offsets3d = (np.array([]), np.array([]), np.array([]))
                centers_scatter._offsets3d = (np.array([]), np.array([]), np.array([]))
                generation_text.set_text('')
                return scatter, centers_scatter, generation_text
            
            # Function to update animation for each frame (generation)
            def update(frame):
                # Filter data for current generation
                gen_data = population_data[population_data['generation'] == generations[frame]]
                
                # Update scatter data
                x = gen_data[f"{method.lower()}_component_1"].values
                y = gen_data[f"{method.lower()}_component_2"].values
                z = gen_data[f"{method.lower()}_component_3"].values
                scatter._offsets3d = (x, y, z)
                
                # Update colors based on clusters within this generation
                if len(gen_data) > n_clusters:
                    kmeans_gen = KMeans(n_clusters=min(n_clusters, len(gen_data) // 2), random_state=42, n_init=10)
                    gen_clusters = kmeans_gen.fit_predict(gen_data[[f"{method.lower()}_component_1", 
                                                                   f"{method.lower()}_component_2",
                                                                   f"{method.lower()}_component_3"]].values)
                    scatter.set_array(gen_clusters)
                    
                    # Update cluster centers
                    centers_gen = kmeans_gen.cluster_centers_
                    centers_scatter._offsets3d = (centers_gen[:, 0], centers_gen[:, 1], centers_gen[:, 2])
                else:
                    scatter.set_array(np.zeros(len(gen_data)))
                    centers_scatter._offsets3d = (np.array([]), np.array([]), np.array([]))
                
                # Update generation text
                generation_text.set_text(f'Generation: {generations[frame]}')
                
                return scatter, centers_scatter, generation_text
        
        # Create animation
        ani = FuncAnimation(fig, update, frames=len(generations), init_func=init, blit=True)
        
        # Title and grid
        plt.title(f"Population Migration in {method_name} Space Over Generations")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save animation
        ani_path = output_path / f"{output_prefix}_{method.lower()}_animation.mp4"
        ani.save(ani_path, writer='ffmpeg', fps=5, dpi=100)
        plt.close()
        
        result_paths['reduced_space_animation'] = str(ani_path)
    
    plt.close('all')
    return result_paths 