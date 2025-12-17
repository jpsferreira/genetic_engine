#!/usr/bin/env python
"""
Create an animated GIF demonstrating the genetic algorithm optimization.

The GIF shows a grid with:
- Top left: Terminal-style output mimicking the live monitor with progress bar and metrics
- Top right: Curve fitting progress (true vs fitted polynomial)
- Bottom: Population evolution in parameter space with fitness coloring
"""

import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from genetic_opt.sga.optimizer import SimpleGeneticAlgorithm


class TerminalDisplay:
    """Create terminal-like display with progress bar and metrics."""
    
    def __init__(self):
        self.lines = []
        self.generation = 0
        self.n_generations = 0
        
    def update(self, generation, n_generations, best_fitness, avg_fitness, std_fitness):
        """Update the terminal display."""
        self.generation = generation
        self.n_generations = n_generations
        
        # Build display text
        lines = []
        lines.append("═" * 58)
        lines.append(f" Genetic Optimization - Gen {generation}/{n_generations}")
        lines.append("═" * 58)
        lines.append("")
        
        # Progress bar
        progress = generation / n_generations if n_generations > 0 else 0
        bar_width = 40
        filled = int(progress * bar_width)
        bar = "█" * filled + "░" * (bar_width - filled)
        lines.append(f"Progress: {bar} {progress*100:.1f}%")
        lines.append("")
        
        # Metrics
        lines.append("FITNESS METRICS")
        lines.append(f"  Current Best:  {best_fitness:>12.6f}")
        lines.append(f"  Population Avg:{avg_fitness:>12.6f}")
        lines.append(f"  Population Std:{std_fitness:>12.6f}")
        lines.append("")
        
        # Mini fitness chart (sparkline-like)
        lines.append("Best Fitness Trend:")
        lines.append("  " + self._create_sparkline(best_fitness))
        lines.append("")
        lines.append("─" * 58)
        
        self.lines = lines
    
    def _create_sparkline(self, current_value):
        """Create a simple text-based sparkline."""
        # Simple trend indicator
        if self.generation < 5:
            return "▁▂▃▄▅▆▇█" * 5 + f" {current_value:.4f}"
        else:
            # Show improving trend
            blocks = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']
            trend = [blocks[min(i, 7)] for i in range(min(self.generation, 40))]
            return ''.join(trend) + f" {current_value:.4f}"
    
    def get_text(self):
        """Get the formatted text."""
        return '\n'.join(self.lines)


class GeneticAlgorithmVisualizer:
    """Create animated visualization of genetic algorithm."""
    
    def __init__(self, figsize=(18, 10)):
        self.figsize = figsize
        self.fig = None
        self.terminal_display = TerminalDisplay()
        
        # Data storage
        self.generation_data = []
        self.x_data = None
        self.y_true = None
        self.y_noisy = None
        self.true_coefficients = None
        
    def setup_problem(self):
        """Setup the polynomial fitting problem."""
        # Set random seeds
        random.seed(42)
        np.random.seed(42)
        
        # Define the true polynomial
        self.true_coefficients = [2, -5, 3, -1]
        
        # Generate data points
        self.x_data = np.linspace(-2, 2, 50)
        self.y_true = np.polyval(self.true_coefficients, self.x_data)
        self.y_noisy = self.y_true + np.random.normal(0, 0.5, size=len(self.y_true))
        
        # Define fitness function
        def polynomial_fitness(coefficients):
            y_pred = np.polyval(coefficients, self.x_data)
            mse = np.mean((y_pred - self.y_noisy) ** 2)
            return -mse
        
        return polynomial_fitness
    
    def run_optimization(self, n_generations=50):
        """Run optimization and capture data at each generation."""
        fitness_function = self.setup_problem()
        
        # Create optimizer
        optimizer = SimpleGeneticAlgorithm(
            fitness_function=fitness_function,
            population_size=100,
            mutation_rate=0.1,
            elite_size=10,
            tournament_size=3,
            verbose=False,
        )
        
        # Callback to capture generation data
        def generation_callback(generation, population, fitness_scores):
            best_idx = fitness_scores.index(max(fitness_scores))
            best_fitness = fitness_scores[best_idx]
            avg_fitness = np.mean(fitness_scores)
            std_fitness = np.std(fitness_scores)
            
            # Update terminal display
            self.terminal_display.update(
                generation, n_generations, 
                best_fitness, avg_fitness, std_fitness
            )
            
            # Store data including terminal text snapshot
            self.generation_data.append({
                'generation': generation,
                'population': [ind.copy() for ind in population],
                'fitness_scores': fitness_scores.copy(),
                'best_solution': population[best_idx].copy(),
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'std_fitness': std_fitness,
                'terminal_text': self.terminal_display.get_text(),  # Store snapshot
            })
        
        # Add callback to optimizer
        original_create = optimizer._create_next_generation
        
        def wrapped_create(population, fitness_scores, bounds):
            generation = len(self.generation_data)
            if generation < n_generations:
                generation_callback(generation, population, fitness_scores)
            return original_create(population, fitness_scores, bounds)
        
        optimizer._create_next_generation = wrapped_create
        
        # Run optimization
        bounds = [(-10, 10)] * 4
        best_solution, best_fitness = optimizer.optimize(
            n_generations=n_generations,
            chromosome_length=4,
            bounds=bounds,
        )
        
        return best_solution, best_fitness
    
    def create_animation(self, output_file='demo.gif', fps=2):
        """Create the animated GIF."""
        # Setup figure with grid and black background
        self.fig = plt.figure(figsize=self.figsize, facecolor='black')
        gs = GridSpec(2, 2, figure=self.fig, hspace=0.35, wspace=0.3,
                     left=0.05, right=0.95, top=0.95, bottom=0.05)
        
        # Create subplots
        ax_terminal = self.fig.add_subplot(gs[0, 0])
        ax_curve = self.fig.add_subplot(gs[0, 1])
        ax_population = self.fig.add_subplot(gs[1, :])
        
        # Style terminal
        ax_terminal.set_xlim(0, 1)
        ax_terminal.set_ylim(0, 1)
        ax_terminal.axis('off')
        ax_terminal.set_facecolor('#000000')
        
        # Add terminal border
        # rect = Rectangle((0.02, 0.02), 0.96, 0.96, fill=False, 
        #                 edgecolor='#00ff00', linewidth=2, alpha=0.7)
        # ax_terminal.add_patch(rect)
        
        # Terminal text
        terminal_text = ax_terminal.text(
            0.06, 0.94, '', 
            fontfamily='monospace', 
            fontsize=16,
            color='#00ff00',
            verticalalignment='top',
            transform=ax_terminal.transAxes
        )
        
        # Setup curve plot
        ax_curve.set_xlim(self.x_data.min() - 0.2, self.x_data.max() + 0.2)
        ax_curve.set_ylim(self.y_noisy.min() - 2, self.y_noisy.max() + 2)
        ax_curve.set_xlabel('x', fontsize=18, fontweight='bold', color='white')
        ax_curve.set_ylabel('y', fontsize=18, fontweight='bold', color='white')
        ax_curve.set_title('Polynomial Curve Fitting Evolution', 
                          fontsize=20, fontweight='bold', pad=10, color='white')
        ax_curve.grid(True, alpha=0.3, linestyle='--', color='gray')
        ax_curve.set_facecolor('#1a1a1a')
        ax_curve.tick_params(colors='white', labelsize=16)
        for spine in ax_curve.spines.values():
            spine.set_color('white')
        
        # Plot data points and true curve
        ax_curve.scatter(self.x_data, self.y_noisy, alpha=0.6, s=40, 
                        color='cyan', label='Noisy Data', zorder=3)
        ax_curve.plot(self.x_data, self.y_true, 'g--', linewidth=2.5, 
                     alpha=0.8, label='True Polynomial', zorder=2)
        
        # Fitted curve (will be updated)
        fitted_line, = ax_curve.plot([], [], 'r-', linewidth=3, 
                                     label='GA Fitted', zorder=4)
        
        # Text for equations (will be updated)
        equation_text = ax_curve.text(
            0.98, 0.02, '',
            transform=ax_curve.transAxes,
            fontsize=14,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.8, edgecolor='white'),
            color='white',
            fontfamily='monospace'
        )
        
        ax_curve.legend(loc='upper left', fontsize=16, framealpha=0.9, 
                       facecolor='black', edgecolor='white', labelcolor='white')
        
        # Setup population plot (showing first two coefficients)
        ax_population.set_xlim(-10, 10)
        ax_population.set_ylim(-10, 10)
        ax_population.set_xlabel('Coefficient a (x³)', fontsize=18, fontweight='bold', color='white')
        ax_population.set_ylabel('Coefficient b (x²)', fontsize=18, fontweight='bold', color='white')
        ax_population.set_title('Population Evolution in Parameter Space', 
                               fontsize=20, fontweight='bold', pad=10, color='white')
        ax_population.grid(True, alpha=0.3, linestyle='--', color='gray')
        ax_population.set_facecolor('#1a1a1a')
        ax_population.tick_params(colors='white', labelsize=16)
        for spine in ax_population.spines.values():
            spine.set_color('white')
        
        # Mark true solution
        ax_population.scatter(
            [self.true_coefficients[0]], 
            [self.true_coefficients[1]], 
            marker='*', s=500, color='gold', 
            edgecolors='white', linewidths=2,
            label='True Solution', zorder=1000
        )
        
        # Population scatter (will be updated)
        population_scatter = ax_population.scatter([], [], alpha=0.6, s=60, 
                                                  c='blue', cmap='plasma')
        best_scatter = ax_population.scatter([], [], marker='X', s=250, 
                                            color='red', edgecolors='white',
                                            linewidths=2, label='Best Individual',
                                            zorder=999)
        ax_population.legend(loc='upper right', fontsize=16, framealpha=0.9,
                           facecolor='black', edgecolor='white', labelcolor='white')
        
        # Add colorbar for fitness
        cbar = plt.colorbar(population_scatter, ax=ax_population, pad=0.02)
        cbar.set_label('Fitness', fontsize=16, fontweight='bold', color='white')
        cbar.ax.tick_params(labelsize=14, colors='white')
        cbar.outline.set_edgecolor('white')
        
        def init():
            """Initialize animation."""
            terminal_text.set_text('')
            fitted_line.set_data([], [])
            population_scatter.set_offsets(np.empty((0, 2)))
            best_scatter.set_offsets(np.empty((0, 2)))
            return terminal_text, fitted_line, population_scatter, best_scatter, equation_text
        
        def update(frame):
            """Update animation frame."""
            if frame >= len(self.generation_data):
                return terminal_text, fitted_line, population_scatter, best_scatter, equation_text
            
            gen_data = self.generation_data[frame]
            
            # Update terminal with stored text for this frame
            terminal_text.set_text(gen_data['terminal_text'])
            
            # Update fitted curve
            y_fitted = np.polyval(gen_data['best_solution'], self.x_data)
            fitted_line.set_data(self.x_data, y_fitted)
            
            # Update equation text
            coef = gen_data['best_solution']
            true = self.true_coefficients
            eq_text = (
                f"True:   y = {true[0]:.1f}x³ + {true[1]:.1f}x² + {true[2]:.1f}x + {true[3]:.1f}\n"
                f"Fitted: y = {coef[0]:.2f}x³ + {coef[1]:.2f}x² + {coef[2]:.2f}x + {coef[3]:.2f}"
            )
            equation_text.set_text(eq_text)
            
            # Update population scatter (first two coefficients)
            population_array = np.array(gen_data['population'])
            population_scatter.set_offsets(population_array[:, :2])
            
            # Color by fitness (normalized)
            fitness_array = np.array(gen_data['fitness_scores'])
            fitness_min = fitness_array.min()
            fitness_max = fitness_array.max()
            if fitness_max > fitness_min:
                fitness_norm = (fitness_array - fitness_min) / (fitness_max - fitness_min)
            else:
                fitness_norm = np.ones_like(fitness_array) * 0.5
            
            population_scatter.set_array(fitness_norm)
            
            # Update best solution
            best_scatter.set_offsets([gen_data['best_solution'][:2]])
            
            return terminal_text, fitted_line, population_scatter, best_scatter, equation_text
        
        # Create animation
        n_frames = len(self.generation_data)
        anim = animation.FuncAnimation(
            self.fig, update, init_func=init,
            frames=n_frames, interval=1000/fps,
            blit=True, repeat=True
        )
        
        # Save as GIF
        print(f"Saving animation to {output_file}...")
        print(f"Creating {n_frames} frames at {fps} fps...")
        anim.save(output_file, writer='pillow', fps=fps, dpi=100)
        print(f"✓ Animation saved to {output_file}")
        
        plt.close(self.fig)
        return anim


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create genetic algorithm demo GIF')
    parser.add_argument('--output', '-o', default='genetic_algorithm_demo.gif',
                       help='Output GIF file (default: genetic_algorithm_demo.gif)')
    parser.add_argument('--generations', '-g', type=int, default=50,
                       help='Number of generations (default: 50)')
    parser.add_argument('--fps', type=int, default=2,
                       help='Frames per second (default: 2)')
    parser.add_argument('--figsize', nargs=2, type=float, default=[18, 10],
                       help='Figure size in inches (default: 18 10)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Genetic Algorithm Demo GIF Creator")
    print("=" * 70)
    print()
    print(f"Output file: {args.output}")
    print(f"Generations: {args.generations}")
    print(f"FPS: {args.fps}")
    print(f"Figure size: {args.figsize[0]}x{args.figsize[1]} inches")
    print()
    
    # Create visualizer
    visualizer = GeneticAlgorithmVisualizer(figsize=tuple(args.figsize))
    
    # Run optimization
    print("Running optimization...")
    start_time = time.time()
    best_solution, best_fitness = visualizer.run_optimization(n_generations=args.generations)
    elapsed = time.time() - start_time
    
    print(f"✓ Optimization complete in {elapsed:.2f} seconds")
    print(f"  Captured {len(visualizer.generation_data)} generations")
    print(f"  Best fitness: {best_fitness:.6f}")
    print()
    
    # Create animation
    visualizer.create_animation(output_file=args.output, fps=args.fps)
    
    print()
    print("=" * 70)
    print("✓ Demo GIF created successfully!")
    print(f"  View it at: {Path(args.output).absolute()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
