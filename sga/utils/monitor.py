"""Terminal-based progress monitor for genetic optimization algorithms."""

import curses
import time
import statistics
from typing import Dict, List, Any, Optional


class OptimizationMonitor:
    """Terminal-based real-time monitor for genetic optimization."""
    
    def __init__(self):
        """Initialize the monitor."""
        self.stdscr = None
        self.started = False
        self.paused = False
        self.height = 0
        self.width = 0
        self.generation = 0
        self.n_generations = 0
        self.metrics: Dict[str, List[Any]] = {}
        self.start_time = time.time()
    
    def start(self, n_generations: int, metrics: Dict[str, List[Any]]) -> None:
        """Start the monitor.
        
        Args:
            n_generations: Total number of generations
            metrics: Dictionary of metrics to track
        """
        self.started = True
        self.n_generations = n_generations
        self.metrics = metrics
        self.start_time = time.time()
        
        # Initialize curses
        self.stdscr = curses.initscr()
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1)  # Good values
        curses.init_pair(2, curses.COLOR_YELLOW, -1)  # Warning values
        curses.init_pair(3, curses.COLOR_RED, -1)  # Bad values
        curses.init_pair(4, curses.COLOR_CYAN, -1)  # Headers
        curses.init_pair(5, curses.COLOR_MAGENTA, -1)  # Highlights
        curses.init_pair(6, curses.COLOR_WHITE, -1)  # Normal text
        
        curses.noecho()
        curses.cbreak()
        self.stdscr.keypad(True)
        curses.curs_set(0)  # Hide cursor
        self.stdscr.timeout(100)  # Non-blocking input check
        
        # Get terminal dimensions
        self.height, self.width = self.stdscr.getmaxyx()
    
    def update(self, generation: int) -> None:
        """Update the display with the latest metrics.
        
        Args:
            generation: Current generation number
        """
        if not self.started or self.paused:
            return
        
        self.generation = generation
        
        try:
            # Check for key presses
            key = self.stdscr.getch()
            if key == ord('q'):
                self.stop()
                return
            elif key == ord('p'):
                self.paused = not self.paused
            
            # Refresh terminal dimensions in case of resize
            self.height, self.width = self.stdscr.getmaxyx()
            
            # Check if terminal is too small
            if self.height < 10 or self.width < 50:
                self._draw_too_small_screen()
                self.stdscr.refresh()
                return
            
            # Clear screen
            self.stdscr.clear()
            
            # Draw header
            self._draw_header()
            
            # Draw progress bar
            self._draw_progress_bar()
            
            # Draw metrics
            self._draw_metrics()
            
            # Draw status bar
            self._draw_status_bar()
            
            # Refresh screen
            self.stdscr.refresh()
            
        except Exception as e:
            self.stop()
            raise e
    
    def stop(self) -> None:
        """Stop the monitor and clean up curses."""
        if not self.started:
            return
        
        self.started = False
        
        # Clean up curses
        curses.nocbreak()
        self.stdscr.keypad(False)
        curses.echo()
        curses.endwin()
    
    def _draw_too_small_screen(self) -> None:
        """Draw a message when the terminal is too small."""
        self.stdscr.clear()
        msg = "Terminal too small"
        if self.height > 0 and self.width > len(msg):
            y = max(0, self.height // 2 - 1)
            x = max(0, (self.width - len(msg)) // 2)
            self.stdscr.addstr(y, x, msg)
    
    def _draw_header(self) -> None:
        """Draw the header with basic information."""
        header = f" Genetic Optimization Monitor - Generation {self.generation}/{self.n_generations} "
        
        # Truncate header if needed
        if len(header) > self.width - 20:
            header = f" Generation {self.generation}/{self.n_generations} "
            
        elapsed = time.time() - self.start_time
        elapsed_str = f" Elapsed: {int(elapsed // 60):02d}:{int(elapsed % 60):02d} "
        
        # Center the header
        self.stdscr.attron(curses.color_pair(4) | curses.A_BOLD)
        x = max(0, min((self.width - len(header)) // 2, self.width - len(header)))
        self.stdscr.addstr(0, x, header[:self.width - x - 1])
        self.stdscr.attroff(curses.color_pair(4) | curses.A_BOLD)
        
        # Add elapsed time on the right
        if self.width > len(elapsed_str):
            x = max(0, min(self.width - len(elapsed_str), self.width - 1))
            self.stdscr.addstr(0, x, elapsed_str[:self.width - x - 1])
    
    def _draw_progress_bar(self) -> None:
        """Draw a progress bar showing generation progress."""
        # Check if we have enough space
        if self.height < 3 or self.width < 25:
            return
            
        bar_width = max(5, self.width - 20)
        progress = self.generation / self.n_generations if self.n_generations > 0 else 0
        filled_width = min(int(progress * bar_width), bar_width)
        
        self.stdscr.addstr(2, 2, "Progress: "[:self.width - 3])
        self.stdscr.attron(curses.color_pair(1))
        filled_str = "█" * filled_width
        self.stdscr.addstr(2, min(12, self.width - len(filled_str) - 1), filled_str)
        self.stdscr.attroff(curses.color_pair(1))
        
        if bar_width > filled_width and self.width > 12 + filled_width:
            unfilled_str = "░" * min(bar_width - filled_width, self.width - 12 - filled_width - 1)
            self.stdscr.addstr(2, 12 + filled_width, unfilled_str)
        
        percent = f"{progress * 100:.1f}%"
        if self.width > 13 + bar_width:
            self.stdscr.addstr(2, min(13 + bar_width, self.width - len(percent) - 1), percent)
    
    def _draw_metrics(self) -> None:
        """Draw the metrics in a tabular format."""
        if not self.metrics or self.height < 8:
            return
            
        # Available space checks
        if self.height < 12:
            # Limited space - only show essential metrics
            self._draw_basic_metrics()
            return
        
        # Draw fitness metrics
        if self.height >= 5:
            fitness_y = 4
            self._safe_addstr(fitness_y, 2, "FITNESS METRICS", curses.color_pair(5) | curses.A_BOLD)
            
            if "best_fitness" in self.metrics and self.metrics["best_fitness"]:
                best_fitness = self.metrics["best_fitness"][-1]
                avg_fitness = (
                    self.metrics["avg_fitness"][-1] 
                    if "avg_fitness" in self.metrics and self.metrics["avg_fitness"] 
                    else 0
                )
                std_fitness = (
                    self.metrics["std_fitness"][-1] 
                    if "std_fitness" in self.metrics and self.metrics["std_fitness"] 
                    else 0
                )
                
                fitness_color = self._get_fitness_color(best_fitness)
                
                if fitness_y + 1 < self.height:
                    self._safe_addstr(fitness_y + 1, 4, "Current Best: ", curses.A_BOLD)
                    self._safe_addstr(fitness_y + 1, 17, f"{best_fitness:.6f}", fitness_color)
                
                if fitness_y + 2 < self.height:
                    self._safe_addstr(fitness_y + 2, 4, "Population Avg: ", curses.A_BOLD)
                    self._safe_addstr(fitness_y + 2, 19, f"{avg_fitness:.6f}")
                
                if fitness_y + 3 < self.height:
                    self._safe_addstr(fitness_y + 3, 4, "Population Std: ", curses.A_BOLD)
                    self._safe_addstr(fitness_y + 3, 19, f"{std_fitness:.6f}")
            
            # Draw mini chart if we have space
            if self.height >= 14:
                self._draw_mini_chart(9, 4, "Best Fitness History", "best_fitness", min(40, self.width - 10))
        
        # Draw performance metrics if we have space
        if self.height >= 16:
            perf_y = 14
            self._safe_addstr(perf_y, 2, "PERFORMANCE METRICS", curses.color_pair(5) | curses.A_BOLD)
            
            # Generation time
            if "generation_time" in self.metrics and self.metrics["generation_time"]:
                gen_time = self.metrics["generation_time"][-1]
                avg_gen_time = (
                    statistics.mean(self.metrics["generation_time"]) 
                    if len(self.metrics["generation_time"]) > 0 
                    else 0
                )
                
                if perf_y + 1 < self.height:
                    self._safe_addstr(perf_y + 1, 4, "Last Gen Time: ", curses.A_BOLD)
                    self._safe_addstr(perf_y + 1, 18, f"{gen_time:.4f} s")
                
                if perf_y + 2 < self.height:
                    self._safe_addstr(perf_y + 2, 4, "Avg Gen Time: ", curses.A_BOLD)
                    self._safe_addstr(perf_y + 2, 18, f"{avg_gen_time:.4f} s")
                
                # Draw time chart if we have space
                if self.height >= 22 and self.width >= 50:
                    self._draw_mini_chart(18, 4, "Generation Time", "generation_time", min(40, self.width - 10))
            
            # Memory usage if we have space
            if "memory_usage_mb" in self.metrics and self.metrics["memory_usage_mb"] and self.width >= 60:
                memory = self.metrics["memory_usage_mb"][-1]
                
                if perf_y + 1 < self.height:
                    self._safe_addstr(perf_y + 1, min(40, self.width - 20), "Memory Usage: ", curses.A_BOLD)
                    self._safe_addstr(perf_y + 1, min(53, self.width - 10), f"{memory:.2f} MB")
                
                # Draw memory chart if we have space
                if self.height >= 22 and self.width >= 85:
                    self._draw_mini_chart(18, min(40, self.width - 45), "Memory Usage", "memory_usage_mb", min(40, self.width - 50))
    
    def _draw_basic_metrics(self) -> None:
        """Draw basic metrics when space is limited."""
        if "best_fitness" in self.metrics and self.metrics["best_fitness"]:
            best_fitness = self.metrics["best_fitness"][-1]
            fitness_color = self._get_fitness_color(best_fitness)
            
            self._safe_addstr(4, 2, "Best Fitness: ", curses.A_BOLD)
            self._safe_addstr(4, 15, f"{best_fitness:.6f}", fitness_color)
            
            if "memory_usage_mb" in self.metrics and self.metrics["memory_usage_mb"]:
                memory = self.metrics["memory_usage_mb"][-1]
                self._safe_addstr(5, 2, "Memory: ", curses.A_BOLD)
                self._safe_addstr(5, 10, f"{memory:.2f} MB")
    
    def _draw_mini_chart(self, y: int, x: int, title: str, metric_key: str, width: int) -> None:
        """Draw a small chart for a given metric.
        
        Args:
            y: Y position
            x: X position
            title: Chart title
            metric_key: Key of the metric to display
            width: Width of the chart
        """
        if (
            metric_key not in self.metrics 
            or not self.metrics[metric_key] 
            or y >= self.height - 1
            or x >= self.width - 1
            or width < 5
        ):
            return
        
        # Draw title if we have space
        if y < self.height:
            self._safe_addstr(y, x, title, curses.A_BOLD)
        
        # Get data
        data = self.metrics[metric_key]
        if not data:
            return
        
        # Determine how many data points to show
        width = min(width, self.width - x - 1)
        points_to_show = min(len(data), width)
        if points_to_show < 2:
            return
        
        # Use the most recent data
        data = data[-points_to_show:]
        
        # Normalize data for display
        min_val = min(data)
        max_val = max(data)
        
        # Avoid division by zero
        if max_val == min_val:
            normalized = [0.5] * len(data)
        else:
            normalized = [(val - min_val) / (max_val - min_val) for val in data]
        
        # Draw chart
        chart_height = min(3, self.height - y - 2)
        if chart_height < 1:
            return
            
        for i, val in enumerate(normalized):
            if x + i >= self.width:
                break
                
            bar_height = int(val * chart_height)
            for h in range(chart_height):
                y_pos = y + 1 + chart_height - h
                x_pos = x + i
                
                if y_pos >= self.height:
                    continue
                    
                if bar_height >= h:
                    self._safe_addch(y_pos, x_pos, "█", curses.color_pair(1))
        
        # Draw min/max values if we have space
        max_text = f"Max: {max_val:.4f}"
        min_text = f"Min: {min_val:.4f}"
        
        if x + width + 2 < self.width - len(max_text) and y + 1 < self.height:
            self._safe_addstr(y + 1, x + width + 2, max_text)
            
        if x + width + 2 < self.width - len(min_text) and y + 1 + chart_height < self.height:
            self._safe_addstr(y + 1 + chart_height, x + width + 2, min_text)
    
    def _draw_status_bar(self) -> None:
        """Draw the status bar at the bottom of the screen."""
        if self.height < 2:
            return
            
        status = " [q] Quit | [p] Pause/Resume "
        
        # Draw status bar with inverted colors
        self.stdscr.attron(curses.A_REVERSE)
        self._safe_addstr(self.height - 1, 0, " " * (self.width - 1))
        self._safe_addstr(self.height - 1, 2, status[:self.width - 3])
        self.stdscr.attroff(curses.A_REVERSE)
    
    def _get_fitness_color(self, fitness: float) -> int:
        """Get the appropriate color for a fitness value.
        
        Args:
            fitness: Fitness value
            
        Returns:
            Curses color pair
        """
        # For negative fitness (minimizing problems like MSE)
        if fitness < 0:
            if fitness > -0.5:
                return curses.color_pair(1)  # Good (close to 0)
            elif fitness > -5.0:
                return curses.color_pair(2)  # Okay
            else:
                return curses.color_pair(3)  # Bad
        # For positive fitness (maximizing problems)
        else:
            if fitness > 0.8:
                return curses.color_pair(1)  # Good 
            elif fitness > 0.4:
                return curses.color_pair(2)  # Okay
            else:
                return curses.color_pair(3)  # Bad
        
        return curses.color_pair(6)  # Default color
    
    def _safe_addstr(self, y: int, x: int, text: str, attr=0) -> None:
        """Safely add a string to the screen, checking bounds.
        
        Args:
            y: Y position
            x: X position
            text: Text to add
            attr: Attributes to use
        """
        if y < 0 or y >= self.height or x < 0 or x >= self.width:
            return
            
        # Truncate text if it would go beyond the screen
        max_len = self.width - x
        if max_len <= 0:
            return
            
        text = text[:max_len]
        
        # Add the text with attributes
        if attr:
            self.stdscr.attron(attr)
            
        try:
            self.stdscr.addstr(y, x, text)
        except curses.error:
            # This can happen at the bottom-right corner of the screen
            pass
            
        if attr:
            self.stdscr.attroff(attr)
    
    def _safe_addch(self, y: int, x: int, ch: str, attr=0) -> None:
        """Safely add a character to the screen, checking bounds.
        
        Args:
            y: Y position
            x: X position
            ch: Character to add
            attr: Attributes to use
        """
        if y < 0 or y >= self.height or x < 0 or x >= self.width:
            return
            
        # Add the character with attributes
        if attr:
            self.stdscr.attron(attr)
            
        try:
            self.stdscr.addch(y, x, ch)
        except curses.error:
            # This can happen at the bottom-right corner of the screen
            pass
            
        if attr:
            self.stdscr.attroff(attr) 