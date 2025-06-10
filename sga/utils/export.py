"""Utilities for exporting optimization data."""

from pathlib import Path
import csv
import json
from datetime import datetime
from typing import Dict, List, Any, Optional


def export_metrics_to_csv(
    metrics: Dict[str, List[Any]], 
    filename: Optional[str] = None, 
    directory: str = "results",
) -> str:
    """Export optimization metrics to a CSV file.
    
    Args:
        metrics: Dictionary of metric names to lists of values
        filename: Optional filename (defaults to timestamp-based name)
        directory: Directory to save the file in
        
    Returns:
        Path to the created CSV file
    """
    # Create directory if it doesn't exist
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    
    # Generate default filename if none provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimization_metrics_{timestamp}.csv"
    
    # Ensure .csv extension
    if not filename.endswith(".csv"):
        filename += ".csv"
    
    filepath = dir_path / filename
    
    # Get all metric names
    metric_names = list(metrics.keys())
    
    # Find the maximum length of any metric
    max_length = max([len(metrics[name]) for name in metric_names]) if metric_names else 0
    
    with filepath.open('w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['generation'] + metric_names)
        
        # Write data rows
        for i in range(max_length):
            row = [i]  # Generation number
            for name in metric_names:
                # Handle cases where metrics have different lengths
                if i < len(metrics[name]):
                    row.append(metrics[name][i])
                else:
                    row.append(None)
            writer.writerow(row)
    
    return str(filepath)


def export_population_history_to_csv(
    population_history: List[List[List[float]]], 
    filename: Optional[str] = None,
    directory: str = "results",
) -> str:
    """Export population history to a CSV file.
    
    Args:
        population_history: List of populations (each a list of individuals)
        filename: Optional filename (defaults to timestamp-based name)
        directory: Directory to save the file in
        
    Returns:
        Path to the created CSV file
    """
    # Create directory if it doesn't exist
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    
    # Generate default filename if none provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"population_history_{timestamp}.csv"
    
    # Ensure .csv extension
    if not filename.endswith(".csv"):
        filename += ".csv"
    
    filepath = dir_path / filename
    
    with filepath.open('w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header - generation, individual_id, gene_1, gene_2, ...
        max_chromosome_length = max(
            [len(ind) for pop in population_history for ind in pop]
        ) if population_history else 0
        
        header = ['generation', 'individual_id']
        header.extend([f'gene_{i+1}' for i in range(max_chromosome_length)])
        writer.writerow(header)
        
        # Write data
        for gen_idx, population in enumerate(population_history):
            for ind_idx, individual in enumerate(population):
                row = [gen_idx, ind_idx]
                row.extend(individual)
                writer.writerow(row)
    
    return str(filepath)


def export_run_metadata(
    config: Dict[str, Any],
    results: Dict[str, Any],
    filename: Optional[str] = None,
    directory: str = "results",
) -> str:
    """Export run metadata (configuration and results) to a JSON file.
    
    Args:
        config: Configuration parameters
        results: Results of the optimization
        filename: Optional filename (defaults to timestamp-based name)
        directory: Directory to save the file in
        
    Returns:
        Path to the created JSON file
    """
    # Create directory if it doesn't exist
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    
    # Generate default filename if none provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"run_metadata_{timestamp}.json"
    
    # Ensure .json extension
    if not filename.endswith(".json"):
        filename += ".json"
    
    filepath = dir_path / filename
    
    # Combine config and results
    data = {
        "configuration": config,
        "results": results,
        "timestamp": datetime.now().isoformat()
    }
    
    # Handle non-serializable types
    def json_serializer(obj):
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)
    
    with filepath.open('w') as jsonfile:
        json.dump(data, jsonfile, indent=2, default=json_serializer)
    
    return str(filepath) 