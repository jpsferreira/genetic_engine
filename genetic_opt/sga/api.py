"""Convenience API for quick optimization."""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from genetic_opt.sga.optimizer import SimpleGeneticAlgorithm


@dataclass
class OptimizationResult:
    """Container for optimization results.

    Attributes:
        x: Best solution found.
        fun: Fitness value of the best solution.
        n_generations: Number of generations actually run.
        converged: Whether early stopping was triggered.
        metrics: Per-generation metrics dictionary.
        optimizer: The underlying optimizer instance (for further analysis).
    """

    x: List[float]
    fun: float
    n_generations: int
    converged: bool
    metrics: Dict[str, List]
    optimizer: SimpleGeneticAlgorithm = field(repr=False)


def minimize(
    func: Callable[[List[float]], float],
    bounds: List[Tuple[float, float]],
    n_generations: int = 200,
    population_size: Optional[int] = None,
    **kwargs,
) -> OptimizationResult:
    """Minimise *func* over the given *bounds*.

    This is a one-call entry point that creates a
    :class:`SimpleGeneticAlgorithm`, runs it, and returns a structured
    result.

    Args:
        func: Objective function to minimise.  Receives a list of floats
            and returns a scalar.
        bounds: ``[(lo, hi), ...]`` for each dimension.
        n_generations: Maximum number of generations.
        population_size: Population size.  Defaults to
            ``max(50, 10 * len(bounds))`` if not provided.
        **kwargs: Forwarded to :class:`SimpleGeneticAlgorithm`.

    Returns:
        An :class:`OptimizationResult` with the best solution.
    """
    return _run(func, bounds, "minimize", n_generations, population_size, **kwargs)


def maximize(
    func: Callable[[List[float]], float],
    bounds: List[Tuple[float, float]],
    n_generations: int = 200,
    population_size: Optional[int] = None,
    **kwargs,
) -> OptimizationResult:
    """Maximise *func* over the given *bounds*.

    See :func:`minimize` for parameter descriptions.
    """
    return _run(func, bounds, "maximize", n_generations, population_size, **kwargs)


def _run(func, bounds, sense, n_generations, population_size, **kwargs):
    ndim = len(bounds)
    if population_size is None:
        population_size = max(50, 10 * ndim)

    # Sensible defaults that scale with problem size
    kwargs.setdefault("elite_size", max(2, population_size // 10))
    kwargs.setdefault("export_data", False)

    opt = SimpleGeneticAlgorithm(
        fitness_function=func,
        population_size=population_size,
        sense=sense,
        **kwargs,
    )

    best_x, best_f = opt.optimize(
        n_generations=n_generations,
        chromosome_length=ndim,
        bounds=bounds,
    )

    return OptimizationResult(
        x=best_x,
        fun=best_f,
        n_generations=opt._n_generations_run,
        converged=opt._stopped_early,
        metrics=opt.metrics,
        optimizer=opt,
    )
