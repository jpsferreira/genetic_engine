"""Genetic algorithm operators: crossover and selection strategies."""

import random
import math
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Crossover operators
# ---------------------------------------------------------------------------


def single_point_crossover(
    parent1: List[float], parent2: List[float], **_kwargs
) -> List[float]:
    """Single-point crossover.

    Picks a random split point and combines the first part of parent1
    with the second part of parent2.
    """
    if len(parent1) <= 1:
        return parent1.copy()
    point = random.randint(1, len(parent1) - 1)
    return parent1[:point] + parent2[point:]


def uniform_crossover(
    parent1: List[float], parent2: List[float], swap_prob: float = 0.5, **_kwargs
) -> List[float]:
    """Uniform crossover.

    Each gene is independently taken from parent1 or parent2 with
    probability *swap_prob* of coming from parent2.
    """
    return [
        p2 if random.random() < swap_prob else p1
        for p1, p2 in zip(parent1, parent2)
    ]


def blx_alpha_crossover(
    parent1: List[float],
    parent2: List[float],
    bounds: List[Tuple[float, float]],
    alpha: float = 0.5,
    **_kwargs,
) -> List[float]:
    """BLX-alpha (blend) crossover.

    For each gene the child value is sampled uniformly from
    ``[min(p1,p2) - alpha*d, max(p1,p2) + alpha*d]`` where
    ``d = |p1 - p2|``, clamped to the gene bounds.
    """
    child = []
    for i, (g1, g2) in enumerate(zip(parent1, parent2)):
        lo, hi = min(g1, g2), max(g1, g2)
        d = hi - lo
        low_bound = lo - alpha * d
        high_bound = hi + alpha * d
        # Clamp to gene bounds
        b_lo, b_hi = bounds[i % len(bounds)]
        low_bound = max(low_bound, b_lo)
        high_bound = min(high_bound, b_hi)
        child.append(random.uniform(low_bound, high_bound))
    return child


def sbx_crossover(
    parent1: List[float],
    parent2: List[float],
    bounds: List[Tuple[float, float]],
    eta: float = 2.0,
    **_kwargs,
) -> List[float]:
    """Simulated Binary Crossover (SBX).

    Produces one child using the SBX operator with distribution index *eta*.
    Higher eta means children closer to parents; lower eta means more spread.
    """
    child = []
    for i, (g1, g2) in enumerate(zip(parent1, parent2)):
        if abs(g1 - g2) < 1e-14:
            child.append(g1)
            continue
        u = random.random()
        if u <= 0.5:
            beta = (2.0 * u) ** (1.0 / (eta + 1.0))
        else:
            beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
        val = 0.5 * ((1 + beta) * g1 + (1 - beta) * g2)
        # Clamp to bounds
        b_lo, b_hi = bounds[i % len(bounds)]
        child.append(max(b_lo, min(b_hi, val)))
    return child


# Registry for easy lookup by name
CROSSOVER_OPERATORS = {
    "single_point": single_point_crossover,
    "uniform": uniform_crossover,
    "blx_alpha": blx_alpha_crossover,
    "sbx": sbx_crossover,
}


# ---------------------------------------------------------------------------
# Selection operators
# ---------------------------------------------------------------------------


def tournament_selection(
    population: List[List[float]],
    fitness_scores: List[float],
    maximize: bool = True,
    tournament_size: int = 3,
    **_kwargs,
) -> List[float]:
    """Tournament selection.

    Randomly picks *tournament_size* individuals and returns the best.
    """
    indices = random.sample(range(len(population)), tournament_size)
    best_func = max if maximize else min
    winner = best_func(indices, key=lambda i: fitness_scores[i])
    return population[winner]


def roulette_wheel_selection(
    population: List[List[float]],
    fitness_scores: List[float],
    maximize: bool = True,
    **_kwargs,
) -> List[float]:
    """Roulette-wheel (fitness-proportionate) selection.

    Probability of selection is proportional to fitness.  When minimizing,
    fitness values are inverted so that lower is better.
    """
    scores = list(fitness_scores)
    if not maximize:
        # Invert: shift so all values are positive then invert
        max_val = max(scores)
        scores = [max_val - s + 1e-10 for s in scores]
    else:
        min_val = min(scores)
        if min_val < 0:
            scores = [s - min_val + 1e-10 for s in scores]

    total = sum(scores)
    if total == 0:
        return random.choice(population)

    pick = random.uniform(0, total)
    cumulative = 0.0
    for i, s in enumerate(scores):
        cumulative += s
        if cumulative >= pick:
            return population[i]
    return population[-1]


def rank_based_selection(
    population: List[List[float]],
    fitness_scores: List[float],
    maximize: bool = True,
    **_kwargs,
) -> List[float]:
    """Rank-based selection.

    Individuals are ranked by fitness and selection probability is
    proportional to rank (best = highest rank).  This avoids the scaling
    issues of roulette-wheel selection.
    """
    indexed = list(enumerate(fitness_scores))
    indexed.sort(key=lambda t: t[1], reverse=maximize)
    n = len(population)
    # Rank weights: best gets rank n, worst gets rank 1
    weights = list(range(n, 0, -1))
    total = sum(weights)
    pick = random.uniform(0, total)
    cumulative = 0.0
    for rank, (orig_idx, _) in enumerate(indexed):
        cumulative += weights[rank]
        if cumulative >= pick:
            return population[orig_idx]
    return population[indexed[-1][0]]


SELECTION_OPERATORS = {
    "tournament": tournament_selection,
    "roulette": roulette_wheel_selection,
    "rank": rank_based_selection,
}
