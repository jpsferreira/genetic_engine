#!/usr/bin/env python
"""
Parameter tuning: optimise a PID controller for a simulated system.
====================================================================

A practical use-case: find PID gains (Kp, Ki, Kd) that make a
second-order system track a step reference with minimal overshoot and
settling time.

The "plant" is a simple second-order transfer function simulated via
Euler integration — no external dependencies beyond NumPy.

Usage::

    python -m genetic_opt.sga.examples.parameter_tuning
"""

import random
import math
from typing import List

import numpy as np

from genetic_opt import minimize


# ──────────────────────────────────────────────────────────────
# Plant simulation
# ──────────────────────────────────────────────────────────────

def simulate_pid(
    kp: float,
    ki: float,
    kd: float,
    dt: float = 0.01,
    t_end: float = 5.0,
    setpoint: float = 1.0,
    plant_wn: float = 5.0,
    plant_zeta: float = 0.3,
) -> np.ndarray:
    """Simulate a PID controller driving a second-order plant.

    Plant transfer function:  G(s) = wn² / (s² + 2·zeta·wn·s + wn²)

    Returns an array of shape ``(n_steps, 2)`` with columns ``[time, output]``.
    """
    n_steps = int(t_end / dt)
    t = np.zeros(n_steps)
    y = np.zeros(n_steps)

    # State-space of the plant: dx1/dt = x2, dx2/dt = -wn²·x1 - 2·zeta·wn·x2 + wn²·u
    x1 = 0.0  # output
    x2 = 0.0  # derivative of output

    integral = 0.0
    prev_error = setpoint

    for k in range(n_steps):
        t[k] = k * dt
        y[k] = x1

        error = setpoint - x1
        integral += error * dt
        derivative = (error - prev_error) / dt
        prev_error = error

        u = kp * error + ki * integral + kd * derivative

        # Clamp control effort to prevent numerical blow-up
        u = max(-100.0, min(100.0, u))

        # Euler integration of the plant
        dx1 = x2
        dx2 = -(plant_wn ** 2) * x1 - 2 * plant_zeta * plant_wn * x2 + (plant_wn ** 2) * u
        x1 += dx1 * dt
        x2 += dx2 * dt

    return np.column_stack([t, y])


# ──────────────────────────────────────────────────────────────
# Fitness function
# ──────────────────────────────────────────────────────────────

def pid_cost(gains: List[float]) -> float:
    """Evaluate PID gains.  Lower is better.

    Penalises:
    - Integrated absolute error (IAE)
    - Overshoot beyond the setpoint
    - Control effort (indirectly via large gains)
    """
    kp, ki, kd = gains
    if kp < 0 or ki < 0 or kd < 0:
        return 1e6  # Infeasible — negative gains

    sim = simulate_pid(kp, ki, kd)
    t, y = sim[:, 0], sim[:, 1]
    setpoint = 1.0

    # Integrated Absolute Error
    error = np.abs(y - setpoint)
    iae = np.trapezoid(error, t)

    # Overshoot penalty
    overshoot = max(0.0, (np.max(y) - setpoint) / setpoint)
    overshoot_penalty = 50.0 * overshoot ** 2

    # Settling time estimate (last time |error| > 2% of setpoint)
    settled = np.where(np.abs(y - setpoint) > 0.02 * setpoint)[0]
    settling_time = t[settled[-1]] if len(settled) > 0 else 0.0

    return iae + overshoot_penalty + 0.5 * settling_time


# ──────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────

def main():
    random.seed(42)
    np.random.seed(42)

    print("PID Controller Tuning via Genetic Algorithm")
    print("=" * 50)
    print()
    print("Plant: second-order system (ωn=5, ζ=0.3)")
    print("Objective: minimise IAE + overshoot penalty + settling time")
    print("Variables: Kp, Ki, Kd")
    print()

    bounds = [
        (0.0, 20.0),   # Kp
        (0.0, 15.0),   # Ki
        (0.0, 5.0),    # Kd
    ]

    # --- Optimise ---
    result = minimize(
        pid_cost,
        bounds,
        n_generations=200,
        population_size=100,
        crossover_type="sbx",
        adaptive_mutation=True,
        convergence_threshold=1e-5,
        convergence_generations=30,
    )

    kp, ki, kd = result.x
    print(f"Optimal PID gains:")
    print(f"  Kp = {kp:.4f}")
    print(f"  Ki = {ki:.4f}")
    print(f"  Kd = {kd:.4f}")
    print(f"  Cost = {result.fun:.4f}")
    print(f"  Generations = {result.n_generations}")
    print()

    # Simulate with the optimal gains and report performance
    sim = simulate_pid(kp, ki, kd)
    t, y = sim[:, 0], sim[:, 1]

    overshoot_pct = max(0.0, (np.max(y) - 1.0) / 1.0 * 100)
    settled = np.where(np.abs(y - 1.0) > 0.02)[0]
    settling = t[settled[-1]] if len(settled) > 0 else 0.0
    ss_error = abs(y[-1] - 1.0)

    print("Step response metrics:")
    print(f"  Overshoot     = {overshoot_pct:.2f}%")
    print(f"  Settling time = {settling:.3f} s (2% band)")
    print(f"  Steady-state error = {ss_error:.6f}")

    # Compare with Ziegler–Nichols-style manual tuning
    print()
    print("Comparison with manual tuning (Kp=4, Ki=3, Kd=0.5):")
    manual_cost = pid_cost([4.0, 3.0, 0.5])
    print(f"  Manual cost = {manual_cost:.4f}")
    print(f"  GA cost     = {result.fun:.4f}")
    print(f"  Improvement = {(manual_cost - result.fun) / manual_cost * 100:.1f}%")


if __name__ == "__main__":
    main()
