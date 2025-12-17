#!/usr/bin/env python
"""
Main entry point for running genetic optimization examples.

This script lists available examples and directs users to the specific entry points.
"""

import os
import sys

# Add the parent directory to path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    """Display available examples and instructions on how to run them."""
    print("Genetic Optimization Examples")
    print("=============================")
    print("Available examples:")
    print("")
    print("1. Polynomial Fitting Example")
    print("   Demonstrates fitting a polynomial curve using genetic optimization.")
    print("   Run with: python scripts/run_polynomial_example.py [options]")
    print("   Options:")
    print("     --no-monitor     Disable live terminal monitor")
    print("     --no-track       Disable population tracking")
    print("     --no-analysis    Disable migration analysis")
    print("     --enhanced-viz   Enable enhanced visualizations including:")
    print("                      - Fitness landscape visualizations")
    print("                      - 3D migration trajectories")
    print("                      - Gene correlation analysis")
    print("")
    print("For more information, see the README.md or use the --help option")
    print("with any specific example script.")


if __name__ == "__main__":
    main()
