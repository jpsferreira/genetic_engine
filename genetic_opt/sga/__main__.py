"""Main entry point for the genetic_opt package."""

import os


def main():
    """Display welcome message and redirect to example script."""
    print("Genetic Optimization Library")
    print("===========================")
    print("")
    print("To run examples, use the dedicated scripts in the 'scripts' directory:")
    print("  python scripts/run_examples.py")
    print("")
    print("Or run a specific example directly:")
    print("  python scripts/run_polynomial_example.py")
    print("")
    print("For more information, see the README.md file.")

    # Check if scripts directory exists
    if not os.path.exists("scripts"):
        print("\nWARNING: 'scripts' directory not found. You may be running this")
        print("directly from an installed package. Please clone the repository")
        print("to access example scripts, or see the README for usage examples.")


if __name__ == "__main__":
    main()
