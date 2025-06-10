# genetic_engine

A Python library for genetic algorithms and optimization.

## Features

- Modular genetic algorithm implementation
- Example scripts for polynomial fitting and other problems
- Utilities for monitoring, exporting, and visualizing results
- Easily extensible for custom optimization tasks

## Project Structure

```text
sga/                # Core genetic algorithm implementation
  optimizer.py      # Main optimizer logic
  examples/         # Example problems (e.g., polynomial_fit.py)
  utils/            # Utilities (export, monitor, visualization)
scripts/            # Example runner scripts
tests/              # Unit tests
pyproject.toml      # Project metadata and dependencies
uv.lock             # Lock file for dependencies
README.md           # Project documentation
```

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/genetic_engine.git
cd genetic_engine
pip install -e .
```

Or use your preferred Python environment manager.

## Usage

Run example scripts:

```bash
python scripts/run_polynomial_example.py
```

Or use the library in your own Python code:

```python
from sga.optimizer import Optimizer
# ... set up and run your optimization ...
```

## Testing

Run unit tests with:

```bash
python -m unittest discover tests
```

## License

[MIT License](LICENSE)

---