# Contributing

## Development Setup
```bash
git clone https://github.com/[yourusername]/teleport-mcmc
cd teleport-mcmc
pip install -e ".[dev]"
```

## Running Tests
```bash
pytest tests/ -v
```

## Code Style
- Black for formatting
- Ruff for linting
- NumPy-style docstrings on all public functions
- Type hints on all function signatures

## Before Submitting a PR
- All tests pass
- New functionality has tests
- Docstrings are complete
- CHANGELOG.md is updated
