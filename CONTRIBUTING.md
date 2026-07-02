# Contributing

## Development Setup
```bash
git clone https://github.com/thunderbolt190/teleport-mcmc
cd teleport-mcmc
pip install -e ".[dev]"
```

## Running in Google Colab
```
!git clone https://github.com/thunderbolt190/teleport-mcmc
%cd teleport-mcmc
!pip install -e ".[dev]"
!pytest tests/ -v
```

## After installation, enable double precision (important for this project)
```
import jax
jax.config.update("jax_enable_x64", True)
```
  
## Running Tests
```bash
pytest tests/ -v
```
## Code Style
- Black for formatting
- Ruff for linting
- NumPy-style docstrings on all public functions
  
## Notes for Contributors
- Before touching the core algorithm: Read Section 2 of the Lindsey–Weare–Zhang (2022) paper:
https://arxiv.org/abs/2106.02686
- Any changes to the teleporting / interacting walkers algorithm must include strong numerical assertions (not just shape checks). This prevents subtle statistical or floating-point bugs.

## Before Submitting a PR
- All tests pass
- New functionality has tests
- Docstrings are complete
- CHANGELOG.md is updated
