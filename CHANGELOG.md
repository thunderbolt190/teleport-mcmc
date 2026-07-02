# Changelog

## [0.1.0] - In Progress

### Added
- Repository structure, `pyproject.toml`, Apache-2.0 license, and `CITATION.cff`
- GitHub Actions CI workflow (tests run on Python 3.10 and 3.11)
- Stage 3 Random Walk Metropolis-Hastings (RWMH) and Goodman-Weare ensemble sampler, verified with strong numerical assertions
- Algorithm design document (`docs/algorithm_design.md`)
- JAX implementation of `compute_log_weights()` (with hand-verified N=2 test)
- JAX implementation of `one_teleporting_step()`
- `lax.scan` wrapper `teleporting_walkers_jax()`
- Correctness test on 2D Gaussian target (mean and covariance match within `atol=0.1`)
