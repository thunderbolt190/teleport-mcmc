# Changelog

## [0.1.0] - In Progress

### Added
- Repository structure, `pyproject.toml`, Apache-2.0 license, and `CITATION.cff`
- GitHub Actions CI workflow (tests run on Python 3.10 and 3.11)
- Phase 1: Random Walk Metropolis-Hastings (RWMH) and Goodman-Weare ensemble sampler baselines
- Algorithm design document (`docs/algorithm_design.md`)
- JAX implementation of `compute_log_weights()` (with hand-verified N=2 test)
- JAX implementation of `one_teleporting_step()`
- `lax.scan` wrapper `teleporting_walkers_jax()`
- Correctness test on 2D Gaussian target (mean and covariance match within `atol=0.1`)
- Correctness test on double-well (bimodal) target, confirming both modes populated
- `jax.jit` compilation on `compute_log_weights`, `one_teleporting_step`, and `teleporting_walkers_jax`
- `tests/test_rwmh.py` and `tests/test_goodman_weare.py`: automated correctness tests for both baseline samplers 
- Phase 2: IAT-vs-N efficiency experiment on the double-well target (`notebooks/03_iat_vs_doublewell.ipynb`)
- `docs/BUGLOG.md` - running record of issues found and resolved during development
- `README.md` - project description, install/usage instructions, status, and links
- Sequential Goodman-Weare implementation, `goodman_weare_sequential_jax`, implementing the standard Goodman & Weare (2010) "Algorithm 2" (one walker updated at a time), distinct from the existing `vmap`-parallel `goodman_weare_jax`
- `tests/test_goodman_weare.py`: added `test_goodman_weare_sequential`, testing the new sequential implementation against the same 2D Gaussian target
- Phase 2: Teleporting walkers vs. Goodman-Weare (both parallel and sequential variants) IAT comparison on the double-well target (`notebooks/04_teleporting_vs_gw.ipynb`)
- Phase 2: Acceptance & teleport probability vs. N, a double-well analog of the paper's Figure 5.5 (`notebooks/05_accept_teleport_vs_n_doublewell.ipynb`)

### Fixed
- `log_prob_doublewell` was missing a scalar reduction, silently corrupting the ensemble weight computation for any N>1; caught during Phase 2 notebook work
- Inconsistent array-indexing pattern in `one_teleporting_step` (direct indexing vs. the documented `jnp.take().squeeze().atleast_1d()` convention)

### Changed
- Consolidated the 2D Gaussian target (previously duplicated separately across three test files) into `teleport/targets.py` as `log_prob_gaussian2d`
- Rewrote `docs/algorithm_design.md` to accurately reflect the implemented code (previously had several inaccuracies); original first draft preserved at `docs/archive/algorithm_design_v1_draft.md`
- Added Phase 2 entries to `benchmarks/results.md` for the Goodman-Weare comparison and the acceptance/teleport-probability-vs-N experiment
