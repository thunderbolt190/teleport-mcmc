# API Reference

## Core Algorithm

### `teleporting_walkers_jax(init_walkers, log_prob_fn, step_size, n_steps, key)`

Runs the teleporting walkers sampler for `n_steps`.

**Parameters:** 
- `init_walkers` - array, shape `(N, dim)`: Initial ensemble positions
- `log_prob_fn` - callable (static): Target probability function. Takes in walker location and returns a scalar.
- `step_size` - float: Standard deviation of the Gaussian proposal.
- `n_steps` - int (static): Number of steps (individual walker moves).
- `key` - `jax.random.PRNGKey`.

**Returns:** `(final_walkers, chain, accepts, teleports)`
- `final_walkers` - shape `(N, dim)`: Ensemble state after the last step.
- `chain` - shape `(n_steps, N, dim)`: Full ensemble state at every step.
- `accepts` - shape `(n_steps,)`, bool: Whether each step's proposal was accepted.
- `teleports` - shape `(n_steps,)`, bool: Whether each *accepted step moved a
  different walker than the one originally selected to propose from
  (`i != j`), meaning a teleport rather than a local move.
---

### `one_teleporting_step(walkers, log_probs, log_prob_fn, step_size, key)`

Runs a single step of the algorithm. Used internally by
`teleporting_walkers_jax`'s `lax.scan` loop.

**Parameters:**
- `walkers` - shape `(N, dim)`: Current ensemble.
- `log_probs` - shape `(N,)`: Current log π(xᵢ) for each walker.
- `log_prob_fn` - callable (static): Target probability function. Takes in walker location and returns a scalar
- `step_size` - float: Standard deviation of the Gaussian proposal.
- `key` - `jax.random.PRNGKey`.

**Returns:** `(walkers, log_probs, key, accepted, teleported)` - updated
ensemble, updated log-probs, the split key, and the two bools
(acceted - if the move is accepted, teleported - if the move is a teleportation as opposed to a local mutation)

---

### `compute_log_weights(walkers, log_probs, z, sigma)`

Computes the log importance weight for every walker in the ensemble,
given a proposed point `z`. See `docs/math.md` (or
`docs/algorithm_design.md`) for the underlying formula.

**Parameters:**
- `walkers` - shape `(N, dim)`: An array of locations of the walkers.
- `log_probs` - shape `(N,)`: Current log π(xᵢ) for each walker.
- `z` - shape `(dim,)`: The proposed new walker location.
- `sigma` - float: Proposal standard deviation (matches `step_size`
  elsewhere).

**Returns:** `log_w` - shape `(N,)`: Log importance weight per walker.

## Baseline Samplers

### `rwmh_jax(log_prob_fn, init, n_steps, step_size, key)`

Standard Random-Walk Metropolis-Hastings, single chain.

**Parameters:**
- `log_prob_fn` - callable (static): Target probability function. Takes in walker location and returns a scalar.
- `init` - shape `(dim,)`: Starting position of the walker.
- `n_steps` - int (static): Number of Metropolis-Hastings steps.
- `step_size` - float: Proposal standard deviation.
- `key` - `jax.random.PRNGKey`.

**Returns:** `(chain, accept_rate)`
- `chain` - shape `(n_steps, dim)`: Location of walker at every step.
- `accept_rate` - scalar float: Fraction of proposals accepted over the
  full run.

---

### `goodman_weare_jax(log_prob_fn, init_walkers, n_steps, key, n_walkers, a=2.0)`

Goodman-Weare affine-invariant ensemble sampler. Note: this implementation updates all walkers within a step
simultaneously via `vmap`, rather than sequentially one at a time.

**Parameters:**
- `log_prob_fn` - callable (static): Target probability function. Takes in walker location and returns a scalar.
- `init_walkers` - shape `(n_walkers, dim)`: Initial locations of all walkers.
- `n_steps` - int (static): Number of Goodman-Weare steps (Each step proposes new locations for every walker)
- `key` - `jax.random.PRNGKey`.
- `n_walkers` - int (static): Number of walkers.
- `a` - float, default `2.0`. Stretch-move scale parameter.

**Returns:** `(chain, accept_rate)`
- `chain` - shape `(n_steps, n_walkers, dim)`: Locations of the walkers at each step.
- `accept_rate` - scalar float: Fraction of individual walker updates
  accepted over the full run (normalized by `n_walkers * n_steps`).

## Target Distributions

### `log_prob_doublewell(x)`

Bimodal double-well log-density: `-4(x⁴ - x²)`, reduced to a scalar via
`jnp.sum`. Accepts any input shape `(dim,)`. (For `dim > 1`, this gives
a separable product of independent 1D double-wells.) All testing and
experiments in this project use `dim=1` specifically and behavior for
`dim > 1` is mathematically well-defined but not independently verified.

### `log_prob_gaussian2d(x, mean=GAUSSIAN2D_MEAN, covinv=GAUSSIAN2D_COVINV)`

2D Gaussian log-density with default `mean=[2.0, -1.0]`,
`cov=[[1.0, 0.8], [0.8, 1.0]]`. Pass different `mean`/`covinv` (the
inverse covariance) to use a different Gaussian target.
