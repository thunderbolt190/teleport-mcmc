# Teleporting Walkers — Algorithm Design Document

## Purpose
This document defines exactly what will be implemented before any code 
is written. It serves as the reference for correctness checks throughout 
Stage 4.

---

## One Full Step — Pseudocode

Input:
  walkers    — current positions, shape (N, dim)
  log_probs  — log π(xᵢ) for each walker, shape (N,)
  key        — JAX random key
  log_prob_fn — target log probability function
  proposal_fn — proposal kernel q(z | x)
  step_size  — proposal step size

Step 1: Select j uniformly from {0, ..., N-1}
  key, subkey = jax.random.split(key)
  j = jax.random.randint(subkey, shape=(), minval=0, maxval=N)

Step 2: Sample z ~ q(·|x_j)
  key, subkey = jax.random.split(key)
  z = proposal_fn(subkey, walkers[j], step_size)

Step 3: Compute log importance weights for all walkers
  For each i:
    log_w_i = log[ q(x_i|z) + Σ_{k≠i} q(x_i|x_k) ] - log π(x_i)
  Z_xz = logsumexp(log_w)   ← this is log Z(x,z)

Step 4: Sample i ~ categorical(softmax(log_w))
  key, subkey = jax.random.split(key)
  i = jax.random.categorical(subkey, log_w)

Step 5: Form proposed ensemble x'
  x'_i = z
  x'_k = x_k for all k ≠ i
  log_prob of z = log_prob_fn(z)

Step 6: Compute log Z(x', x_i) — the reverse move weights
  Same formula as Step 3 but:
    - ensemble is now x' (position i replaced by z)
    - the "proposed point" is x_i (the old position being deleted)
  Z_x'xi = logsumexp(log_w_reverse)

Step 7: Accept/reject
  log_accept_prob = Z_xz - Z_x'xi
  key, subkey = jax.random.split(key)
  accepted = log(uniform()) < log_accept_prob

Step 8: Update state
  if accepted:
    walkers[i] = z
    log_probs[i] = log π(z)
  else:
    no change

Output:
  new_walkers   — shape (N, dim)
  new_log_probs — shape (N,)
  info:
    accepted  — bool
    teleported — bool, True if i ≠ j

---

## The Numerically Hard Part — Importance Weights

For fixed ensemble x and proposed point z, the log importance weight 
for walker i is:

  log_w_i = log[ q(x_i|z) + Σ_{k≠i} q(x_i|x_k) ] - log π(x_i)

The term inside the log has two parts:

  Part A: q(x_i|z)
    How likely is x_i given the new proposal z?
    Intuition: high if x_i is close to z

  Part B: Σ_{k≠i} q(x_i|x_k)
    How likely is x_i given each other walker?
    Intuition: high if x_i is close to many other walkers
    This is what makes crowded walkers likely to be deleted

For Gaussian proposal q(y|x) = N(y; x, σ²I):
  log q(y|x) = -0.5 * ||y - x||² / σ²  (constant dropped, cancels in ratio)

Part B requires summing over all k≠i for each i.
This is an (N x N) computation — every pair of walkers.
In JAX this is done with vmap, not a Python loop.

Numerical stability — CRITICAL:
  Never compute exp(log_w_i) directly then sum.
  Always use log-sum-exp:
    log Σ exp(aᵢ) = max(a) + log Σ exp(aᵢ - max(a))
  JAX provides: jax.scipy.special.logsumexp(a)

The full weight computation in matrix form:
  - Compute all pairwise log_q(x_i, x_k) → shape (N, N) matrix
  - Diagonal entries (i==k) are excluded from Part B
  - Part A adds log_q(x_i, z) for each i → shape (N,) vector
  - Combine using log-sum-exp for numerical stability
  - Subtract log_probs → shape (N,) vector of log weights

---

## The Reverse Move — Why It's Needed

The acceptance ratio is:

  min(1, Z(x,z) / Z(x',xᵢ))

Z(x,z) is computed in Step 3 — log sum of weights over current ensemble.

Z(x',xᵢ) is the same computation but:
  - The ensemble is x' where position i has been replaced by z
  - The proposed point is xᵢ (the old walker being evaluated for deletion)

What changes between Z(x,z) and Z(x',xᵢ):
  - Row i of the pairwise matrix changes (x_i is now z)
  - Column i of the pairwise matrix changes (x_i is now z)
  - The query point changes from z to x_i

This is not a full recomputation from scratch.
Only the entries involving index i change.
This is an optimization opportunity in Stage 4 implementation.

---

## JAX Implementation Notes

carry in lax.scan:
  (walkers, log_probs, key)
  shapes: ((N, dim), (N,), (2,))

output collected at each step:
  walkers shape (N, dim)
  — store full ensemble at each step, flatten after for statistics

static arguments:
  log_prob_fn — not a JAX array, must be static
  n_steps     — determines scan length, must be static
  N           — used in randint, must be a concrete Python int

jax.random.categorical:
  Takes unnormalized log probabilities directly
  Returns integer index sampled from categorical distribution
  This is how Step 4 is implemented — no manual CDF needed

jnp.where for Step 8:
  new_walkers = jnp.where(accepted, 
                    walkers.at[i].set(z), 
                    walkers)
  new_log_probs = jnp.where(accepted,
                    log_probs.at[i].set(log_prob_z),
                    log_probs)


