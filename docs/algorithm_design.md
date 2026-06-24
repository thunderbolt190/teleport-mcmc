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
