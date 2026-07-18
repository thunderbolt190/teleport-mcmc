# Teleporting Walkers — Algorithm Design Document

## Purpose
This document defines exactly what will be implemented before any code 
is written. It serves as the reference for correctness checks throughout 
Stage 4.

---

## One Full Step

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
  Z_xz = logsumexp(log_w)  - this is log Z(x,z)

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
    - the proposed point is x_i (the old position being deleted)
  Z_x'xi = logsumexp(log_w_reverse)

Step 7: Accept/reject
  log_accept_prob = Z_xz - Z_x'xi
  key, subkey = jax.random.split(key)
  accepted = log(jax.random.uniform(key)) < log_accept_prob

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

## Importance Weights

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
In JAX this is done via broadcasting (pairwise differences computed with Numpy-style broadcasting: walkers[:, None, :] - walkers[None, :, :], instead of a python loop

Numerical stability:
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

## The Reverse

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

---

## Questions That Must Be Answered Before Coding

These were answered during design. Recorded here as a reference.

**Q1: What JAX operation computes Part B without a Python loop?**

Part B for walker i is Σ_{k≠i} q(x_i|x_k).
In matrix form:
  - Compute full (N x N) matrix M where M[i,k] = log_q(x_i, x_k)
  - Set diagonal to -inf (excludes k==i term)
  - Row-wise logsumexp gives log Part B for each i
  JAX operation: jax.scipy.special.logsumexp(M, axis=1)
  where diagonal has been masked with -inf

**Q2: What exactly changes between Z(x,z) and Z(x',xᵢ)?**

Z(x,z):
  ensemble = x = (x_0, ..., x_N)
  query point = z
  pairwise matrix M[i,k] = log_q(x_i, x_k), diagonal masked
  Part A vector: log_q(x_i, z) for each i

Z(x',xᵢ):
  ensemble = x' where x'_i = z, x'_k = x_k for k≠i
  query point = x_i (old position)
  pairwise matrix M'[r,k] = log_q(x'_r, x'_k), diagonal masked
    — row i changes: x'_i = z so M'[i,k] = log_q(z, x_k)
    — col i changes: M'[r,i] = log_q(x'_r, z)
  Part A vector: log_q(x'_r, x_i) for each r

**Q3: Is the lax.scan carry correct?**

carry = (walkers, log_probs, key)
shapes = ((N, dim), (N,), (2,))
output per step = walkers shape (N, dim)

Yes this is correct.
To track teleportation events, add to output:
  (walkers, accepted, teleported)
  shapes: ((N, dim), (), ())
  teleported = (i != j) after accept step

**Q4: How is weighted categorical sampling done in JAX?**

jax.random.categorical(key, logits)
  - logits = log_w (the unnormalized log weights from Step 3)
  - internally applies logsumexp normalization
  - returns integer index in {0, ..., N-1}
  - no manual softmax or CDF needed

---

## Verification Plan

After implementation, verify in this exact order:

1. compute_log_weights() in isolation
   - Input: 5 walkers, known positions, Gaussian proposal
   - Compute by hand for N=2 case
   - Assert JAX output matches hand calculation

2. Single step on N=2 walkers
   - Verify output shape is correct
   - Verify accepted is bool
   - Verify teleported is bool
   - Verify when walkers are far apart: reduces to standard MH

3. Full scan on 2D Gaussian (unimodal)
   - 50 walkers, 5000 steps, burn-in 500
   - Assert mean within atol=0.1 of [2.0, -1.0]
   - Assert cov within atol=0.1 of [[1.0,0.8],[0.8,1.0]]
   - Assert results match Anna Zhang reference on same seed

4. Full scan on double-well (bimodal)
   - 20 walkers, 5000 steps, all initialized at left mode
   - Assert both modes populated
   - Assert plain GW fails same test

---

## Verified Implementations Details

Thesewere verified and resolved during implementation:

- [x] Does jnp.where work correctly when i is a traced integer 
      index into walkers? Need to verify walkers.at[i].set(z) 
      inside jit.
      Resolved: Yes, it does. All 5 tests in the test suite use this and pass. 
- [x] What happens when N=1? Should reduce to standard MH.
      Verify this edge case explicitly.
      Resolved: Confirmed both analytically and empirically.

      Analytically: at N=1, the pairwise matrix M is 1x1 with its only
      entry being -inf, so
      log_w = log q(x0|z) - log pi(x0). By proposal symmetry,
      log q(z|x0) = log q(x0|z), so these terms cancel exactly in the
      acceptance ratio, leaving
      log_accept_prob = log pi(z) - log pi(x0), which the standard MH
      acceptance ratio for a symmetric proposal. Also, i = j = 0 always
      at N=1, so teleported is always False, consistent with there
      being no second walker to teleport between.

      Empirically: N=1 runs in the IAT experiment
      (notebooks/03_iat_vs_doublewell.ipynb) show mode coverage ~0.50
      across all 10 seeds, confirming the implementation doesn't break
      or get stuck at this edge case (1x1 weight matrix, single-option
      categorical sampling) which is consistent with, though not on its own
      proof of, the analytical MH-reduction above.
