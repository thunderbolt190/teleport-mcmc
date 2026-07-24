# Bug / Issue Log — teleport-mcmc

Running record of issues found during development, how they were caught, and how they were resolved. Intended as a companion to the test suite and CHANGELOG — a place to show *how* correctness was actually verified, not just that tests are green.

---

## 1. Table 1 target mismatch (planning error, not a code bug)

**Found:** while scoping Phase 2 (Issue #7). The task was originally written as "reproduce Table 1 (IAT for N=1,10,50) on the double-well target."

**Problem:** Table 1 in the paper (Section 5.2.1) reports IAT for the **univariate Gaussian process regression posterior** (θ = α, ρ, σ, Cauchy⁺(0,3) priors) — not the double-well target. The double-well example in the paper (Section 5.1) is only used to show PDE convergence (Figures 5.1–5.3); it never reports an IAT table. The original task conflated two separate experiments.

**Resolution:** Rewrote Issue #7 to be a **Table-1-style efficiency check on double-well** rather than a literal reproduction, with the distinction stated explicitly in the issue, the notebook, and `benchmarks/results.md`.

**Status:** Resolved.

---

## 2. Indexing convention inconsistency in `one_teleporting_step`

**Found:** while verifying a documented convention that dynamic indexing inside `lax.scan` should use `jnp.take().squeeze().atleast_1d()`, not direct indexing.

**Problem:** Only the delete-extraction line used the documented pattern; `z = walkers[j] + ...` still used direct indexing. No live bug reproduced empirically, but the code was inconsistent with its own documented convention.

**Resolution:** Applied the same pattern to the `z` line for consistency/defensiveness. Full test suite rerun and confirmed passing.

**Status:** Resolved.

---

## 3. `log_prob_doublewell` missing scalar reduction — **previously-hidden bug**

**Found:** during notebook Step 2, when `teleports.shape` came back `(5000, 20)` instead of `(5000,)`.

**Problem:** `log_prob_doublewell(x)` returned shape `(1,)` instead of a scalar, missing a `jnp.sum(...)`. This silently corrupted `compute_log_weights`' output shape from `(N,)` to `(N,N)`, causing `jax.random.categorical` to make N independent per-row decisions instead of one shared ensemble-wide decision.

**Why the test suite didn't catch it:** `test_double_well_teleporting_walkers` only checks mode coverage — weak enough to pass despite the broken math.

**Resolution:**
```python
def log_prob_doublewell(x):
    return jnp.sum(-4.0 * (x**4 - x**2))
```
Verified independently: full suite re-passes, `teleports.shape` confirmed correct.

**Status:** Resolved.

---

## 4. Algorithm design doc: incorrect claim about `vmap`

**Found:** doc-vs-code review of the algorithm design document.

**Problem:** Doc claimed the crowding-term computation "is done with vmap." Actual implementation uses broadcasting — no `vmap` involved.

**Resolution:** Corrected to describe broadcasting accurately.

**Status:** Resolved.

---

## 5. Algorithm design doc: Open Questions

**Found:** same review.

- **Q1 (traced index in `.at[i].set()`):** Resolved — confirmed by test suite and direct verification.
- **Q2 (reverse-move ordering):** Removed as a listed question — JAX's array immutability makes this safe by construction, not a real implementation risk worth tracking as open.
- **Q3 (N=1 edge case):** Resolved — both analytically (the weight formula reduces exactly to the standard MH acceptance ratio at N=1) and empirically (N=1 runs in the IAT experiment show mode coverage ~0.50 across 10 seeds, confirming the implementation doesn't break at this edge case).

**Status:** Resolved.

---

## 6. Notebook plot silently saved outside the repo

**Found:** while building `notebooks/03_iat_vs_doublewell.ipynb`, cell 8 (the plot).

**Problem:** `os.makedirs("../benchmarks/results", ...)` assumed the working directory was `notebooks/`, but cell 2 only runs `%cd teleport-mcmc` (the repo root) — the path actually resolved to one level above the repo entirely. No error was thrown; the plot displayed correctly inline while silently saving to the wrong location.

**Resolution:** Changed both paths to `benchmarks/results/...` (no `../`). Verified via `!ls -la` for correct file size, then confirmed via git log that the committed image only updated once the fix was applied.

**Status:** Resolved.

---

## 7. Blank plot from save/show ordering in Colab

**Found:** first attempt at saving the IAT-vs-N plot. The pushed PNG was a valid file but rendered as completely blank.

**Problem:** `plt.savefig(...)` was called in a separate cell from the plotting code. Colab's inline backend clears the current figure after a cell displays it, so the later `savefig` cell had nothing left to save. Initially misdiagnosed as a GitHub rendering issue — only caught properly by checking actual pixel content (0 non-white pixels), not just file validity.

**Resolution:** Moved plotting, labeling, `savefig`, and `plt.show()` into a single cell, `savefig` before `show()`.

**Status:** Resolved.

---

## 8. Goodman-Weare silently produces a frozen walker at N=1

**Found:** while running the teleporting-vs-GW IAT comparison experiment,
checking whether N=1 could be included as a comparison point.

**Problem:** Both `goodman_weare_jax` and `goodman_weare_sequential_jax`
require at least one other walker to propose a move — the stretch-move
formula needs a "complementary" walker to move relative to. With
`n_walkers=1`, `select_complementary_jax` returns an out-of-bounds index
(the only other walker's index doesn't exist). JAX's default array
indexing silently clips out-of-bounds indices rather than raising an
error, so the complementary walker resolves to the walker itself. This
makes every proposal identical to the walker's current position.

**Status:** Not fixed. Currently avoided by excluding N=1 from any GW
comparison and starting at N=2 (the smallest N at which GW is
well-defined). A proper fix
would have `goodman_weare_jax`/`goodman_weare_sequential_jax` raise an
explicit error for `n_walkers < 2`, rather than silently returning
degenerate results.

---

## Summary

| # | Issue | Type | Status |
|---|-------|------|--------|
| 1 | Table 1 target mismatch | Planning/scoping | ✅ Resolved |
| 2 | Inconsistent indexing pattern | Defensive fix | ✅ Resolved |
| 3 | `log_prob_doublewell` missing reduction | **Real correctness bug** | ✅ Resolved |
| 4 | Design doc: incorrect vmap claim | Documentation | ✅ Resolved |
| 5 | Design doc: Open Questions | Documentation | ✅ Resolved |
| 6 | Notebook plot saved outside repo (wrong path) | Real bug (silent) | ✅ Resolved |
| 7 | Blank plot from save/show ordering | Real bug (silent) | ✅ Resolved |
| 8 | Goodman-Weare frozen walker at N=1 | Correctness bug (silent) | ⚠️ Open — not fixed |
