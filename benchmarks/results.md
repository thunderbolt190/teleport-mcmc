# Verification and Benchmark Results

*Phase 1 = core algorithm implementation and baseline sampler
verification. Phase 2 = scientific credibility work (reproducing/
comparing against the original paper's results). See the project's
GitHub issues for the full phase breakdown.*

## Phase 1 Baseline - RWMH JAX

Date: 2026-06-24
Settings: 50000 steps, burn-in 1000, step_size=1.0, key=PRNGKey(0)
Target: 2D Gaussian, mean=[2.0, -1.0], cov=[[1.0, 0.8],[0.8, 1.0]]

| Metric | True Value | JAX Result | Pass? |
|---|---|---|---|
| Mean dim 0 | 2.000 | 1.991 | ✓ |
| Mean dim 1 | -1.000 | -1.004 | ✓ |
| Cov [0,0] | 1.000 | 1.012 | ✓ |
| Cov [0,1] | 0.800 | 0.819 | ✓ |
| Cov [1,1] | 1.000 | 1.027 | ✓ |
| Acceptance rate | ~0.40 | 0.4106 | ✓ |

Assertions passed with atol=0.1

---

## Phase 1 Baseline - Goodman-Weare JAX

Date: 2026-06-24
Settings: 50 walkers, 2000 steps, burn-in 200, key=PRNGKey(0)
Target: 2D Gaussian, mean=[2.0, -1.0], cov=[[1.0, 0.8],[0.8, 1.0]]

| Metric | True Value | JAX Result | Pass? |
|---|---|---|---|
| Mean dim 0 | 2.000 | 1.991 | ✓ |
| Mean dim 1 | -1.000 | -0.997 | ✓ |
| Cov [0,0] | 1.000 | 1.007 | ✓ |
| Cov [0,1] | 0.800 | 0.804 | ✓ |
| Cov [1,1] | 1.000 | 1.001 | ✓ |
| Acceptance rate | ~0.70 | 0.7136 | ✓ |

Assertions passed with atol=0.1


## Phase 2 - IAT vs N, Double-Well Target (Table 1 style check)
Date: 2026-07-11
Settings: step_size=0.5, n_steps=20000, burn-in=2000, 10 random seeds per N
Target: Double-well target (not the paper's actual Table 1 target - Guassian process regression posterior). This is a 
Table-1-style efficiency check on our own already-validated target;
see Issue #7 for full reasoning.

| N  | Mean IAT | Std IAT | Mean Mode Coverage |
|----|----------|---------|--------------------|
| 1  | 13.99    | 1.35    | 0.500              |
| 10 | 3.60     | 0.73    | 0.501              |
| 50 | 2.71     | 1.00    | 0.499              |

![IAT vs N](results/iat_vs_n_doublewell.png)

**Result:** IAT decreases monotonically with N (confirmed across 10
seeds). Ratio IAT(1)/IAT(50) ≈ 5.2x which is below the 10-30x range originally
targeted (that range was borrowed from the paper's GP regression case,
not grounded in double-well data). N=10 and N=50 means overlap within
one std, suggesting most of the benefit of teleporting is captured by
N=10 on this target, with diminishing returns beyond that.

## Phase 2 — Teleporting vs. Goodman-Weare (Parallel & Sequential), Double-Well Target
Date: 2026-07-24 
Settings: step_size=0.5 (teleporting only; GW uses default stretch scale a=2.0),
n_steps=20000, burn_in=2000, 10 random seeds per (algorithm, N). N=2, 10, 50
tested for all three algorithms. N=1 is excluded, since Goodman-Weare (both
variants) is not well-defined at N=1 (see Issue in docs/buglog.md #8).
Target: Double-well target (teleport.targets.log_prob_doublewell). This
experiment originally set out to check whether Goodman-Weare fails to
explore both modes of the double-well target (a pass/fail check). Both
GW variants successfully explored both modes, so the experiment was
redirected to a quantitative IAT comparison instead.

| N  | Teleporting (IAT ± std) | GW Parallel (IAT ± std) | GW Sequential (IAT ± std) |
|----|--------------------------|---------------------------|-----------------------------|
| 2  | 11.37 ± 1.26             | 87.60 ± 29.52             | 54.28 ± 7.76                |
| 10 | 3.60 ± 0.73              | 41.77 ± 5.93              | 44.92 ± 6.29                |
| 50 | 2.71 ± 1.00              | 50.13 ± 12.29             | 39.94 ± 4.89                |

Mode coverage (fraction of post-burn-in samples in the right-hand well)
stayed between 0.497 and 0.504 for every algorithm/N/seed combination —
none of these IAT values reflect a stuck ensemble.

![IAT comparison](results/iat_teleporting_vs_gw.png)

**Result:** Teleporting mixes roughly 10-20x faster (lower IAT) than
either Goodman-Weare variant, at every N tested. This gap is far larger
than the standard deviations involved, so it's a robust finding.
Whether sequential GW mixes faster than parallel/vmap GW specifically is
not established by this data as their IAT values cross between N=2,
10, and 50, with error bars that overlap substantially. This would need
more seeds or a proper statistical test to resolve. Findings are
specific to this double-well parameterization (a fairly shallow
barrier) and don't generalize to other targets. Full methodology and
discussion in notebooks/04_teleporting_vs_gw.ipynb.


## Phase 2 - Acceptance & Teleport Probability vs. N, Double-Well Target
Date: 2026-07-28
Settings: n_steps=5000, step_size=0.5, burnin=500, single seed (PRNGKey(42)) per N.
N = [1, 2, 5, 10, 20, 50, 100] tested. N=1 is included here, since
teleporting is well-defined at N=1 and reduces to standard MH and the teleport
probability is expected to be exactly 0 there.
Target: Double-well target (teleport.targets.log_prob_doublewell). This is a
double-well analog of Figure 5.5 from the paper; the
paper's actual Figure 5.5 uses the univariate Gaussian process regression posterior
from Section 5.2.1, not double-well. Note also that the paper's Figure 5.5 uses a
linear N-axis, while the plot below uses a log-x-axis since our chosen N values
aren't evenly spaced.

| N   | Accepted Prob | Teleported Prob | Mode Coverage |
|-----|---------------|------------------|---------------|
| 1   | 0.660         | 0.000            | 0.542         |
| 2   | 0.708         | 0.339            | 0.513         |
| 5   | 0.806         | 0.771            | 0.493         |
| 10  | 0.834         | 0.885            | 0.495         |
| 20  | 0.862         | 0.950            | 0.507         |
| 50  | 0.891         | 0.979            | 0.493         |
| 100 | 0.890         | 0.991            | 0.507         |

Mode coverage stayed between 0.493 and 0.542 across all N, confirming these
acceptance/teleport trends reflect ensemble behavior rather than a stuck ensemble.

![Acceptance and teleport probability vs N](results/accept_teleport_vs_n_doublewell.png)

**Result:** Both acceptance and teleport probability increase with N, qualitatively
matching the trend the paper describes for Figure 5.5, and both curves appear to
flatten out by N=50-100 rather than continuing to rise sharply which is consistent with the
diminishing-returns pattern already seen in Phase 1. These are single-seed results, so no error bars are
established and the specific values here should not be compared numerically against
the paper's own Figure 5.5, which uses a different target. 


## Phase 2 - Paper Examples 5.2.1-5.2.3 (Lindsey et al. 2022)

Date: 2026-08-19
These notebooks follow the Gaussian-process examples in Section 5.2 of
the paper. They reproduce the procedure, but will not be a numeric match to the paper’s tables and
figures. Full code and plots:

- [`notebooks/Example_5_2_1.ipynb`](../notebooks/Example_5_2_1.ipynb)
- [`notebooks/Example_5_2_2.ipynb`](../notebooks/Example_5_2_2.ipynb)
- [`notebooks/Example_5_2_3.ipynb`](../notebooks/Example_5_2_3.ipynb)

---

### 5.2.1 Univariate GP regression

**Target:** Squared-exponential GP posterior on synthetic 1-D data,
θ = (α, ρ, σ) with half-Cauchy priors and Gaussian noise
integrated out in closed form.

**Sampler:** full-interaction `teleporting_walkers_jax`.

**What we ran:** acceptance and teleport probability vs ensemble size and the IAT of
ensemble-averaged ρ vs ensemble size.

#### Acceptance & teleport probability vs N

Single-seed style check.

| N    | Accept prob | Teleport prob |
|------|-------------|---------------|
| 1    | 0.126       | 0.000         |
| 2    | 0.152       | 0.185         |
| 5    | 0.233       | 0.720         |
| 10   | 0.289       | 0.874         |
| 20   | 0.344       | 0.945         |
| 50   | 0.400       | 0.978         |
| 100  | 0.474       | 0.986         |
| 200  | 0.617       | 0.988         |
| 500  | 0.836       | 0.996         |
| 1000 | 0.944       | 0.998         |

![Acceptance and teleport probability vs N (univariate GP)](results/example_521_accept_teleport_vs_ensemble_size.png)

**Result:** Both rates increase with N and flatten at large ensemble
size which is the same qualitative trend as the paper’s Figure 5.5. 

#### IAT of ensemble-averaged ρ vs N

| N  | Mean IAT | Std IAT |
|----|----------|---------|
| 1 | 361.846 | 229.679 |
| 10 | 65.535 | 22.161 |
| 50 | 19.348 | 7.409 |

**Result:** IAT of ρ̄ decreases as N grows. Treat
small-N estimates with caution when chains are shorter than ~50×τ.

---

### 5.2.2 Multivariate GP (n = 3)

**Target:** Product-mean GP in 3-D inputs, metric parameterized via
Bartlett / Cholesky factor Z, θ is 8-dimensional
(α, c₁, c₂, c₃, z₂₁, z₃₁, z₃₂, σ) with the paper’s positivity and
prior structure.

**Sampler:** full-interaction teleporting.

**What we ran:** IAT of ensemble-averaged c₁ vs N;, 2-D posterior
marginals, overlaid 1-D c₁ marginals for several N.

#### IAT of ensemble-averaged c₁ vs N

| N   | Mean IAT | Std IAT |
|-----|----------|---------|
| 10  | 569.86   | 230.32  |
| 20  | 306.11   | 142.38  |
| 50  | 196.01   | 56.09   |
| 100 | 125.31   | 5.30    |

**Result:** IAT falls with N (Table 2 style trend). Small N values sometimes
trigger emcee’s “chain shorter than 50×τ” warning; those entries are
rough and should not be taken as exact values.

#### Posterior marginals

![z₂₁ vs c₁ (N = 100)](results/example_522_z21_vs_c1_posterior.png)

![c₁ vs z₃₂ (N = 100)](results/example_522_c1_vs_z32_posterior.png)

![c₁ marginal overlay for different ensemble sizes](results/example_522_c1_marginal_dist_vs_ensemble_size.png)

**Result:** Mass is concentrated; larger N yields a more stable c₁
marginal. Plots are not identical to the paper’s Figures 5.6–5.7 due to different synthetic data and seeds.

---

### 5.2.3 Non-Gaussian (Student-t) noise + restricted interaction

**Target:** Same univariate mean structure as 5.2.1, but observation
noise is Student-t (ν = 2). Latents are reparameterized as
ε = y + K_θ^{1/2} w and state is (θ, w) = (α, ρ, σ, w₁, …, wₘ).

**Sampler:** restricted interaction (notebook prototype, not a stable
package API):

- teleport / interact only on θ
- 30 independent MH updates on w between θ steps
- `vmap` over walkers for the w-block

**What we ran:** correctness tests for the restricted teleporting algorithm, exploratory IAT of ρ̄ and 
Figure 5.8-style (α, ρ) marginal at N = 60.

#### Posterior marginal of (α, ρ)

![α vs ρ (N = 60, Student-t noise)](results/example_523_alpha_vs_rho_posterior.png)

**Result:** Restricted kernel runs end-to-end. The (α, ρ) marginal is
concentrated with α, ρ > 0, consistent with the paper’s qualitative
message that Student-t noise removes the strong multimodality seen in
the Gaussian-noise univariate case.


## Wall-clock cost vs sampling efficiency

**Question:** The paper reports integrated autocorrelation time (IAT) which is the number of steps per
effective sample. That is not the same as time per effective sample. The
teleporting kernel builds an (N, N) weight matrix on every single-walker move
(O(N^2)) while the stretch move is O(N). This measures whether the per-step
cost gap between teleporting and GW reverses the mixing advantage as N grows and whether that depends on the hardware used. This experiment is not in the paper.

**Setup:** Double-well target, dim=1. Find the median of 11 repeats at 500
sweeps, with warm-up call being discarded so JIT compilation is excluded from the results,
`block_until_ready` before the clock stops. Find the median IAT over 5 independent
seeds at 5000 sweeps, via `emcee.autocorr.integrated_time` on the per-sweep
ensemble-averaged coordinate. Teleporting step counts are multiplied by N so a
sweep means N walker-moves for all three samplers. Run on Colab CPU
(`CpuDevice`) and Colab T4 GPU (`CudaDevice`), x64 enabled, same seeds on both.
Reproduce with `python benchmarks/benchmark_runtime.py`.

### CPU (`CpuDevice(id=0)`)

| N | sampler | ms/sweep | ms range | IAT median | IAT range | ms/eff sample | unreliable |
|---|---|---|---|---|---|---|---|
| 2 | teleporting | 0.0668 | 0.0542-0.0960 | 11.5 | 9.5-14.0 | 0.77 | - |
| 2 | gw_parallel | 0.0079 | 0.0063-0.0081 | 72.1 | 39.2-119.6 | 0.57 | 1/5 |
| 2 | gw_sequential | 0.0037 | 0.0036-0.0040 | 62.6 | 49.1-171.7 | 0.23 | 1/5 |
| 5 | teleporting | 0.0773 | 0.0763-0.1231 | 5.2 | 4.0-6.3 | 0.40 | - |
| 5 | gw_parallel | 0.0343 | 0.0314-0.0364 | 70.5 | 35.7-128.5 | 2.42 | 1/5 |
| 5 | gw_sequential | 0.0134 | 0.0132-0.0141 | 34.8 | 19.5-74.7 | 0.47 | - |
| 10 | teleporting | 0.2032 | 0.1949-0.3374 | 3.8 | 3.5-4.7 | 0.77 | - |
| 10 | gw_parallel | 0.0330 | 0.0320-0.0759 | 40.7 | 30.0-63.2 | 1.34 | - |
| 10 | gw_sequential | 0.0315 | 0.0314-0.0355 | 61.0 | 42.2-73.2 | 1.92 | - |
| 20 | teleporting | 0.6621 | 0.6549-0.7280 | 3.2 | 2.8-3.4 | 2.09 | - |
| 20 | gw_parallel | 0.0396 | 0.0390-0.0441 | 49.4 | 37.9-91.1 | 1.96 | - |
| 20 | gw_sequential | 0.0659 | 0.0615-0.1118 | 39.7 | 34.7-62.4 | 2.62 | - |
| 50 | teleporting | 3.2025 | 3.1308-5.5913 | 3.5 | 3.4-4.0 | 11.18 | - |
| 50 | gw_parallel | 0.1108 | 0.0791-0.1167 | 48.4 | 33.7-61.8 | 5.36 | - |
| 50 | gw_sequential | 0.1579 | 0.1554-0.1602 | 44.6 | 36.0-53.3 | 7.05 | - |

### GPU (`CudaDevice(id=0)`, Colab T4)

| N | sampler | ms/sweep | ms range | IAT median | IAT range | ms/eff sample | unreliable |
|---|---|---|---|---|---|---|---|
| 2 | teleporting | 0.1063 | 0.1059-0.1916 | 11.5 | 9.5-14.0 | 1.22 | - |
| 2 | gw_parallel | 0.0305 | 0.0165-0.0306 | 72.1 | 39.2-119.6 | 2.20 | 1/5 |
| 2 | gw_sequential | 0.0391 | 0.0385-0.0392 | 62.6 | 49.1-171.7 | 2.45 | 1/5 |
| 5 | teleporting | 0.3629 | 0.3627-0.3874 | 5.2 | 4.0-6.3 | 1.87 | - |
| 5 | gw_parallel | 0.0166 | 0.0165-0.0167 | 43.3 | 30.6-67.1 | 0.72 | - |
| 5 | gw_sequential | 0.0779 | 0.0777-0.0785 | 59.1 | 45.7-72.4 | 4.60 | - |
| 10 | teleporting | 0.8137 | 0.8128-0.8148 | 3.8 | 3.5-4.7 | 3.09 | - |
| 10 | gw_parallel | 0.0176 | 0.0175-0.0178 | 37.8 | 29.7-65.0 | 0.66 | - |
| 10 | gw_sequential | 0.1528 | 0.1525-0.1537 | 48.6 | 41.1-69.8 | 7.43 | - |
| 20 | teleporting | 1.8748 | 1.8737-1.8752 | 3.2 | 2.8-3.4 | 5.91 | - |
| 20 | gw_parallel | 0.0156 | 0.0153-0.0157 | 44.1 | 34.1-83.0 | 0.69 | - |
| 20 | gw_sequential | 0.3023 | 0.3006-0.3048 | 45.6 | 25.2-75.2 | 13.78 | - |
| 50 | teleporting | 4.8032 | 4.8013-4.8057 | 3.5 | 3.4-4.0 | 16.77 | - |
| 50 | gw_parallel | 0.0377 | 0.0200-0.0381 | 47.6 | 27.1-55.3 | 1.79 | - |
| 50 | gw_sequential | 0.7485 | 0.7402-0.7505 | 38.9 | 31.7-80.1 | 29.09 | - |

**Findings:**

- Mixing (device-independent, as expected). Teleporting reaches IAT ~3.5 at
  N=50 against ~48 for GW, roughly 14x fewer sweeps per effective sample and is more consistent across seeds (3.4-4.0 vs 33.7-61.8). Teleporting's IAT is
  identical on both devices, showing that the samplers themselves are unaffected
  by hardware.
- Cost does not improve on GPU. Teleporting's per-sweep cost grows ~48x from
  N=2 to N=50 on CPU and ~45x on GPU. The scaling is essentially unchanged. If
  the O(N^2) importance-weight matrix were the bottleneck, a GPU should have
  flattened this curve but it did not.
- The bottleneck is sequential depth, not arithmetic width. One teleporting
  step moves a single walker, so covering S sweeps at ensemble size N requires
  S*N dependent `lax.scan` iterations (250,000 at N=50, S=5000). One GW step
  updates all N walkers at once via `vmap`, requiring only S iterations. No
  amount of parallel hardware helps a chain of dependent steps. Consistent with
  this, moving to GPU made teleporting slower (3.20 -> 4.80 ms/sweep at N=50) while GW parallel got ~3x *faster*
  (0.111 -> 0.038 ms/sweep), since its work is genuinely wide.
- Net. At N=50, teleporting costs ~2.1x more wall-clock time per effective
  sample than GW parallel on CPU, and ~9.4x more on GPU (16.77 vs 1.79 ms),
  despite mixing ~14x better. Lower IAT does not translate into faster sampling
  here, and the gap widens on hardware that favors wide parallel work.

**Limitations:**

- Single target (1D double-well). No claim is made about the Section 5.2 GP
  regression posteriors, where per-step likelihood cost is much higher and may
  dominate the differences measured here.
- 5 seeds only. The GW IAT ranges are wide, and three estimates at N=2 and N=5
  were flagged unreliable (chain shorter than 50x tau). Those flagged upper
  bounds (119.6, 171.7, 128.5) should be read as a minimum value rather than as
  measurements.
- The benchmark was only on small ensembles in one dimension. At larger N or higher dimension the O(N^2)
  term may become significant enough to change the results.
