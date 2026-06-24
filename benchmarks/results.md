# Verification and Benchmark Results

## Stage 3 Baseline — RWMH JAX

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

## Stage 3 Baseline — Goodman-Weare JAX

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
