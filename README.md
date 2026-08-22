# teleport-mcmc
JAX implementation of Ensemble Markov Chain Monte Carlo with Teleporting Walkers.

## About

This implements the teleporting walkers algorithm, an ensemble MCMC
method where walkers can occasionally jump directly to another point in
parameter space, rather than only moving locally. A jump is accepted
based on how redundant the walker's current position is relative to
the rest of the ensemble, favoring jumps away from crowded regions.
This helps the ensemble mix across separated modes that standard
ensemble samplers can get stuck between.

Lindsey, M., Weare, J., & Zhang, A. (2022). "Ensemble Markov Chain Monte
Carlo with Teleporting Walkers." *SIAM/ASA Journal on Uncertainty
Quantification*, 10(3), 860–885.
[arXiv:2106.02686](https://arxiv.org/abs/2106.02686)

## Results

- Reproduces the paper's Section 5.2 Gaussian process regression
  experiments (univariate, multivariate, and Student-t noise with
  restricted interaction). The trends match the paper qualitatively but the
  numbers are not a direct reproduction, since the synthetic data and
  seeds differ.
- On a bimodal double-well target, teleporting reaches roughly 14x lower
  integrated autocorrelation time (IAT) than Goodman-Weare and does so
  more consistently across seeds.
- Lower IAT does not always mean faster sampling.
  Teleporting costs about 2x more wall-clock time per effective sample
  than Goodman-Weare on CPU and about 9x more on GPU, despite its large
  mixing advantage. The cause of this appears to be sequential depth rather than
  arithmetic cost. One teleporting step moves a single walker, so
  covering the same number of sweeps requires N times more dependent
  scan iterations than Goodman-Weare, which updates all N walkers at
  once. This experiment is not in the paper.

Full numbers, methodology, and caveats:
[`benchmarks/results.md`](benchmarks/results.md).

## Installation

```bash
git clone https://github.com/thunderbolt190/teleport-mcmc.git
cd teleport-mcmc
pip install -e ".[dev,benchmarks]"
```

`dev` includes testing tools (pytest); `benchmarks` includes emcee,
matplotlib, and numpy, needed to run the experiment notebooks under
`notebooks/` (double-well studies and paper Examples 5.2.1–5.2.3) and
the scripts under `benchmarks/`.

## Usage

```python
import jax
from teleport.kernels.teleporting import teleporting_walkers_jax
from teleport.targets import log_prob_gaussian2d

key = jax.random.PRNGKey(0)
walkers = jax.random.normal(key, shape=(20, 2))

final_walkers, chain, accepts, teleports = teleporting_walkers_jax(
    walkers, log_prob_gaussian2d, step_size=0.5, n_steps=5000, key=key
)
# final_walkers: (N, dim) final ensemble state
# chain:         (n_steps, N, dim) full trajectory
# accepts:       (n_steps,) bool, whether each step's proposal was accepted
# teleports:     (n_steps,) bool, whether each accepted step was a teleport move
```


## Status

- Core teleporting walkers algorithm, RWMH, and Goodman-Weare baseline
  samplers implemented and tested (8 passing tests, see
  [`tests/`](tests/))
- Baseline correctness (2D Gaussian) and experiment write-ups:
  see [`benchmarks/results.md`](benchmarks/results.md)
- Double-well experiments:
  - IAT vs N - [`notebooks/03_iat_vs_doublewell.ipynb`](notebooks/03_iat_vs_doublewell.ipynb)
  - Teleporting vs Goodman-Weare - [`notebooks/04_teleporting_vs_gw.ipynb`](notebooks/04_teleporting_vs_gw.ipynb)
  - Acceptance & teleport probability vs N - [`notebooks/05_accept_teleport_vs_n_doublewell.ipynb`](notebooks/05_accept_teleport_vs_n_doublewell.ipynb)
- Paper example notebooks (Section 5.2):
  - 5.2.1 Univariate GP - [`notebooks/Example_5_2_1.ipynb`](notebooks/Example_5_2_1.ipynb)
  - 5.2.2 Multivariate GP - [`notebooks/Example_5_2_2.ipynb`](notebooks/Example_5_2_2.ipynb)
  - 5.2.3 Non-Gaussian (Student-t) noise / restricted interaction -
    [`notebooks/Example_5_2_3.ipynb`](notebooks/Example_5_2_3.ipynb)
- Wall-clock cost vs sampling efficiency (CPU and GPU) -
    [`benchmarks/benchmark_runtime.py`](benchmarks/benchmark_runtime.py)

Further work is tracked in
[GitHub Issues](https://github.com/thunderbolt190/teleport-mcmc/issues).


## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
