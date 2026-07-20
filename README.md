# teleport-mcmc
GPU-accelerated implementation of Ensemble Markov Chain Monte Carlo with Teleporting Walkers, written in JAX.

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

## Installation

```bash
git clone https://github.com/thunderbolt190/teleport-mcmc.git
cd teleport-mcmc
pip install -e ".[dev,benchmarks]"
```

`dev` includes testing tools (pytest); `benchmarks` includes emcee,
matplotlib, and numpy, needed to run `notebooks/03_iat_vs_doublewell.ipynb`.

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
  samplers implemented and tested (7 passing tests, see `tests/`)
- Baseline correctness verified against known 2D Gaussian statistics;
  see [`benchmarks/results.md`](benchmarks/results.md)
- IAT-vs-N efficiency check on the double-well target (Table 1 style, not a paper reproduction); see
  [`notebooks/03_iat_vs_doublewell.ipynb`](notebooks/03_iat_vs_doublewell.ipynb)

Planned work (paper reproductions, additional benchmarks) is tracked in
[GitHub Issues](https://github.com/thunderbolt190/teleport-mcmc/issues).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
