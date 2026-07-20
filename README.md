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
