import pytest
import jax
import jax.numpy as jnp
from teleport.kernels.goodman_weare import goodman_weare_jax
from teleport.targets import log_prob_gaussian2d, GAUSSIAN2D_MEAN, GAUSSIAN2D_COV
jax.config.update("jax_enable_x64", True)

def test_goodman_weare():
  mean = GAUSSIAN2D_MEAN
  cov = GAUSSIAN2D_COV

  key = jax.random.PRNGKey(0)
  key1, key2 = jax.random.split(key)

  n_walkers = 50
  n_steps = 2000
  burnin = 200
  dim = 2

  init_walkers = jax.random.normal(key1, shape = (n_walkers, dim))

  chain, accept_rate = goodman_weare_jax(log_prob_gaussian2d, init_walkers, n_steps, key2, n_walkers)

  valid_samples = chain[burnin:].reshape(-1, dim)
  sample_mean = jnp.mean(valid_samples, axis = 0)
  sample_cov = jnp.cov(valid_samples, rowvar = False)

  assert chain.shape == (2000, 50, 2)
  assert jnp.allclose(sample_mean, mean, atol = 0.1)
  assert jnp.allclose(sample_cov, cov, atol = 0.1)
  assert 0.4 < accept_rate < 0.8
