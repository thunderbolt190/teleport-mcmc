import pytest
import jax
import jax.numpy as jnp
from teleport.kernels.rwmh import rwmh_jax
from teleport.targets import log_prob_gaussian2d, GAUSSIAN2D_MEAN, GAUSSIAN2D_COV
jax.config.update("jax_enable_x64", True)

def test_rwmh():
  mean = GAUSSIAN2D_MEAN
  cov = GAUSSIAN2D_COV

  key = jax.random.PRNGKey(0)
  init = jnp.array([0.0, 0.0])
  n_steps = 50000
  step_size = 1.0
  burnin = 1000

  chain, accept_rate = rwmh_jax(log_prob_gaussian2d, init, n_steps, step_size, key)

  valid_samples = chain[burnin:]
  sample_mean = jnp.mean(valid_samples, axis = 0)
  sample_cov = jnp.cov(valid_samples, rowvar = False)

  assert chain.shape == (50000, 2)
  assert jnp.allclose(sample_mean, mean, atol = 0.1)
  assert jnp.allclose(sample_cov, cov, atol = 0.1)
  assert 0.2 < accept_rate < 0.6
