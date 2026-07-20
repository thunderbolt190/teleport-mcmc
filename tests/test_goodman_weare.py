import pytest
import jax
import jax.numpy as jnp
from teleport.kernels.goodman_weare import goodman_weare_jax
jax.config.update("jax_enable_x64", True)

def test_goodman_weare():
  mean = jnp.array([2.0, -1.0])
  cov = jnp.array([[1.0, 0.8], [0.8, 1.0]])
  covinv = jnp.linalg.inv(cov)

  def log_prob_jax(x):
    diff = x - mean
    return -0.5 * diff.T @ covinv @ diff

  key = jax.random.PRNGKey(0)
  key1, key2 = jax.random.split(key)

  n_walkers = 50
  n_steps = 2000
  burnin = 200
  dim = 2

  init_walkers = jax.random.normal(key1, shape = (n_walkers, dim))

  chain, accept_rate = goodman_weare_jax(log_prob_jax, init_walkers, n_steps, key2, n_walkers)

  valid_samples = chain[burnin:].reshape(-1, dim)
  sample_mean = jnp.mean(valid_samples, axis = 0)
  sample_cov = jnp.cov(valid_samples, rowvar = False)

  assert chain.shape == (2000, 50, 2)
  assert jnp.allclose(sample_mean, mean, atol = 0.1)
  assert jnp.allclose(sample_cov, cov, atol = 0.1)
  assert 0.4 < accept_rate < 0.8
