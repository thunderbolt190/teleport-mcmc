import pytest
import jax 
import jax.numpy as jnp
from teleport.kernels.teleporting import one_teleporting_step
from teleport.kernels.teleporting import teleporting_walkers_jax
jax.config.update("jax_enable_x64", True)

def test_one_teleporting_step_function():
  walkers = jnp.array([[0.0], [3.0]])
  log_probs = jnp.array([-0.5, -2.0])
  z = jnp.array([1.0])
  sigma = 1.0

  def log_prob_fn(x):
    return -0.5 * jnp.sum(x**2)

  key = jax.random.PRNGKey(42)

  walkers, log_probs, key, accepted, teleported = one_teleporting_step(walkers, log_probs, log_prob_fn, sigma, key)

  assert walkers.shape == (2, 1)
  assert log_probs.shape == (2,)
  assert accepted.dtype == jnp.bool_
  assert teleported.dtype == jnp.bool_


def test_teleporting_walkers_jax_function():
  walkers = jnp.array([[0.0], [3.0]])
  def log_prob_fn(x):
    return -0.5 * jnp.sum(x**2)
  step_size = 1.0
  n_steps = 5
  key = jax.random.PRNGKey(42)

  walkers, chain, accepts, teleports = teleporting_walkers_jax(walkers, log_prob_fn, step_size, n_steps, key)

  assert walkers.shape == (2, 1)
  assert chain.shape == (5, 2, 1)
  assert accepts.shape == (5,)
  assert teleports.shape == (5,)
  assert not jnp.any(jnp.isnan(chain))
  assert not jnp.any(jnp.isnan(walkers))


def test_2d_gaussian_teleporting_walkers():
  mean = jnp.array([2.0, -1.0])
  cov = jnp.array([[1.0, 0.8], [0.8, 1.0]])
  covinv = jnp.linalg.inv(cov)

  def log_prob_jax(x):
    diff = x - mean
    return -0.5 * diff.T @ covinv @ diff

  key = jax.random.PRNGKey(42)
  key1, key2 = jax.random.split(key)

  walkers = jax.random.normal(key1, shape=(50, 2))
  step_size = 0.5
  n_steps = 5000
  burnin = 500
  dim = 2

  final_walkers, chain, accepts, teleports = teleporting_walkers_jax(walkers, log_prob_jax, step_size, n_steps, key2)

  valid_samples = chain[burnin:].reshape(-1, 2)
  sample_mean = jnp.mean(valid_samples, axis = 0)
  sample_cov = jnp.cov(valid_samples, rowvar = False)

  assert jnp.allclose(sample_mean, mean, atol = 0.1)
  assert jnp.allclose(sample_cov, cov, atol = 0.1)


def test_double_well_teleporting_walkers():
  def log_prob_doublewell(x):
    return -4.0 * (x**4 - x**2)

  key = jax.random.PRNGKey(42)
  key, subkey = jax.random.split(key)
  walkers = -0.707 + 0.1 * jax.random.normal(subkey, shape=(20, 1))

  n_steps = 5000
  step_size = 0.5

  final_walkers, chain, accepts, teleports = teleporting_walkers_jax(walkers, log_prob_doublewell, step_size, n_steps, key)
  valid_samples = chain[500:].reshape(-1, 1)
  right_mode = jnp.mean(valid_samples > 0)
  assert right_mode > 0.05 
  assert right_mode < 0.95 
