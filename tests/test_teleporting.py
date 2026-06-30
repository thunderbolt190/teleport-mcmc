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
