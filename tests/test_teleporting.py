import pytest
import jax 
import jax.numpy as jnp
from teleport.kernels.teleporting import one_teleporting_step
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
