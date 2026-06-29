import pytest
import jax 
import jax.numpy as jnp
from teleport.utils.weights import compute_log_weights
jax.config.update("jax_enable_x64", True)

def test_weights_function():
  walkers = jnp.array([[0.0], [3.0]])
  log_probs = jnp.array([-0.5, -2.0])
  z = jnp.array([1.0])
  sigma = 1.0
  log_w = compute_log_weights(walkers, log_probs, z, sigma)

  actual = jnp.array([0.01815, 0.07889])

  assert jnp.allclose(log_w, actual, atol=1e-4)
