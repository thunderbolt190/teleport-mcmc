import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums = (1,))
def sample_stretch_factor_jax(key, n_samples, a = 2.0):
  u = jax.random.uniform(key, shape = (n_samples,))
  return (u * (jnp.sqrt(a) - 1/jnp.sqrt(a)) + 1/jnp.sqrt(a)) ** 2

@jax.jit
def gw_proposal_jax(walker_i, walker_j, z):
  return walker_j + z * (walker_i - walker_j)

