import jax
import jax.numpy as jnp
from teleport.utils.weights import compute_log_weights
from functools import partial
jax.config.update("jax_enable_x64", True)

@partial(jax.jit, static_argnums=(2,))
def one_teleporting_step(walkers, log_probs, log_prob_fn, step_size, key):
  N = walkers.shape[0]
  dim = walkers.shape[1]
  key, subkey0, subkey1, subkey2, subkey3 = jax.random.split(key, 5)
  j = jax.random.randint(subkey0, shape=(), minval=0, maxval=N)
  z = jnp.take(walkers, j, axis=0).squeeze()
  z = jnp.atleast_1d(z) + step_size * jax.random.normal(subkey1, shape=(dim,))
  
  log_w = compute_log_weights(walkers, log_probs, z, step_size)
  log_Z = jax.scipy.special.logsumexp(log_w)

  i = jax.random.categorical(subkey2, log_w)

  poss_walkers = walkers.at[i].set(z)
  poss_log_probs = log_probs.at[i].set(log_prob_fn(z))

  delete = jnp.take(walkers, i, axis=0).squeeze()
  delete = jnp.atleast_1d(delete)
  log_Z_prime = jax.scipy.special.logsumexp(compute_log_weights(poss_walkers, poss_log_probs, delete, step_size))
  teleported = (i != j)

  log_accept = log_Z - log_Z_prime
  accepted = jnp.log(jax.random.uniform(subkey3)) < log_accept
  walkers = jnp.where(accepted, poss_walkers, walkers)
  log_probs = jnp.where(accepted, poss_log_probs, log_probs)

  return walkers, log_probs, key, accepted, teleported


@partial(jax.jit, static_argnums=(1, 3))
def teleporting_walkers_jax(init_walkers, log_prob_fn, step_size, n_steps, key):
  def step_func(carry, _):
    walkers, log_probs, key = carry
    walkers, log_probs, key, accepted, teleported = one_teleporting_step(walkers, log_probs, log_prob_fn, step_size, key)
    carry = (walkers, log_probs, key)
    return carry, (walkers, accepted, teleported)
  
  init_log_probs = jax.vmap(log_prob_fn)(init_walkers)
  init_carry = (init_walkers, init_log_probs, key)

  (final_walkers, final_lp, _), (chain, accepts, teleports) = jax.lax.scan(step_func, init_carry, xs = None, length = n_steps)
  return final_walkers, chain, accepts, teleports
