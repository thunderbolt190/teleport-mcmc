import jax
import jax.numpy as jnp
from teleport.utils.weights import compute_log_weights
jax.config.update("jax_enable_x64", True)

def one_teleporting_step(walkers, log_probs, log_prob_fn, step_size, key):
  N = walkers.shape[0]
  dim = walkers.shape[1]
  key, subkey0, subkey1, subkey2, subkey3 = jax.random.split(key, 5)
  j = jax.random.randint(subkey0, shape=(), minval=0, maxval=N)
  z = walkers[j] + step_size * jax.random.normal(subkey1, shape=(dim,))
  
  log_w = compute_log_weights(walkers, log_probs, z, step_size)
  log_Z = jax.scipy.special.logsumexp(log_w)

  i = jax.random.categorical(subkey2, log_w)

  poss_walkers = walkers.at[i].set(z)
  poss_log_probs = log_probs.at[i].set(log_prob_fn(z))

  log_Z_prime = jax.scipy.special.logsumexp(compute_log_weights(poss_walkers, poss_log_probs, walkers[i], step_size))
  teleported = (i != j)

  log_accept = log_Z - log_Z_prime
  accepted = jnp.log(jax.random.uniform(subkey3)) < log_accept
  walkers = jnp.where(accepted, poss_walkers, walkers)
  log_probs = jnp.where(accepted, poss_log_probs, log_probs)

  return walkers, log_probs, key, accepted, teleported
