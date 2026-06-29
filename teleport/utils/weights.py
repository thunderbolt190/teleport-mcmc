import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

def compute_log_weights(walkers, log_probs, z, sigma):
  N = walkers.shape[0]
  diff = walkers[:, None, :] - walkers[None, :, :]
  M = -0.5 * jnp.sum(diff ** 2, axis = -1) / sigma ** 2
  M = M.at[jnp.arange(N), jnp.arange(N)].set(-jnp.inf)
  diff_z = walkers - z[None, :]
  log_q_z = -0.5 * jnp.sum(diff_z ** 2, axis = -1) / sigma ** 2
  all_log_q = jnp.concatenate([log_q_z[:, None], M], axis = 1)
  log_w = jax.scipy.special.logsumexp(all_log_q, axis = 1) - log_probs
  return log_w
