import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

@jax.jit
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

@jax.jit
def pairwise_log_q(walkers, sigma):
    N = walkers.shape[0]
    diff = walkers[:, None, :] - walkers[None, :, :]
    M = -0.5 * jnp.sum(diff ** 2, axis=-1) / sigma ** 2
    M = M.at[jnp.arange(N), jnp.arange(N)].set(-jnp.inf)
    L = jax.scipy.special.logsumexp(M, axis=1)
    return M, L

@jax.jit
def compute_log_weights_cached(walkers, log_probs, z, sigma, L):
    diff_z = walkers - z[None, :]
    log_q_z = -0.5 * jnp.sum(diff_z ** 2, axis=-1) / sigma ** 2
    log_w = jnp.logaddexp(log_q_z, L) - log_probs
    return log_w

@jax.jit
def update_weight_cache(M, L, walkers, i, z, sigma):
    M_old_col = M[:, i]
    diff_z = walkers - z[None, :]
    log_q_z = -0.5 * jnp.sum(diff_z ** 2, axis=-1) / sigma ** 2
    new_col = log_q_z.at[i].set(-jnp.inf)

    M = M.at[i, :].set(new_col)
    M = M.at[:, i].set(new_col)

    L_without = jnp.where(M_old_col == -jnp.inf, L, L + jnp.log1p(-jnp.exp(-(L - M_old_col))))
    L_inc = jnp.logaddexp(L_without, new_col)
    unstable = M_old_col > (L - 1e-8)
    L_rebuild = jax.scipy.special.logsumexp(M, axis=1)
    L = jnp.where(unstable, L_rebuild, L_inc)
    L = L.at[i].set(jax.scipy.special.logsumexp(new_col))
    return M, L
