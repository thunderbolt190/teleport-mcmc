## One Step Pseudocode

```python
def one_step(walkers, log_probs, log_prob_fn, key, step_size):
    N = walkers.shape[0]
    dim = walkers.shape[1]
    key, subkey0, subkey1, subkey2, subkey3 = jax.random.split(key, 5)
    j = jax.random.randint(subkey0, shape=(), minval=0, maxval=N)
    z = walkers[j] + step_size * jax.random.normal(subkey1, shape=(dim,))

    diff = walkers[:, None, :] - walkers[None, :, :]  # shape (N, N, dim)
    M = -0.5 * jnp.sum(diff**2, axis=-1) / step_size**2  # shape (N, N)
    M = M.at[jnp.arange(N), jnp.arange(N)].set(-jnp.inf)
    diff_z = walkers - z[None, :]  # shape (N, dim)
    log_q_z = -0.5 * jnp.sum(diff_z**2, axis=-1) / step_size**2  # shape (N,)
    all_log_q = jnp.concatenate([log_q_z[:, None], M], axis=1)  # shape (N, N+1)
    log_w = jax.scipy.special.logsumexp(all_log_q, axis=1) - log_probs  # shape (N,)
    log_Z_xz = jax.scipy.special.logsumexp(log_w)

    i = jax.random.categorical(subkey2, log_w)
    delete = walkers[i]

    poss_walkers = walkers.at[i].set(z)
    poss_log_probs = log_probs.at[i].set(log_prob_fn(z))

    diff2 = poss_walkers[:, None, :] - poss_walkers[None, :, :]  # shape (N, N, dim)
    M2 = -0.5 * jnp.sum(diff2**2, axis=-1) / step_size**2  # shape (N, N)
    M2 = M2.at[jnp.arange(N), jnp.arange(N)].set(-jnp.inf)
    diff3 = poss_walkers - delete[None, :]  # shape (N, dim)
    log2 = -0.5 * jnp.sum(diff3**2, axis=-1) / step_size**2  # shape (N,)
    all_log2 = jnp.concatenate([log2[:, None], M2], axis=1)  # shape (N, N+1)
    log_w2 = jax.scipy.special.logsumexp(all_log2, axis=1) - poss_log_probs  # shape (N,)
    log_Z_x_prime_xi = jax.scipy.special.logsumexp(log_w2)

    log_accept = log_Z_xz - log_Z_x_prime_xi
    accepted = jnp.log(jax.random.uniform(subkey3)) < log_accept
    walkers = jnp.where(accepted, poss_walkers, walkers)
    log_probs = jnp.where(accepted, poss_log_probs, log_probs)
    teleported = (i != j)
    return walkers, log_probs, key, accepted, teleported
``` 
