import jax.numpy as jnp

def log_prob_doublewell(x):
    return jnp.sum(-4.0 * (x**4 - x**2))
