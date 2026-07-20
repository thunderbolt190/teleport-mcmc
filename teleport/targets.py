import jax.numpy as jnp

def log_prob_doublewell(x):
    return jnp.sum(-4.0 * (x**4 - x**2))

gaussian2d_mean = jnp.array([2.0, -1.0])
gaussian2d_cov = jnp.array([[1.0, 0.8], [0.8, 1.0]])
gaussian2d_covinv = jnp.linalg.inv(gaussian2d_cov)

def log_prob_gaussian2d(x, mean=gaussian2d_mean, covinv=gaussian2d_covinv):
    diff = x - mean
    return -0.5 * diff.T @ covinv @ diff
