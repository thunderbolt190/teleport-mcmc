import jax.numpy as jnp

def log_prob_doublewell(x):
    return jnp.sum(-4.0 * (x**4 - x**2))

GAUSSIAN2D_MEAN = jnp.array([2.0, -1.0])
GAUSSIAN2D_COV = jnp.array([[1.0, 0.8], [0.8, 1.0]])
GAUSSIAN2D_COVINV = jnp.linalg.inv(GAUSSIAN2D_COV)

def log_prob_gaussian2d(x, mean = GAUSSIAN2D_MEAN, covinv = GAUSSIAN2D_COVINV):
    diff = x - mean
    return -0.5 * diff.T @ covinv @ diff
