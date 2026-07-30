import jax

@jax.jit
def proposal_step_jax(current, step_size, key):
  noise = step_size * jax.random.normal(key, shape = current.shape)
  return current + noise
