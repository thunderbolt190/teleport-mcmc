import jax
import jax.numpy as jnp
from functools import partial
jax.config.update("jax_enable_x64", True)

@jax.jit
def proposal_step_jax(current, step_size, key):
  noise = step_size * jax.random.normal(key, shape = current.shape)
  return current + noise

@jax.jit
def accept_jax(current_log_prob, proposal_log_prob, key):
  log_a = proposal_log_prob - current_log_prob
  log_u = jnp.log(jax.random.uniform(key))
  return log_u < log_a

@partial(jax.jit, static_argnums = (0, 2))
def rwmh_jax(log_prob_fn, init, n_steps, step_size, key):
  def rwmh_step(carry, _):
    current, curr_log_prob, key, acc_counter = carry
    key, key1, key2 = jax.random.split(key, 3)
    proposal = proposal_step_jax(current, step_size, key1)
    prop_log_prob = log_prob_fn(proposal)
    accepted = accept_jax(curr_log_prob, prop_log_prob, key2)
    current = jnp.where(accepted, proposal, current)
    curr_log_prob = jnp.where(accepted, prop_log_prob, curr_log_prob)
    acc_counter = jnp.where(accepted, acc_counter + 1, acc_counter)
    new_carry = (current, curr_log_prob, key, acc_counter)
    return new_carry, current
  init_carry = (init, log_prob_fn(init), key, 0)
  (final_pos, final_lp, _, n_accepted), chain = jax.lax.scan(rwmh_step, init_carry, xs = None, length = n_steps)
  return chain, n_accepted / n_steps

