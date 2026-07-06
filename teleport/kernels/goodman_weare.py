import jax
import jax.numpy as jnp
from functools import partial
jax.config.update("jax_enable_x64", True)

@partial(jax.jit, static_argnums = (1,))
def sample_stretch_factor_jax(key, n_samples, a = 2.0):
  u = jax.random.uniform(key, shape = (n_samples,))
  return (u * (jnp.sqrt(a) - 1/jnp.sqrt(a)) + 1/jnp.sqrt(a)) ** 2

@jax.jit
def gw_proposal_jax(walker_i, walker_j, z):
  return walker_j + z * (walker_i - walker_j)

@jax.jit
def gw_accept_jax(current_log_prob, proposal_log_prob, z, dim, key):
  log_a = (dim - 1) * jnp.log(z) + proposal_log_prob - current_log_prob
  log_u = jnp.log(jax.random.uniform(key))
  return log_u < log_a

@jax.jit
def select_complementary_jax(n_walkers, i, key):
  j = jax.random.randint(key, shape=(), minval=0, maxval=n_walkers - 1)
  return jnp.where(j >= i, j + 1, j)

@partial(jax.jit, static_argnums = (0, 2, 4))
def goodman_weare_jax(log_prob_fn, init_walkers, n_steps, key, n_walkers, a = 2.0):
  dim = int(init_walkers.shape[1])
  def update_one_walker(i, walkers, log_probs, key, a):
    key1, key2, key3 =jax.random.split(key, 3)
    comp_walker = select_complementary_jax(n_walkers, i, key1)
    z = sample_stretch_factor_jax(key2, 1, a)[0]
    proposal = gw_proposal_jax(walkers[i], walkers[comp_walker], z)
    prop_log_prob = log_prob_fn(proposal)
    accepted = gw_accept_jax(log_probs[i], prop_log_prob, z, dim, key3)
    new_pos = jnp.where(accepted, proposal, walkers[i])
    new_lp = jnp.where(accepted, prop_log_prob, log_probs[i])
    acc = jnp.where(accepted, 1, 0)
    return new_pos, new_lp, acc
  update_walkers = jax.vmap(update_one_walker, in_axes = (0, None, None, 0, None), out_axes = (0, 0, 0))
  def gw_step(carry, _):
    walkers, log_probs, key, acc_counter = carry
    new_key, *subkeys = jax.random.split(key, n_walkers + 1)
    subkeys = jnp.array(subkeys)
    new_walkers, new_log_probs, new_acc = update_walkers(jnp.arange(n_walkers), walkers, log_probs, subkeys, a)
    acc_counter = acc_counter + jnp.sum(new_acc)
    new_carry = (new_walkers, new_log_probs, new_key, acc_counter)
    return new_carry, new_walkers
  log_probs = jax.vmap(log_prob_fn)(init_walkers)
  init_carry = (init_walkers, log_probs, key, 0)
  (final_pos, final_lp, _, n_accepted), chain = jax.lax.scan(gw_step, init_carry, xs = None, length = n_steps)
  return chain, n_accepted / (n_walkers * n_steps)
