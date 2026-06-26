from scipy.special import logsumexp
import numpy as np

def compute_log_weights(walkers, log_probs, z, sigma):
  N = walkers.shape[0]
  diff = walkers[:, None, :] - walkers[None, :, :]
  M = -0.5 * np.sum(diff ** 2, axis = -1) / sigma ** 2
  M[np.arange(N), np.arange(N)] = -np.inf
  diff_z = walkers - z[None, :]
  log_q_z = -0.5 * np.sum(diff_z ** 2, axis = -1) / sigma ** 2
  all_log_q = np.concatenate([log_q_z[:, None], M], axis = 1)
  log_w = logsumexp(all_log_q, axis = 1) - log_probs
  return log_w
