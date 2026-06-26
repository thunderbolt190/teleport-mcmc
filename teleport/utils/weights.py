import pytest
import numpy as np 
from teleport.utils.weights import compute_log_weights

def test_weights_function():
  walkers = np.array([[0.0], [3.0]])
  log_probs = np.array([-0.5, -2.0])
  z = np.array([1.0])
  sigma = 1.0
  log_w = compute_log_weights(walkers, log_probs, z, sigma)

  actual = [0.01815, 0.07889]

  assert np.allclose(log_w, actual, atol=1e-4)
