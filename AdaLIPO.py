import numpy as np
from statistical_analysis import LIPO_Statistics
from collections import deque
from utils import *

def Bernoulli(p: float):
    '''
    This function generates a random variable following a Bernoulli distribution.
    p: probability of success (float)
    '''
    a = np.random.uniform(0, 1)
    if a <= p:
        return 1
    else:
        return 0
        

def AdaLIPO(f, n: int, fig_path: str, delta=0.05, p=0.5):
  """
  f: class of the function to maximize (class)
  n: number of function evaluations (int)
  p: probability of success for exploration/exploitation (float)
  fig_path: path to save the statistics figures (str)
  delta: confidence level for bounds (float)
  max_slope: maximum slope for the nb_samples vs nb_evaluations curve (float)
  """
  
  # Initialization
  t = 1
  alpha = 10e-2
  k_hat = 0

  X_1 = Uniform(f.bounds)
  nb_samples = 1

  # We keep track of the last 3 values of nb_samples to compute the slope
  last_nb_samples = deque([1], maxlen=3)

  points = X_1.reshape(1, -1)
  values = np.array([f(X_1)])

  # Statistics
  stats = LIPO_Statistics(f, fig_path, delta=delta)

  def k(i):
    """
    Series of potential Lipschitz constants.
    """
    return (1 + alpha)**i

  def condition(x, values, k, points):
    """
    Subfunction to check the condition in the loop, depending on the set of values we already have.
    values: set of values of the function we explored (numpy array)
    x: point to check (numpy array)
    k: Lipschitz constant (float)
    points: set of points we have explored (numpy array)
    """
    max_val = np.max(values)

    left_min = np.min(values.reshape(-1) + k * np.linalg.norm(x - points, ord=2, axis=1))

    return left_min >= max_val
          
  # Main loop
  ratios = []
  while t < n:
    B_tp1 = Bernoulli(p)
    if B_tp1 == 1:
      # Exploration
      X_tp1 = Uniform(f.bounds)
      nb_samples += 1
      last_nb_samples[-1] = nb_samples
      points = np.concatenate((points, X_tp1.reshape(1, -1)))
      value = f(X_tp1)
    else:
      # Exploitation
      while True:
        X_tp1 = Uniform(f.bounds)
        nb_samples += 1
        last_nb_samples[-1] = nb_samples
        if condition(X_tp1, values, k_hat, points):
          points = np.concatenate((points, X_tp1.reshape(1, -1)))
          break
      value = f(X_tp1)
      
    values = np.concatenate((values, np.array([value])))
    for i in range(points.shape[0]-1):
      ratios.append(np.abs(value - values[i]) / np.linalg.norm(X_tp1 - points[i], ord=2))

    i_hat = int(np.ceil(np.log(max(ratios)) / np.log(1 + alpha)))
    k_hat = k(i_hat)
    
    # Statistical analysis
    stats.update(np.max(values), nb_samples, k_hat=k_hat)

    t += 1
    last_nb_samples.append(0)

    if nb_samples >= 500*n:
      ValueError("AdaLIPO has likely explored every possible \
        region in which the maximum can be, but did not finish \
        the main loop. Please reduce the number of function evaluations.")
  
    if t % 200 == 0:
      print(f"Iteration: {t} Lipschitz constant: {k_hat:.4f} Number of samples: {nb_samples}")


  stats.plot()

  # Output
  return points, values, t
