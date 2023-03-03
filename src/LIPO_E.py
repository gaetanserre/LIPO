"""
Copyright (c) 2023 Perceval Beja-Battais, Gaëtan Serré and Sophia Chirrane

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <http://www.gnu.org/licenses/>. 
"""

import numpy as np
from statistical_analysis import LIPO_Statistics
from collections import deque
from utils import *
        

"""
This function implements the LIPO-E algorithm
(in the paper, Algorithm 3 with the stopping criterion).
"""
def LIPO_E(f, n: int, fig_path: str, delta=0.05, size_slope=5, max_slope=600.0):
  """
  f: class of the function to maximize (class)
  n: number of function evaluations (int)
  fig_path: path to save the statistics figures (str)
  size_slope: size of the window to compute the slope of the nb_samples vs nb_evaluations curve (int)
  max_slope: maximum slope for the nb_samples vs nb_evaluations curve (float)
  """
  
  # Initialization
  t = 1

  X_1 = Uniform(f.bounds)
  nb_samples = 1

  # We keep track of the last `size_slope` values of nb_samples to compute the slope
  last_nb_samples = deque([1], maxlen=size_slope)

  points = X_1.reshape(1, -1)
  values = np.array([f(X_1)])

  # Statistics
  stats = LIPO_Statistics(f, fig_path, delta=delta)

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
  while t < n:
    X_tp1 = Uniform(f.bounds)
    nb_samples += 1
    last_nb_samples[-1] = nb_samples
    if condition(X_tp1, values, f.k, points):
      points = np.concatenate((points, X_tp1.reshape(1, -1)))

      values = np.concatenate((values, np.array([f(X_tp1)])))
      
      # Statistical analysis
      stats.update(np.max(values), nb_samples)

      t += 1
      last_nb_samples.append(0)
    
    elif slope_stop_condition(last_nb_samples, size_slope, max_slope):
      print(f"Exponential growth of the number of samples. Stopping the algorithm at iteration {t}.")
      stats.plot()
      return points, values, t

  stats.plot()
          
  # Output
  return points, values, t
