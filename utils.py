import numpy as np

def Uniform(X: np.array):
  """
  This function generates a random point in the feasible region X. We assume that X is a subset of R^n 
  described by the inequalities X = {x in R^n | a_i <= x_i <= b_i, i = 0, ..., m-1} where a_i, b_i are given
  such that X[i,j] = [a_i, b_i] for i = 0, ..., m-1 and j = 0, 1.
  For simplicity, we assume that X C Rectangle given by an infinite norm (i.e. X = {x in R^n | -M <= x_i <= M, i = 1, ..., n}).
  X: feasible region (numpy array)
  """

  theta = np.zeros(X.shape[0])
  for i in range(X.shape[0]):
    theta[i] = np.random.uniform(X[i,0], X[i,1])
  return theta

def slope_stop_condition(last_nb_samples, size_slope, max_slope):
    """
    Check if the slope of the last `size_slope` points of the the nb_samples vs nb_evaluations curve 
    is greater than max_slope.
    """
    if len(last_nb_samples) == size_slope:
      slope = (last_nb_samples[-1] - last_nb_samples[0]) / (len(last_nb_samples) - 1)
      return slope > max_slope
    else:
      return False