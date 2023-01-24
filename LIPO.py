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
        

def LIPO(f, n: int):
  """
  f: class of the function to maximize (class)
  n: number of function evaluations (int)
  """
  
  # Initialization
  t = 1
  nb_samples = 0
  X_1 = Uniform(f.bounds)
  nb_samples += 1
  points = X_1.reshape(1, -1)
  value = f(X_1)
  values = np.array([value])

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
    if condition(X_tp1, values, f.k, points):
      points = np.concatenate((points, X_tp1.reshape(1, -1)))

      value = f(X_tp1)
      values = np.concatenate((values, np.array([value])))
      t += 1
          
  # Output
  return points, values, nb_samples