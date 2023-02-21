import numpy as np
from utils import *

def Uniform(X: np.array):
    '''
    This function generates a random point in the feasible region X. We assume that X is a subset of R^n 
    described by the inequalities X = {x in R^n | a_i <= x_i <= b_i, i = 0, ..., m-1} where a_i, b_i are given
    such that X[i,j] = [a_i, b_i] for i = 0, ..., m-1 and j = 0, 1.
    For simplicity, we assume that X C Rectangle given by an infinite norm (i.e. X = {x in R^n | -M <= x_i <= M, i = 1, ..., n}).
    X: feasible region (numpy array)
    '''

    theta = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
      theta[i] = np.random.uniform(X[i,0], X[i,1])
    return theta
        
    

def random_search(f, n: int):
    '''
    f: class of the function to maximize (class)
    n: number of function evaluations (int)
    '''
    
    values = []
    points = []

    # Initialization
    x = Uniform(f.bounds)
    val = f(x)
    values.append(val)
    points.append(x)

    t = 1
    while t < n:
      x = Uniform(f.bounds)
      val = f(x)
      values.append(val)
      points.append(x)
      t += 1

    return np.array(points), np.array(values), t