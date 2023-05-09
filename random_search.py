import numpy as np


def Uniform(X: np.ndarray):
    """
    This function generates a random point in the feasible region X. We assume that X is a subset of R^n
    described by the inequalities X = {x in R^n | a_i <= x_i <= b_i, i = 0, ..., m-1} where a_i, b_i are given
    such that X[i,j] = [a_i, b_i] for i = 0, ..., m-1 and j = 0, 1.
    For simplicity, we assume that X C Rectangle given by an infinite norm (i.e. X = {x in R^n | -M <= x_i <= M, i = 1, ..., n}).
    X: feasible region (numpy array)
    """

    theta = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        theta[i] = np.random.uniform(X[i, 0], X[i, 1])
    return theta


def random_search(f, X: np.ndarray, n: int):
    """
    f: class of the function to maximize (class)
    X: bounds of the parameters (numpy array)
    n: number of function evaluations (int)
    """

    values = np.zeros(n)
    points = np.zeros((n, X.shape[0]))

    for i in range(n):
        x = Uniform(X)
        val = f(x)
        values[i] = val
        points[i] = x

    return points, values, len(values)
