import numpy as np
from collections import deque


def Uniform(X: np.ndarray):
    """
    This function generates a random point in the feasible region X. We assume that X is a subset of R^n
    described by the inequalities X = {x in R^n | a_i <= x_i <= b_i, i = 0, ..., m-1} where a_i, b_i are given
    such that X[i,j] = [a_i, b_i] for i = 0, ..., m-1 and j = 0, 1.
    For simplicity, we assume that X C Rectangle given by an infinite norm (i.e. X = {x in R^n | -M <= x_i <= M, i = 1, ..., n}).
    X: bounds of the parameters (np.ndarray)
    """

    theta = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        theta[i] = np.random.uniform(X[i, 0], X[i, 1])
    return theta


def LIPO_P(f, X: np.ndarray, k: float, n: int, size_slope=5, max_slope=1000.0):
    """
    f: function to maximize (lambda function)
    X: bounds of the parameters (np.ndarray)
    k: Lipschitz constant (float)
    n: number of function evaluations (int)
    size_slope: size of the window to compute the slope of the nb_samples vs nb_evaluations curve (int)
    max_slope: maximum slope for the nb_samples vs nb_evaluations curve (float)
    """

    # Initialization
    t = 1

    X_1 = Uniform(X)
    nb_samples = 1

    # We keep track of the last `size_slope` values of nb_samples to compute the slope
    last_nb_samples = deque([1], maxlen=size_slope)

    points = X_1.reshape(1, -1)
    values = np.array([f(X_1)])

    def slope_stop_condition():
        """
        Check if the slope of the last `size_slope` points of the the nb_samples vs nb_evaluations curve
        is greater than max_slope.
        """
        if len(last_nb_samples) == size_slope:
            slope = (last_nb_samples[-1] - last_nb_samples[0]) / (
                len(last_nb_samples) - 1
            )
            return slope > max_slope
        else:
            return False

    def condition(x, values, k, points):
        """
        Subfunction to check the condition in the loop, depending on the set of values we already have.
        values: set of values of the function we explored (numpy array)
        x: point to check (numpy array)
        k: Lipschitz constant (float)
        points: set of points we have explored (numpy array)
        """
        max_val = np.max(values)

        left_min = np.min(
            values.reshape(-1) + k * np.linalg.norm(x - points, ord=2, axis=1)
        )

        return left_min >= max_val

    # Main loop
    while t < n:
        X_tp1 = Uniform(X)
        nb_samples += 1
        last_nb_samples[-1] = nb_samples
        if condition(X_tp1, values, k, points):
            points = np.concatenate((points, X_tp1.reshape(1, -1)))

            values = np.concatenate((values, np.array([f(X_tp1)])))

            t += 1
            last_nb_samples.append(0)

        elif slope_stop_condition():
            print(
                f"Exponential growth of the number of samples. Stopping the algorithm at iteration {t}."
            )
            break

    # Output
    return points, values, t
