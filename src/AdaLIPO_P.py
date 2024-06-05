"""
Copyright (c) 2023 Perceval Beja-Battais, Gaëtan Serré and Sophia Chirrane

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <http://www.gnu.org/licenses/>. 
"""

import numpy as np
from collections import deque
from utils import *


"""
This function implements the AdaLIPO algorithm
(in the paper, Algorithm 4 with a decreasing Bernoulli parameter and the stopping criterion).
"""


def AdaLIPO_P(f, n: int, window_slope=5, max_slope=800.0):
    """
    f: class of the function to maximize (class)
    n: number of function evaluations (int)
    p: probability of success for exploration/exploitation (float)
    fig_path: path to save the statistics figures (str)
    delta: confidence level for bounds (float)
    window_slope: size of the window to compute the slope of the nb_samples vs nb_evaluations curve (int)
    max_slope: maximum slope for the nb_samples vs nb_evaluations curve (float)
    """

    # Initialization
    t = 1
    alpha = 10e-2
    k_hat = 0

    X_1 = Uniform(f.bounds)
    nb_samples = 1

    # We keep track of the last `window_slope` values of nb_samples to compute the slope
    last_nb_samples = deque([1], maxlen=window_slope)

    points = X_1.reshape(1, -1)
    values = np.array([f(X_1)])

    def k(i):
        """
        Series of potential Lipschitz constants.
        """
        return (1 + alpha) ** i

    # Statistics
    stats = []

    def p(t):
        """
        Probability of success for exploration/exploitation.
        """
        if t == 1:
            return 1
        else:
            return 1 / np.log(t)

    # Main loop
    ratios = []
    while t < n:
        B_tp1 = Bernoulli(p(t))
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
                if LIPO_condition(X_tp1, values, k_hat, points):
                    points = np.concatenate((points, X_tp1.reshape(1, -1)))
                    break
                elif slope_stop_condition(last_nb_samples, max_slope):
                    print(
                        f"Exponential growth of the number of samples. Stopping the algorithm at iteration {t}."
                    )
                    stats = (points, values, t, stats)
                    # Output
                    return np.max(values), stats
            value = f(X_tp1)

        values = np.concatenate((values, np.array([value])))
        for i in range(points.shape[0] - 1):
            ratios.append(
                np.abs(value - values[i]) / np.linalg.norm(X_tp1 - points[i], ord=2)
            )

        i_hat = int(np.ceil(np.log(max(ratios)) / np.log(1 + alpha)))
        k_hat = k(i_hat)

        # Statistical analysis
        stats.append((np.max(values), nb_samples, k_hat))

        t += 1

        # As `last_nb_samples` is a deque, we insert a 0 at the end of the list and increment this value by 1 for each point sampled instead of making a case distinction for the first sample and the others.
        last_nb_samples.append(0)

        if t % 200 == 0:
            print(
                f"Iteration: {t} Lipschitz constant: {k_hat:.4f} Number of samples: {nb_samples}"
            )

    stats = (points, values, stats)
    # Output
    return np.max(values), stats
