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
This function implements the LIPO-E algorithm
(in the paper, Algorithm 3 with the stopping criterion).
"""


def LIPO_P(f, n: int, window_slope=5, max_slope=600.0):
    """
    f: class of the function to maximize (class)
    n: number of function evaluations (int)
    fig_path: path to save the statistics figures (str)
    window_slope: size of the window to compute the slope of the nb_samples vs nb_evaluations curve (int)
    max_slope: maximum slope for the nb_samples vs nb_evaluations curve (float)
    """

    # Initialization
    t = 1

    X_1 = Uniform(f.bounds)
    nb_samples = 1

    # We keep track of the last `window_slope` values of nb_samples to compute the slope
    last_nb_samples = deque([1], maxlen=window_slope)

    points = X_1.reshape(1, -1)
    values = np.array([f(X_1)])

    # Statistics
    stats = []

    # Main loop
    while t < n:
        X_tp1 = Uniform(f.bounds)
        nb_samples += 1
        last_nb_samples[-1] = nb_samples
        if LIPO_condition(X_tp1, values, f.k, points):
            points = np.concatenate((points, X_tp1.reshape(1, -1)))

            values = np.concatenate((values, np.array([f(X_tp1)])))

            # Statistical analysis
            stats.append((np.max(values), nb_samples))

            t += 1
            last_nb_samples.append(0)

        elif slope_stop_condition(last_nb_samples, max_slope):
            print(
                f"Exponential growth of the number of samples. Stopping the algorithm at iteration {t}."
            )
            stats = (points, values, stats)
            # Output
            return np.max(values), stats

    stats = (points, values, stats)
    # Output
    return np.max(values), stats
