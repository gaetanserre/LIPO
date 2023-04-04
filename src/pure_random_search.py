"""
Copyright (c) 2023 Perceval Beja-Battais, Gaëtan Serré and Sophia Chirrane

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <http://www.gnu.org/licenses/>. 
"""


import numpy as np
from utils import *

"""
This function implements the Pure Random Search algorithm.
"""


def pure_random_search(f, n: int):
    """
    f: class of the function to maximize (class)
    n: number of function evaluations (int)
    """

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
