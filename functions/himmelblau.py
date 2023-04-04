import numpy as np


class Function:
    def __init__(self) -> None:
        self.bounds = np.array([(-4, 4), (-4, 4)])
        self.k = 283

        self.radius = 4
        self.diam = 8 * np.sqrt(2)

    def __call__(self, x: np.ndarray) -> float:
        return -((x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2)
