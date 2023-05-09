import numpy as np


class Function:
    def __init__(self) -> None:
        self.bounds = np.array([(-10, 10)])
        self.k = 2 * 10

    def __call__(self, x: np.ndarray) -> float:
        return -(x**2)
