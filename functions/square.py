import numpy as np

class Function:
  def __init__(self) -> None:
    self.bounds = np.array([(-10, 10), (-10, 10)])
    self.k = 28.29

    self.radius = 10
    self.diam = 2 * 10 * np.sqrt(2)

  def __call__(self, x: np.ndarray) -> float:
    return -(x[0]**2 + x[1]**2)
