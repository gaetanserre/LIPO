import numpy as np

class Function:
  def __init__(self) -> None:
    self.bounds = np.array([(-10, 10), (-10, 10)])
    self.k = 28.29
    self.kappa = 2
    self.c_kappa = 1

    self.radius = 10
    self.diam = 2 * 10 * np.sqrt(2)

    self.max = 0
    self.mean = -66.65471044810451
    self.target_t = lambda t: self.max - (self.max - self.mean) * (1 - t)

  def __call__(self, x: np.ndarray) -> float:
    return -(x[0]**2 + x[1]**2)