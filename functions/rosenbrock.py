import numpy as np

class Function:
  def __init__(self) -> None:
    self.bounds = np.array([(-3, 3), (-3, 3)])
    self.k = 14607

    self.radius = 3
    self.diam = 2 * 3 * np.sqrt(2)

    self.max = 0
    self.dist_max = 1e-1
    
  def __call__(self, x: np.ndarray) -> float:
    return -(100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2)