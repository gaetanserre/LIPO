import numpy as np

class Function:
  def __init__(self) -> None:
    self.bounds = np.array([(-3, 3), (-3, 3)])
    self.k = 7304
    
  def __call__(self, x: np.ndarray) -> float:
    return -(100 * np.square(x[1] - np.square(x[0])) + np.square(1 - x[0]))