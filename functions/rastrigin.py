import numpy as np

class Function:
  def __init__(self) -> None:
    self.bounds = np.array([(-5.12, 5.12), (-5.12, 5.12)])
    self.k = 96
    self.kappa = 2
    self.c_kappa = 1 + 20 * np.pi **2
    
  def __call__(self, x: np.ndarray) -> float:
    return -(10 * 2 + (x[0]**2 - 10 * np.cos(2*np.pi*x[0])) + (x[1]**2 - 10 * np.cos(2*np.pi*x[1])))