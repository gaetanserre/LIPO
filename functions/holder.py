import numpy as np

class Function:
  def __init__(self) -> None:
    self.bounds = np.array([(-10, 10), (-10, 10)])
    self.k = 30
    
    self.radius = 1 # Dummy value
    self.diam = 2 * self.radius

  def __call__(self, x: np.ndarray) -> float:
    return np.abs( 
      np.sin(x[0])
      * np.cos(x[1])
      * np.exp(np.abs(1 - (np.sqrt(x[0]**2 + x[1]**2)) / np.pi)) )