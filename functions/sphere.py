import numpy as np

class Function:
  def __init__(self) -> None:
    self.bounds = np.array([(0, 1), (0, 1)])
    self.k = 1.5
    self.kappa = 1
    self.c_kappa = 1

    self.radius = 1 # Dummy value
    self.diam = 2 * self.radius
    
  def __call__(self, x: np.ndarray) -> float:
    return -np.sqrt( (x[0] - np.pi / 16)**2 + (x[1]-np.pi/16)**2 )