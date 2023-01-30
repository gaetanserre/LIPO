# Credits https://gist.github.com/mauris/3704440

import numpy as np

class Function:
  def __init__(self) -> None:
    self.bounds = np.array([(0, 10)])
    self.k = 200

    self.radius = 5
    self.diam = 10
    
  def __call__(self, x: np.ndarray) -> float:
    return 9 * np.sin(20 * x**(6/7)) * np.sin(4 * x**(3/2))