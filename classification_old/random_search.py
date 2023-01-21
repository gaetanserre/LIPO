#
# Created in 2023 by Gaëtan Serré
#

import torch

def random_search(f, shape, device, n_iter=100):
  """
  Random search algorithm.
  `f`: black-box function to optimize
  `bounds`: bounds of the search space
  `n_iter`: number of iterations
  """
  best = None
  for i in range(n_iter):
    # Generate random point
    theta = torch.rand(shape, device=device) * 2 - 1

    # Evaluate objective
    obj = f(theta)
    print(obj)
    # Update best
    if best is None or obj > best[0]:
      best = (obj, theta)
  return best