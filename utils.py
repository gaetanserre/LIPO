def percentage_difference(x, y):
  """
  Computes the percentage difference between x and y.
  x: first value (float)
  y: second value (float)
  """
  return abs(x - y)

def slope_stop_condition(last_nb_samples, size_slope, max_slope):
    """
    Check if the slope of the last `size_slope` points of the the nb_samples vs nb_evaluations curve 
    is greater than max_slope.
    """
    if len(last_nb_samples) == size_slope:
      slope = (last_nb_samples[-1] - last_nb_samples[0]) / (len(last_nb_samples) - 1)
      return slope > max_slope
    else:
      return False