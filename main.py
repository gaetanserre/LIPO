import argparse
import importlib

import numpy as np
from LIPO import LIPO
from random_search import random_search
from fig_generator import FigGenerator

def cli():
  args = argparse.ArgumentParser()
  args.add_argument("--function", "-f", type=str, help="Function class to maximize", required=True)
  args.add_argument("--n_eval", "-n", type=int, help="Number of function evaluations", required=True)
  args.add_argument("--n_run", "-r", type=int, help="Number of runs", default=100)
  return args.parse_args()


def runs(n_run: int, n_eval: int, f, optimizer, method):
  """
  Run the optimizer several times and return the points and values of the last run.
  n_run: number of runs (int)
  n_eval: number of function evaluations (int)
  f: function class to maximize (class)
  optimizer: optimizer function (function)
  """
  vs = []
  for k in range(n_run):
    values, points = optimizer(f, n=n_eval)
    max_value = np.max(values)
    vs.append(max_value)
  print(f"Method: {method}")
  print(f"Mean value: {np.mean(vs):.4f}, std: {np.std(vs):.4f}\n")
  return points, values

if __name__ == '__main__':
  args = cli()

  # remove the .py extension
  args.function = args.function.split(".")[0]

  # Dynamically import the function class
  f = importlib.import_module(args.function).Function()

  # Check that the function is 1D or 2D
  assert f.bounds.shape[0] <= 2, "Only 1D and 2D functions are supported for this demo."

  # Instantiate the figure generator
  fig_gen = FigGenerator(f)

  # Several runs of random search
  points, values = runs(args.n_run, args.n_eval, f, random_search, "random_search")
  # Generate the figure using the last run
  fig_gen.gen_figure(points, values, "random_search", path="figures/random_search.pdf")

  # Several runs of random search
  points, values = runs(args.n_run, args.n_eval, f, LIPO, "LIPO")
  # Generate the figure using the last run
  fig_gen.gen_figure(points, values, "LIPO", path="figures/LIPO.pdf")


  """ from lipo import GlobalOptimizer

  def function(x, y):
    return -(100 * (y - x ** 2) ** 2 + (1 - x) ** 2)

  pre_eval_x = dict(x=9.3, y=-9.4)
  evaluations = [(pre_eval_x, function(**pre_eval_x))]

  search = GlobalOptimizer(
      function,
      lower_bounds={"x": -10.0, "y": -10.0},
      upper_bounds={"x": 10.0, "y": 10.0},
      evaluations=evaluations,
      maximize=True,
  )

  num_function_calls = 1000
  search.run(num_function_calls)

  print(f"Max value: {search.optimum}") """