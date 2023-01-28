import argparse
import os
import subprocess

import numpy as np
from fig_generator import FigGenerator
from LIPO import LIPO
from random_search import random_search
from AdaLIPO import AdaLIPO

def cli():
  L = np.array([1*10**(-3), 2*10**(-3), 3*10**(-3), 4*10**(-3), 5*10**(-3), 6*10**(-3), 7*10**(-3), 8*10**(-3), 9*10**(-3)])
  L = np.concatenate((L, np.array([1*10**(-2), 2*10**(-2), 3*10**(-2), 4*10**(-2), 5*10**(-2), 6*10**(-2), 7*10**(-2), 8*10**(-2), 9*10**(-2)])))
  L = np.concatenate((L, np.array([1*10**(-1), 2*10**(-1), 3*10**(-1), 4*10**(-1), 5*10**(-1), 6*10**(-1), 7*10**(-1), 8*10**(-1), 9*10**(-1)])))
  L = np.concatenate((L, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])))
  L = np.concatenate((L, np.array([20, 30, 40, 50, 60, 70, 80, 90, 100])))
  L = np.concatenate((L, np.array([200, 300, 400, 500, 600, 700, 800, 900, 1000])))
  args = argparse.ArgumentParser()

  args.add_argument("--function", "-f", type=str, help="Numpy function to maximize", required=True)
  args.add_argument("--n_eval", "-n", type=int, help="Number of function evaluations", required=True)
  args.add_argument("--bounds", "-b", type=float, nargs='+', help="Bounds of the parameters", required=True)
  args.add_argument("--k", "-k", type=float, help="Lipchitz constants", default=None)
  args.add_argument("--name", type=str, help="Name of the function", default="fun")
  return args.parse_args()


def runs(n_run: int, n_eval: int, f, X, optimizer, method, k=None):
  """
  Run the optimizer several times and return the points and values of the last run.
  n_run: number of runs (int)
  n_eval: number of function evaluations (int)
  f: function class to maximize (class)
  X: bounds of the parameters (np.ndarray)
  optimizer: optimizer function (function)
  method: name of the optimizer (str)
  k: Lipschitz constant (float)
  """
  vs = []
  nb_evals = []
  for i in range(n_run):
    if k is None:
      points, values, nb_eval = optimizer(f, X, n=n_eval)
    else:
      points, values, nb_eval = optimizer(f, X, k, n=n_eval)
    vs.append(np.max(values))
    nb_evals.append(nb_eval)

  print(f"Method: {method}")
  print(f"Number of samples: {np.mean(nb_evals):.2f} +- {np.std(nb_evals):.2f}")
  print(f"Mean value: {np.mean(vs):.4f}, std: {np.std(vs):.4f}")
  print(f"Best maximizer: {points[np.argmax(values)]}\n")
  return points, values

if __name__ == '__main__':
  args = cli()

  n_runs = 2

  # Parse the function expression and create a lambda function from it
  if subprocess.run(["/workdir/bin/numpy_parser.exe", args.function]).returncode != 0:
    raise Exception("Function expression is not valid.")
  f = lambda x: eval(args.function)


  # Parse the bounds and verify that they are in 1D or 2D
  if len(args.bounds) == 2:
    X = np.array([(args.bounds[0], args.bounds[1])])
  elif len(args.bounds) == 4:
    X = np.array([(args.bounds[0], args.bounds[1]), (args.bounds[2], args.bounds[3])])
  else: raise ValueError("Only 1D and 2D functions are supported for this demo.")

  # Instantiate the figure generator
  fig_gen = FigGenerator(f, X)
  if not os.path.exists(f"figures/"):
    os.mkdir(f"figures/")

  # Several runs of random search
  points, values = runs(n_runs, args.n_eval, f, X, random_search, "random_search")
  # Generate the figure using the last run
  path = f"figures/{args.name}_random_search.png"
  fig_gen.gen_figure(points, values, "random_search", path=path)

  # Several runs of LIPO
  points, values = runs(n_runs, args.n_eval, f, X, LIPO, "LIPO", k=args.k)
  # Generate the figure using the last run
  path = f"figures/{args.name}_LIPO.png"
  fig_gen.gen_figure(points, values, "LIPO", path=path)
  
  """ # Several runs of AdaLIPO
  points, values = runs(args.n_run, args.n_eval, f, AdaLIPO, "AdaLIPO", k=args.k, p=args.p)
  # Generate the figure using the last run
  path = f"figures/{args.function}_AdaLIPO.pdf"
  fig_gen.gen_figure(points, values, "AdaLIPO", path=path) """


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