import argparse
import os
import subprocess

import numpy as np
from fig_generator import FigGenerator
from LIPO import LIPO
from random_search import random_search
from AdaLIPO import AdaLIPO

current_dir = os.path.dirname(os.path.realpath(__file__))

def cli():
  args = argparse.ArgumentParser()

  args.add_argument("--function", "-f", type=str, help="Numpy function to maximize", required=True)
  args.add_argument("--n_eval", "-n", type=int, help="Number of function evaluations", required=True)
  args.add_argument("--bounds", "-b", type=str, help="Bounds of the parameters", required=True)
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
  print(f"Method: {method}")
  vs = []
  nb_evals = []
  for i in range(n_run):
    if optimizer == LIPO:
      points, values, nb_eval = optimizer(f, X, k, n=n_eval)
    else:
      points, values, nb_eval = optimizer(f, X, n=n_eval)
    vs.append(np.max(values))
    nb_evals.append(nb_eval)

  print(f"Number of evaluations: {np.mean(nb_evals):.2f} +- {np.std(nb_evals):.2f}")
  print(f"Mean value: {np.mean(vs):.4f}, std: {np.std(vs):.4f}")
  print(f"Best maximizer: {points[np.argmax(values)]}\n")
  return points, values

if __name__ == '__main__':
  args = cli()

  n_runs = 2

  # Parse the function expression and create a lambda function from it
  if subprocess.run([f"{current_dir}/numpy_parser.exe", args.function]).returncode != 0:
    print("Function expression is not valid.")
    exit(0)
  f = lambda x: eval(args.function)


  # Parse the bounds and verify that they are in 1D or 2D
  args.bounds = [float(b) for b in args.bounds.split(' ')]
  if len(args.bounds) == 2:
    X = np.array([(args.bounds[0], args.bounds[1])])
  elif len(args.bounds) == 4:
    X = np.array([(args.bounds[0], args.bounds[1]), (args.bounds[2], args.bounds[3])])
  else:
    print("Only 1D and 2D functions are supported for this demo.")
    exit(0)

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
  
  # Several runs of AdaLIPO
  points, values = runs(n_runs, args.n_eval, f, X, AdaLIPO, "AdaLIPO", k=args.k)
  # Generate the figure using the last run
  path = f"figures/{args.name}_AdaLIPO.png"
  fig_gen.gen_figure(points, values, "AdaLIPO", path=path)