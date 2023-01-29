import argparse
import importlib
import os
import sys
# Add the example functions folder to the path
sys.path.append("./functions")

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
  args.add_argument("--function", "-f", type=str, help="Function class to maximize", required=True)
  args.add_argument("--n_eval", "-n", type=int, help="Number of function evaluations", required=True)
  args.add_argument("--n_run", "-r", type=int, help="Number of runs", default=100)
  args.add_argument("--k", "-k", type=int, help="Sequence of Lipchitz constants", default=L)
  args.add_argument("--p", "-p", type=float, help="Probability of success", default=0.5)
  args.add_argument("--delta", "-delta", type=float, help="With proba 1-delta, the bounds are made", default=0.05)
  args.add_argument("--radius", "-rad", type=float, help="Radius of the set", default=np.sqrt((5.12)**2 + (5.12)**2))
  args.add_argument("--diameter", "-D", type=float, help="Diameter of the set", default=2*np.sqrt((5.12)**2 + (5.12)**2))
  return args.parse_args()


def runs(n_run: int, n_eval: int, f, optimizer, method, delta=0.05, radius=1, diameter=2, k=None, p=None):
  """
  Run the optimizer several times and return the points and values of the last run.
  n_run: number of runs (int)
  n_eval: number of function evaluations (int)
  f: function class to maximize (class)
  optimizer: optimizer function (function)
  """
  vs = []
  nb_evals = []
  for i in range(n_run):
    if optimizer == random_search:
      points, values, nb_eval = optimizer(f, n=n_eval)
    elif optimizer == AdaLIPO:
      points, values, nb_eval = optimizer(f, n=n_eval, k=k, p=p)
    elif optimizer == LIPO:   
      points, values, nb_eval = optimizer(
        f,
        n=n_eval,
        delta=delta,
        radius=radius,
        diameter=diameter
      )
    vs.append(np.max(values))
    nb_evals.append(nb_eval)

  print(f"Method: {method}")
  print(f"Number of samples: {np.mean(nb_evals):.2f} +- {np.std(nb_evals):.2f}")
  print(f"Mean value: {np.mean(vs):.4f}, std: {np.std(vs):.4f}")
  print(f"Best maximizer: {points[np.argmax(values)]}")
  print(f"Best value: {np.max(values):.4f}\n")
  return points, values

if __name__ == '__main__':
  args = cli()

  # Remove the folder name
  if len(args.function.split('/')) > 1:
    args.function = args.function.split('/')[1]
  # remove the .py extension
  args.function = args.function.split('.')[0]

  # Dynamically import the function class
  f = importlib.import_module(args.function).Function()

  # Check that the function is 1D or 2D
  assert f.bounds.shape[0] <= 2, "Only 1D and 2D functions are supported for this demo."

  # Instantiate the figure generator
  fig_gen = FigGenerator(f)
  if not os.path.exists(f"figures/"):
    os.mkdir(f"figures/")

  # Several runs of random search
  points, values = runs(args.n_run, args.n_eval, f, random_search, "random_search")
  # Generate the figure using the last run
  path = f"figures/{args.function}_random_search.pdf"
  fig_gen.gen_figure(points, values, "random_search", path=path)

  # Several runs of LIPO
  points, values = runs(
    args.n_run,
    args.n_eval,
    f,
    LIPO,
    "LIPO",
    delta=args.delta,
    radius=args.radius,
    diameter=args.diameter
  )
  # Generate the figure using the last run
  path = f"figures/{args.function}_LIPO.pdf"
  fig_gen.gen_figure(points, values, "LIPO", path=path)
  
  
  # Several runs of AdaLIPO
  points, values = runs(args.n_run, args.n_eval, f, AdaLIPO, "AdaLIPO", k=args.k, p=args.p)
  # Generate the figure using the last run
  path = f"figures/{args.function}_AdaLIPO.pdf"
  fig_gen.gen_figure(points, values, "AdaLIPO", path=path)


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