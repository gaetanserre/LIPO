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
  args = argparse.ArgumentParser()
  args.add_argument("--function", "-f", type=str, help="Function class to maximize", required=True)
  args.add_argument("--n_eval", "-n", type=int, help="Number of function evaluations", required=True)
  args.add_argument("--n_run", "-r", type=int, help="Number of runs", default=100)
  args.add_argument("--delta", "-delta", type=float, help="With proba 1-delta, the bounds are made", default=0.05)
  return args.parse_args()


def runs(
  n_run: int,
  n_eval: int,
  f,
  optimizer,
  method: str,
  delta=0.05,
  fig_path=None,
):
  """
  Run the optimizer several times and return the points and values of the last run.
  n_run: number of runs (int)
  n_eval: number of function evaluations (int)
  f: function class to maximize (class)
  optimizer: optimizer function (function)
  method: name of the optimizer (str)
  delta: with proba 1-delta, the bounds holds (float)
  p: probability of success (float)
  fig_path: path to save the statistics figures (str)
  """
  vs = []
  nb_evals = []
  for i in range(n_run):
    if optimizer == random_search:
      points, values, nb_eval = optimizer(f, n=n_eval)
    elif optimizer == AdaLIPO:
      points, values, nb_eval = optimizer(
        f,
        n=n_eval,
        delta=delta,
        fig_path=fig_path
      )
    elif optimizer == LIPO:   
      points, values, nb_eval = optimizer(
        f,
        n=n_eval,
        delta=delta,
        fig_path=fig_path
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
  if not os.path.exists("figures/"):
    os.mkdir(f"figures/")
  
  if not os.path.exists(f"figures/{args.function}"):
    os.mkdir(f"figures/{args.function}")

  # Several runs of random search
  points, values = runs(args.n_run, args.n_eval, f, random_search, "random_search")
  # Generate the figure using the last run
  path = f"figures/{args.function}/random_search.pdf"
  fig_gen.gen_figure(points, values, path=path)

  # Several runs of LIPO
  fig_path = f"figures/{args.function}/LIPO"
  points, values = runs(
    args.n_run,
    args.n_eval,
    f,
    LIPO,
    "LIPO",
    delta=args.delta,
    fig_path=fig_path)
  # Generate the figure using the last run
  path = f"{fig_path}.pdf"
  fig_gen.gen_figure(points, values, path=path)
  
  
  # Several runs of AdaLIPO
  fig_path = f"figures/{args.function}/AdaLIPO"
  points, values = runs(
    args.n_run,
    args.n_eval,
    f,
    AdaLIPO,
    "AdaLIPO",
    delta=args.delta,
    fig_path=fig_path)
  # Generate the figure using the last run
  path = f"{fig_path}.pdf"
  fig_gen.gen_figure(points, values, path=path)