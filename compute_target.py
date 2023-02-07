import sys
import numpy as np
from utils import *
# Add the example functions folder to the path
sys.path.append("./functions")

import argparse
import importlib
from tqdm.auto import tqdm

def cli():
  args = argparse.ArgumentParser()
  args.add_argument("--function", "-f", type=str, help="Function class to maximize", required=True)
  return args.parse_args()


if __name__ == '__main__':
  args = cli()

  # Remove the folder name
  if len(args.function.split('/')) > 1:
    args.function = args.function.split('/')[1]
  # remove the .py extension
  args.function = args.function.split('.')[0]

  # Dynamically import the function class
  f = importlib.import_module(args.function).Function()

  xs = []
  values = []
  for _ in tqdm(range(10_000_000)):
    x = Uniform(f.bounds)
    xs.append(x)
    values.append(f(x))
  
  print(np.mean(values))
  print(np.max(values))

