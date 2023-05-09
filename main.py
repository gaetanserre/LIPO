import argparse
import os
import subprocess

import numpy as np
from fig_generator import FigGenerator
from LIPO_E import LIPO_E
from random_search import random_search
from AdaLIPO_E import AdaLIPO_E

current_dir = os.path.dirname(os.path.realpath(__file__))


def return_error(error):
    with open("demo_failure.txt", "w") as f:
        f.write(error)
    exit(0)


def cli():
    args = argparse.ArgumentParser()

    args.add_argument(
        "--function", "-f", type=str, help="Numpy function to maximize", required=True
    )
    args.add_argument(
        "--n_eval", "-n", type=int, help="Number of function evaluations", required=True
    )
    args.add_argument(
        "--bounds", "-b", type=str, help="Bounds of the parameters", required=True
    )
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
    k: Lipschitz constant (float)s
    """
    print(f"Method: {method}")
    vs = []
    nb_evals = []
    for i in range(n_run):
        if optimizer == LIPO_E:
            points, values, nb_eval = optimizer(f, X, k, n=n_eval)
        else:
            points, values, nb_eval = optimizer(f, X, n=n_eval)
        vs.append(np.max(values))
        nb_evals.append(nb_eval)

    print(f"Number of evaluations: {np.mean(nb_evals):.2f} +- {np.std(nb_evals):.2f}")
    print(f"Mean value: {np.mean(vs):.4f}, std: {np.std(vs):.4f}")
    print(f"Best maximizer: {points[np.argmax(values)]}\n")
    return points, values


if __name__ == "__main__":
    args = cli()

    n_runs = 2

    # Parse the function expression and create a lambda function from it
    if (
        subprocess.run(
            [
                f"{current_dir}/numpy_parser numpy_parser/numpy_primitives.txt",
                args.function,
            ]
        ).returncode
        != 0
    ):
        return_error("Function expression is not valid.")
    f = lambda x: eval(args.function)

    # Parse the bounds and verify that they are in 1D or 2D
    bounds = []
    for b in args.bounds.split(" "):
        try:
            bounds.append(float(b))
        except:
            return_error("Bounds must be float.")
    if len(bounds) == 2:
        X = np.array([(bounds[0], bounds[1])])
    elif len(bounds) == 4:
        X = np.array([(bounds[0], bounds[1]), (bounds[2], bounds[3])])
    else:
        return_error("Only 1D and 2D functions are supported for this demo.")

    # Instantiate the figure generator
    fig_gen = FigGenerator(f, X)
    if not os.path.exists(f"figures/"):
        os.mkdir(f"figures/")

    # Several runs of random search
    points, values = runs(n_runs, args.n_eval, f, X, random_search, "random_search")
    # Generate the figure using the last run
    path = f"figures/{args.name}_random_search.png"
    fig_gen.gen_figure(points, values, "random_search", path=path)

    # Several runs of LIPO-E
    if args.k > 0:
        points, values = runs(n_runs, args.n_eval, f, X, LIPO_E, "LIPO-E", k=args.k)
        # Generate the figure using the last run
        path = f"figures/{args.name}_LIPO-E.png"
        fig_gen.gen_figure(points, values, "LIPO-E", path=path)

    # Several runs of AdaLIPO-E
    points, values = runs(n_runs, args.n_eval, f, X, AdaLIPO_E, "AdaLIPO-E", k=args.k)
    # Generate the figure using the last run
    path = f"figures/{args.name}_AdaLIPO-E.png"
    fig_gen.gen_figure(points, values, "AdaLIPO-E", path=path)
