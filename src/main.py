"""
Copyright (c) 2023 Perceval Beja-Battais, Gaëtan Serré and Sophia Chirrane

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <http://www.gnu.org/licenses/>. 
"""

import argparse
import importlib
import os
import sys
import numpy as np
import time

# Add the example functions folder to the path
sys.path.append("./functions")

from fig_generator import FigGenerator
from pure_random_search import pure_random_search
from LIPO import LIPO
from LIPO_P import LIPO_P
from AdaLIPO import AdaLIPO
from AdaLIPO_P import AdaLIPO_P
from statistical_analysis import LIPO_Statistics


def cli():
    args = argparse.ArgumentParser()
    args.add_argument(
        "--function", "-f", type=str, help="Function class to maximize", required=True
    )
    args.add_argument(
        "--n_eval", "-n", type=int, help="Number of function evaluations", required=True
    )
    args.add_argument("--n_run", "-r", type=int, help="Number of runs", default=100)
    args.add_argument(
        "--delta",
        "-delta",
        type=float,
        help="With proba 1-delta, the bounds are made",
        default=0.05,
    )
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
    print(f"Method: {method}")
    times = []
    vs = []
    nb_evals = []
    for _ in range(n_run):
        start_time = time.time()
        if optimizer == pure_random_search:
            points, values, nb_eval = optimizer(f, n=n_eval)
        elif optimizer == AdaLIPO or optimizer == AdaLIPO_P:
            _, stats = optimizer(f, n=n_eval)
            points, values, nb_eval, stats = stats
            LIPO_stats = LIPO_Statistics(f, fig_path, stats, delta=delta, k_hat=True)
            LIPO_stats.plot()

        elif optimizer == LIPO or optimizer == LIPO_P:
            _, stats = optimizer(f, n=n_eval)
            points, values, nb_eval, stats = stats
            LIPO_stats = LIPO_Statistics(f, fig_path, stats, delta=delta)
            LIPO_stats.plot()

        times.append(time.time() - start_time)
        vs.append(np.max(values))
        nb_evals.append(nb_eval)

    print(f"Number of evaluations: {np.mean(nb_evals):.2f} +- {np.std(nb_evals):.2f}")
    print(f"Mean value: {np.mean(vs):.4f}, std: {np.std(vs):.4f}")
    print(f"Best maximizer: {points[np.argmax(values)]}")
    print(f"Best value: {np.max(values):.4f}")
    print(f"Mean time: {np.mean(times):.2f} +- {np.std(times):.2f}\n")
    return points, values


if __name__ == "__main__":
    args = cli()

    # Remove the folder name
    if len(args.function.split("/")) > 1:
        args.function = args.function.split("/")[1]
    # remove the .py extension
    args.function = args.function.split(".")[0]

    # Dynamically import the function class
    f = importlib.import_module(args.function).Function()

    # Check that the function is 1D or 2D
    gen_fig = f.bounds.shape[0] <= 2

    # Instantiate the figure generator
    fig_gen = FigGenerator(f)
    if not os.path.exists("figures/"):
        os.mkdir(f"figures/")

    if not os.path.exists(f"figures/{args.function}"):
        os.mkdir(f"figures/{args.function}")
        os.mkdir(f"figures/{args.function}/LIPO")
        os.mkdir(f"figures/{args.function}/LIPO_P")
        os.mkdir(f"figures/{args.function}/AdaLIPO")
        os.mkdir(f"figures/{args.function}/AdaLIPO_P")

    # Several runs of random search
    points, values = runs(
        args.n_run, args.n_eval, f, pure_random_search, "pure_random_search"
    )
    # Generate the figure using the last run
    path = f"figures/{args.function}/pure_random_search.pdf"
    if gen_fig:
        fig_gen.gen_figure(points, values, path=path)

    """ # Several runs of LIPO
    fig_path = f"figures/{args.function}/LIPO"
    points, values = runs(
        args.n_run, args.n_eval, f, LIPO, "LIPO", delta=args.delta, fig_path=fig_path
    )
    # Generate the figure using the last run
    path = f"{fig_path}/plot.pdf"
    if gen_fig:
        fig_gen.gen_figure(points, values, path=path)

    # Several runs of LIPO+
    fig_path = f"figures/{args.function}/LIPO_P"
    points, values = runs(
        args.n_run,
        args.n_eval,
        f,
        LIPO_P,
        "LIPO_P",
        delta=args.delta,
        fig_path=fig_path,
    )
    # Generate the figure using the last run
    path = f"{fig_path}/plot.pdf"
    if gen_fig:
        fig_gen.gen_figure(points, values, path=path) """

    # Several runs of AdaLIPO
    fig_path = f"figures/{args.function}/AdaLIPO"
    points, values = runs(
        args.n_run,
        args.n_eval,
        f,
        AdaLIPO,
        "AdaLIPO",
        delta=args.delta,
        fig_path=fig_path,
    )
    # Generate the figure using the last run
    path = f"{fig_path}/plot.pdf"
    if gen_fig:
        fig_gen.gen_figure(points, values, path=path)

    # Several runs of AdaLIPO+
    fig_path = f"figures/{args.function}/AdaLIPO_P"
    points, values = runs(
        args.n_run,
        args.n_eval,
        f,
        AdaLIPO_P,
        "AdaLIPO_P",
        delta=args.delta,
        fig_path=fig_path,
    )
    # Generate the figure using the last run
    path = f"{fig_path}/plot.pdf"
    if gen_fig:
        fig_gen.gen_figure(points, values, path=path)
