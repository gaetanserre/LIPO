"""
Copyright (c) 2023 Perceval Beja-Battais, Gaëtan Serré and Sophia Chirrane

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <http://www.gnu.org/licenses/>. 
"""

import matplotlib.pyplot as plt

import numpy as np


class FigGenerator:
    def __init__(self, f):
        """
        Class to generate figures for visualizing the optimization process
        f: function optimized (Function)
        """

        self.f = f

    def gen_figure(
        self, eval_points: np.array, eval_values: np.array, path: str = None
    ):
        """
        Generates a figure
        eval_points: points where the function was evaluated (np.array)
        eval_values: values of the function at the evaluation points (np.array)
        path: path to save the figure (str) (optional)
        """

        dim = eval_points.shape[1]
        if dim == 1:
            self.gen_1D(eval_points, eval_values)
        elif dim == 2:
            self.gen_2D(eval_points, eval_values)
        else:
            raise ValueError(
                f"Cannot generate a figure for {dim}-dimensional functions"
            )

        if path is not None:
            plt.savefig(path, bbox_inches="tight")
        else:
            plt.show()
        plt.clf()
        plt.close()

    def gen_1D(self, eval_points, eval_values):
        """
        Generates a figure for 1D functions
        """

        x = np.linspace(self.f.bounds[0][0], self.f.bounds[0][1], 1000)
        y = self.f(x)

        plt.plot(x, y)
        plt.scatter(
            eval_points,
            eval_values,
            c=eval_values,
            label="evaluations",
            cmap="viridis",
            zorder=2,
        )
        plt.colorbar(fraction=0.046, pad=0.04)
        # plt.plot(eval_points, eval_values, linewidth=0.5, color="black")
        plt.xlabel("$X$")
        plt.ylabel("$f(x)$")
        plt.legend()

    def gen_2D(self, eval_points, eval_values):
        """
        Generates a figure for 2D functions
        """

        x = np.linspace(self.f.bounds[0][0], self.f.bounds[0][1], 1000)
        y = np.linspace(self.f.bounds[1][0], self.f.bounds[1][1], 1000)
        x, y = np.meshgrid(x, y)
        z = self.f([x, y])

        fig = plt.figure(figsize=(15, 15))
        ax = plt.axes(projection="3d", computed_zorder=False)

        ax.plot_surface(
            x, y, z, cmap="coolwarm", linewidth=0, antialiased=True, zorder=4.4
        )

        cb = ax.scatter(
            eval_points[:, 0],
            eval_points[:, 1],
            eval_values,
            c=eval_values,
            label="evaluations",
            cmap="viridis",
            zorder=4.5,
        )

        # plt.colorbar(cb, fraction=0.046, pad=0.04)

        ax.set_xlabel("$X$", fontsize=22)
        ax.set_ylabel("$Y$", fontsize=22)
        ax.legend(fontsize=22)
