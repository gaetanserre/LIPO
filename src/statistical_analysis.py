"""
Copyright (c) 2023 Perceval Beja-Battais, Gaëtan Serré and Sophia Chirrane

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <http://www.gnu.org/licenses/>. 
"""

import numpy as np
import matplotlib.pyplot as plt

# Those convergence rates are supposed to be for LIPO. We can try to adapt them for AdaLIPO, but they won't make sense before
# the Lipscitz constant is properly approximated.
# THOSE BOUNDS ARE CORRECT ONLY FOR UNIQUE MAXIMUMS, so in our examples, only for sphere, rastrigin, and square in our examples
# but not for himmelblau and holder table. For rosenbrock, the maximum is unique but the function does not satisfy the second condition
# from Condition 1


def theoritical_bounds(max_val, delta, k, radius, diameter, n, d):
    # Compute the theoritical bounds with probability 1-delta, independently from the function f (Corollary 13)
    return (
        max_val + k * radius * (delta / n) ** (1 / d),
        max_val + k * diameter * (np.log(1 / delta) / n) ** (1 / d),
    )


def fast_rates(max_val, delta, k, radius, diameter, n, d, kappa, c_kappa):
    # Compute the results from Theorem 15 and 16
    C_kk = (
        c_kappa * (diameter ** (kappa - 1)) / (8 * k)
    ) ** d  # Replacing max ||x-x*|| by diameter so we still have the upper bound
    if kappa == 1:
        ub = max_val + k * diameter * np.exp(
            -C_kk * n * np.log(2) / (np.log(n / delta) + 2 * (2 * np.sqrt(d)) ** d)
        )
    elif kappa > 1:
        ub = max_val + k * diameter * 2 ** (kappa - 1) * (
            1
            + C_kk
            * n
            * (2 ** (d * (kappa - 1)) - 1)
            / (np.log(n / delta) + 2 * (2 * np.sqrt(d)) ** d)
        ) ** (-kappa / (d * (kappa - 1)))
    else:
        ValueError("kappa must be greater than 1")
    lb = max_val + c_kappa * radius**kappa * np.exp(
        -kappa / d * (n + np.sqrt(2 * n * np.log(1 / delta)) + np.log(1 / delta))
    )
    return (lb, ub)


class LIPO_Statistics:
    """
    This class is used to store the statistics during the execution of LIPO or AdaLIPO.
    It can plot the results at the end of the execution.
    """

    def __init__(self, f, fig_path: str, stats, delta=0.05, k_hat=False):
        self.f = f
        self.delta = delta
        self.fig_path = fig_path

        self.k_hats = []
        self.max_vals = []
        self.naive_bounds = []
        self.LIPO_bounds = []
        self.nb_samples_vs_t = []

        self.t = 1

        if not k_hat:
            for max_val, nb_samples in stats:
                self.update(max_val, nb_samples)
        else:
            for max_val, nb_samples, k_hat in stats:
                self.update(max_val, nb_samples, k_hat=k_hat)

    def update(self, max_val, nb_samples, k_hat=None):
        if k_hat is not None:
            self.k_hats.append(k_hat)

        self.max_vals.append(max_val)

        self.naive_bounds.append(
            theoritical_bounds(
                max_val,
                self.delta,
                self.f.k,
                self.f.radius,
                self.f.diam,
                self.t,
                self.f.bounds.shape[0],
            )
        )

        if hasattr(self.f, "kappa"):
            self.LIPO_bounds.append(
                fast_rates(
                    max_val,
                    self.delta,
                    self.f.k,
                    self.f.radius,
                    self.f.diam,
                    self.t,
                    self.f.bounds.shape[0],
                    self.f.kappa,
                    self.f.c_kappa,
                )
            )

        self.nb_samples_vs_t.append(nb_samples)

        self.t += 1

    def plot(self):
        self.naive_bounds = np.array(self.naive_bounds)
        self.LIPO_bounds = np.array(self.LIPO_bounds)

        plt.figure(figsize=(10, 8))
        plt.style.use("seaborn-v0_8")
        plt.grid(True)

        plt.plot(self.max_vals, label="Max values")
        plt.plot(self.naive_bounds[:, 1], label="Naive upper bound")
        if len(self.LIPO_bounds) > 0:
            plt.plot(self.LIPO_bounds[:, 0], label="LIPO lower bound")
            plt.plot(self.LIPO_bounds[:, 1], label="LIPO upper bound")
        plt.legend()
        plt.savefig(f"{self.fig_path}/convergence.svg")
        plt.clf()

        plt.plot(self.nb_samples_vs_t)
        plt.xlabel("Number of evaluations")
        plt.ylabel("Number of draws")
        plt.savefig(f"{self.fig_path}/draws_vs_evaluations.svg", bbox_inches="tight")
        plt.clf()

        if len(self.k_hats) > 0:
            plt.plot(self.k_hats, label="$\hat{k}$")
            if hasattr(self.f, "k"):
                plt.hlines(
                    self.f.k,
                    0,
                    len(self.k_hats),
                    label="$k$",
                    color="r",
                    linestyles="dashed",
                )
            plt.legend()
            plt.savefig(f"{self.fig_path}/Lipschitz_estimation.svg")
            plt.clf()
        plt.style.use("default")
        plt.close()
