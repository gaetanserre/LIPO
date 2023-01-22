import matplotlib.pyplot as plt
import numpy as np

class FigGenerator:
  def __init__(self, f):
    """
    Class to generate figures for visualizing the optimization process
    f: function optimized (Function)
    """
    
    self.f = f
  
  def gen_figure(self, eval_points: np.array, eval_values: np.array, method: str, path: str = None):
    """
    Generates a figure
    path: path to save the figure (str) (optional)
    method: name of the method used (str)
    """

    dim = eval_points.shape[1]
    if dim == 1:
      self.gen_1D(eval_points, eval_values, method)
    elif dim == 2:
      self.gen_2D(eval_points, eval_values, method)
    else:
      raise ValueError(f"Cannot generate a figure for {dim}-dimensional functions")
    
    if path is not None:
      plt.savefig(path)
    else:
      plt.show()
    plt.clf()
  
  def gen_1D(self, eval_points, eval_values, method):
    """
    Generates a figure for 1D functions
    """

    x = np.linspace(self.f.bounds[0][0], self.f.bounds[0][1], 100)
    y = self.f(x)

    plt.plot(x, y)
    plt.scatter(eval_points, eval_values, c=eval_values, label="evaluations", cmap="viridis", zorder=2)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.plot(eval_points, eval_values, linewidth=0.5, color="black")
    plt.xlabel("$X$")
    plt.ylabel("$f(x)$")
    plt.legend()
  
  def gen_2D(self, eval_points, eval_values, method):
    """
    Generates a figure for 2D functions
    """

    x = np.linspace(self.f.bounds[0][0], self.f.bounds[0][1], 1000)
    y = np.linspace(self.f.bounds[1][0], self.f.bounds[1][1], 1000)
    x, y = np.meshgrid(x, y)
    z = self.f([x, y])

    fig = plt.figure(figsize=(15, 15))
    ax  = plt.axes(projection="3d", computed_zorder=False)

    ax.plot_surface(x, y, z, cmap="coolwarm", linewidth=0, antialiased=True, zorder=4.4)

    cb = ax.scatter(
      eval_points[:, 0],
      eval_points[:, 1],
      eval_values,
      c=eval_values,
      label="evaluations",
      cmap="viridis",
      zorder=4.5
    )

    plt.colorbar(cb, fraction=0.046, pad=0.04)

    ax.set_xlabel("$X$", fontsize=15)
    ax.set_ylabel("$Y$", fontsize=15)
    ax.legend(fontsize=15)