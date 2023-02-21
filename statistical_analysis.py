import numpy as np
import matplotlib.pyplot as plt

# Those convergence rates are supposed to be for LIPO. We can try to adapt them for AdaLIPO, but they won't make sense before 
# the Lipscitz constant is properly approximated.
# THOSE BOUNDS ARE CORRECT ONLY FOR UNIQUE MAXIMUMS, so in our examples, only for sphere, rastrigin, and square in our examples
# but not for himmelblau and holder table. For rosenbrock, the maximum is unique but the function does not satisfy the second condition
# from Condition 1

def theoritical_bounds_lipo(max_val, delta, k, radius, diameter, n, d):
    # Compute the theoritical bounds with probability 1-delta, independently from the function f (Corollary 13)
    return max_val + k*diameter*(np.log(1/delta)/n)**(1/d)

def fast_rates_lipo(max_val, delta, k, radius, diameter, n, d, kappa, c_kappa):
    # Compute the results from Theorem 15 and 16
    C_kk = (c_kappa * (diameter**(kappa - 1)) / (8*k))**d # Replacing max ||x-x*|| by diameter so we still have the upper bound
    if kappa == 1:
        ub = max_val + k*diameter*np.exp(-C_kk * n * np.log(2) / (np.log(n/delta) + 2*(2*np.sqrt(d))**d))
    elif kappa > 1:
        ub = max_val + k*diameter*2**(kappa - 1) * (1 + C_kk * n * (2**(d*(kappa-1)) - 1) / (np.log(n/delta) + 2*(2*np.sqrt(d))**d))**(-kappa/(d*(kappa - 1)))
    else:
        ValueError("kappa must be greater than 1")
    lb = max_val + c_kappa * radius ** kappa * np.exp(-kappa/d * (n + np.sqrt(2*n*np.log(1/delta)) + np.log(1/delta)))
    return (lb, ub)
  
def gamma(f, k, samples=1000):
  # Compute the coefficient gamma from Prop 19
  bounds = f.bounds
  d = bounds.shape[0]
  count = 0
  for i in range(samples):
    X_1 = np.zeros((d))
    X_2 = np.zeros((d))
    for j in range(d):
      X_1[j] = np.random.uniform(bounds[j, 0], bounds[j, 1])
      X_2[j] = np.random.uniform(bounds[j, 0], bounds[j, 1])
    if (f(X_1) - f(X_2))/np.linalg.norm(X_1 - X_2, ord=2) > k:
      count += 1
  proba = count/samples
  return proba

def theoricital_bounds_adalipo(f, max_val, delta, k_seq, i_star, radius, diameter, n, d, p):
    # Upper bound from Prop 24
    k_i_star = k_seq[i_star]
    k_i_star_minus_1 = k_seq[i_star - 1]
    return max_val + k_i_star*diameter*(5/p + (2*np.log(delta/3))/(p*np.log(1-gamma(f, k_i_star_minus_1))))**(1/d)*(np.log(3/delta)/n)**(1/d)

def fast_rates_adalipo(f, max_val, delta, k_seq, i_star, radius, diameter, n, d, p, kappa, c_kappa):
    # Compute the results from Theorem 25
    k_i_star = k_seq[i_star]
    k_i_star_minus_1 = k_seq[i_star - 1]
    C_k_i_star_kappa = (c_kappa * (diameter**(kappa - 1)) / (8*k_i_star))**d # Replacing max ||x-x*|| by diameter so we still have the upper bound
    grosse_constante = np.exp(2*np.log(delta/4)/(p*np.log(1-gamma(f, k_i_star_minus_1)))+7*np.log(4/delta)/(p*(1-p)**2))
    if kappa == 1:
        ub = max_val + grosse_constante*np.exp(-C_k_i_star_kappa*n*(1-p)*np.log(2)/(2*np.log(n/delta)+4*(2*np.sqrt(d))**d))
    elif kappa > 1:
        ub = max_val + 2**kappa*(1+C_k_i_star_kappa*n*(1-p)*(2**(d*(kappa-1))-1)/(2*np.log(n/delta)+4*(2*np.sqrt(d))**d))**(-kappa/(d*(kappa-1)))
    else:
        ValueError("kappa must be greater than 1")
    return ub
    

class LIPO_Statistics:
  """
  This class is used to store the statistics during the execution of LIPO or AdaLIPO.
  It can plot the results at the end of the execution.
  """

  def __init__(self, f, fig_path: str, delta=0.05, k_seq=None, optimizer="LIPO"):
    self.f = f
    self.delta = delta
    self.fig_path = fig_path
    
    self.k_hats = []
    self.max_vals = []
    self.naive_bounds = []
    self.LIPO_bounds = []
    self.nb_samples_vs_t = []
    self.k_seq = k_seq
    self.optimizer = optimizer

    self.t = 1

  def update(
    self,
    max_val,
    nb_samples,
    k_hat=None,
    p=None
  ):
    if self.optimizer == "LIPO":
      if k_hat is not None:
        self.k_hats.append(k_hat)

      self.max_vals.append(max_val)

      self.naive_bounds.append(theoritical_bounds_lipo(
        max_val,
        self.delta,
        self.f.k,
        self.f.radius,
        self.f.diam,
        self.t,
        self.f.bounds.shape[0]
      ))

      if hasattr(self.f, "kappa"):
        self.LIPO_bounds.append(fast_rates_lipo(
          max_val,
          self.delta,
          self.f.k,
          self.f.radius,
          self.f.diam,
          self.t,
          self.f.bounds.shape[0],
          self.f.kappa,
          self.f.c_kappa
        ))

      self.nb_samples_vs_t.append(nb_samples)

      self.t += 1
    elif self.optimizer == "AdaLIPO":
      if k_hat is not None:
        self.k_hats.append(k_hat)

      self.max_vals.append(max_val)
      
      self.i_star = np.argwhere(np.array(self.k_seq) >= k_hat)[0][0]
      # print(self.i_star)
    
      self.naive_bounds.append(theoricital_bounds_adalipo(
        self.f,
        max_val,
        self.delta,
        self.k_seq,
        self.i_star,
        self.f.radius,
        self.f.diam,
        self.t,
        self.f.bounds.shape[0],
        p
      ))

      if hasattr(self.f, "kappa"):
        self.LIPO_bounds.append(fast_rates_adalipo(
          self.f,
          max_val,
          self.delta,
          self.k_seq,
          self.i_star,
          self.f.radius,
          self.f.diam,
          self.t,
          self.f.bounds.shape[0],
          p,
          self.f.kappa,
          self.f.c_kappa
        ))

      self.nb_samples_vs_t.append(nb_samples)

      self.t += 1

  def plot(self):

    self.naive_bounds = np.array(self.naive_bounds)
    self.LIPO_bounds = np.array(self.LIPO_bounds)

    plt.figure(figsize=(10, 8))
    plt.style.use("seaborn-v0_8")
    plt.grid(True)

    """ plt.plot(self.max_vals, label="Max values")
    plt.plot(self.naive_bounds[:, 0], label="Naive lower bound")
    plt.plot(self.naive_bounds[:, 1], label="Naive upper bound")
    if len(self.LIPO_bounds) > 0:
      plt.plot(self.LIPO_bounds[:, 0], label="LIPO lower bound")
      plt.plot(self.LIPO_bounds[:, 1], label="LIPO upper bound")
    elif len(self.LIPO_bounds) > 0 and self.optimizer == "AdaLIPO":
      plt.plot(self.LIPO_bounds, label="AdaLIPO upper bound")
    plt.legend()
    # plt.ylim(-100, 1000)
    plt.savefig(f"{self.fig_path}/convergence.pdf")
    plt.clf()
    
    plt.plot(self.nb_samples_vs_t)
    plt.xlabel("Number of evaluations")
    plt.ylabel("Number of samples")
    plt.savefig(f"{self.fig_path}/samples_vs_evaluations.pdf")
    plt.clf() """

    if len(self.k_hats) > 0:
      plt.plot(self.k_hats, label="$\hat{k}$")
      if hasattr(self.f, 'k'):
        plt.hlines(self.f.k, 0, len(self.k_hats), label="$k$", color='r', linestyles="dashed")
      plt.legend()
      plt.savefig(f"{self.fig_path}/Lipschitz_estimation.pdf")
      plt.clf()
    plt.style.use("default")
    plt.close()
  