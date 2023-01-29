import numpy as np

# Those convergence rates are supposed to be for LIPO. We can try to adapt them for AdaLIPO, but they won't make sense before 
# the Lipscitz constant is properly approximated.
# THOSE BOUNDS ARE CORRECT ONLY FOR UNIQUE MAXIMUMS, so in our examples, only for sphere, rastrigin, and square in our examples
# but not for himmelblau and holder table. For rosenbrock, the maximum is unique but the function does not satisfy the second condition
# from Condition 1

def theoritical_bounds(max_val, delta, k, radius, diameter, n, d):
    # Compute the theoritical bounds with probability 1-delta, independently from the function f (Corollary 13)
    return [max_val + k*radius*(delta/n)**(1/d), max_val + k*diameter*(np.log(1/delta)/n)**(1/d)]

def fast_rates(max_val, delta, k, radius, diameter, n, d, kappa, c_kappa):
    # Compute the results from Theorem 15 and 16
    C_kk = (c_kappa * (diameter**(kappa - 1)) / (8*k))**d # Replacing max ||x-x*|| by diameter so we still have the upper bound
    if kappa == 1:
        ub = max_val + k*diameter*np.exp(-C_kk * n * np.log(2) / (np.log(n/delta) + 2*(2*np.sqrt(d))**d))
    elif kappa > 1:
        ub = max_val + k*diameter*2**(kappa - 1) * (1 + C_kk * n * (2**(d*(kappa-1)) - 1) / (np.log(n/delta) + 2*(2*np.sqrt(d))**d))**(-kappa/(d*(kappa - 1)))
    else:
        ValueError("kappa must be greater than 1")
    lb = max_val + c_kappa * radius ** kappa * np.exp(-kappa/d * (n + np.sqrt(2*n*np.log(1/delta)) + np.log(1/delta)))
    return [lb, ub]
    


    