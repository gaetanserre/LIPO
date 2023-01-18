import numpy as np
import torch

def Uniform(X):
    '''
    This function generates a random point in the feasible region X. We assume that X is a subset of R^n 
    described by the inequalities X = {x in R^n | a_i <= x_i <= b_i, i = 0, ..., m-1} where a_i, b_i are given
    such that X[i,j] = [a_i, b_i] for i = 0, ..., m-1 and j = 0, 1.
    For simplicity, we assume that X C Rectangle given by an infinite norm (i.e. X = {x in R^n | -M <= x_i <= M, i = 1, ..., n}).
    X : feasible region (numpy array)
    '''

    theta = torch.rand(X.shape[0]) * (X[:,1] - X[:,0]) + X[:,0]
    theta /= np.sqrt(X.shape[0])

    return theta
        
    

def LIPO(n, k, X, f, max_fails=10):
    '''
    n: number of iterations (int)
    k: Lipschitz constant (float)
    X: feasible region (numpy array)
    f: objective function (function)
    device: device on which we run the code (torch.device)
    '''
    
    # Initialization
    t = 1
    X_1 = Uniform(X)
    points = X_1.unsqueeze(0)
    value = f(X_1)
    print(f"{t}: {-value}")
    values = torch.tensor([value])
    def condition(x, values, k, points):
        '''
        Subfunction to check the condition in the loop, depending on the set of values we already have.
        values: set of values of the function we explored (numpy array)
        x: point to check (numpy array)
        k: Lipschitz constant (float)
        points: set of points we explored (numpy array)
        '''
        max_val = torch.max(values)

        left_min = torch.min(values + k * torch.linalg.norm(x - points, ord=2, dim=1))
        return left_min >= max_val
            
    
    # Main loop
    while t < n:
        X_tp1 = Uniform(X)
        if condition(X_tp1, values, k, points):
            points = torch.cat((points, X_tp1.unsqueeze(0)))

            value = f(X_tp1)
            values = torch.cat((values, torch.tensor([value])))
            t += 1
            fails = 0
            print(f"{t}: {-value}")
        elif fails < max_fails:
            fails += 1
            print("Fail")
        else:
            print("Too many fails in a row")
            break
            # Output because we have failed too many times in a row
            
    # Output
    max_idx = torch.argmax(values).item()
    return (values[max_idx], points[max_idx])

def main():
    # Example
    n = 100
    k = 3
    X = np.array([[-1, 1], [-1, 1], [-1, 1]]) # X = {x in R^3 | -1<= x_1 <=1, -1 <= x_2 <= 1, -1 <= x_3 <= 1}
    f = lambda x: x[0] - x[1]**2 + x[2]**3 # f(x) = x_1 - x_2^2 + x_3^3
    M = 1
    print(LIPO(n, k, X, f, M))
    
if __name__ == "__main__":
    main()