import numpy as np
import random as rd

def Uniform(X, M):
    '''
    This function generates a random point in the feasible region X. We assume that X is a subset of R^n 
    described by the inequalities X = {x in R^n | a_i <= x_i <= b_i, i = 0, ..., m-1} where a_i, b_i are given
    such that X[i,j] = [a_i, b_i] for i = 0, ..., m-1 and j = 0, 1.
    For simplicity, we assume that X C Rectangle given by an infinite norm (i.e. X = {x in R^n | -M <= x_i <= M, i = 1, ..., n}).
    X : feasible region (numpy array)
    M : maximum value of the coordinates of the feasible region (float)
    '''
    x = np.zeros(len(X))
    for i in range(len(X)):
        x_i = rd.uniform(-M, M)
        while X[i,0] * x_i > X[i,1]:
            x_i = rd.uniform(-M, M)
        x[i] = x_i
    return x
        
    

def LIPO(n, k, X, f, M):
    '''
    n: number of iterations (int)
    k: lipschitz constant (float)
    X: feasible region (numpy array)
    f: objective function (function)
    M: maximum value of the coordinates of the feasible region (float)
    '''
    
    # Initialization
    t = 1
    X_1 = Uniform(X, M)
    points = np.array([X_1])
    values = np.array([f(X_1)])
    def condition(x, values, k, points):
        '''
        Subfunction to check the condition in the loop, depending on the set of values we already have.
        values: set of values of the function we explored (numpy array)
        x: point to check (numpy array)
        k: lipschitz constant (float)
        points: set of points we explored (numpy array)
        '''
        t = len(values)
        max_val = np.max(values)
        left_array = values.copy()
        for i in range(t):
            left_array[i] += k * np.linalg.norm(x - points[i], ord=2)
        left_min = np.min(left_array)
        return left_min >= max_val
    
    max_fails = 100
        
    
    # Main loop
    while t < n:
        X_tp1 = Uniform(X, M)
        if condition(X_tp1, values, k, points):
            points = np.append(points, [X_tp1], axis=0)
            values = np.append(values, f(X_tp1))
            t += 1
            fails = 0
        elif fails < max_fails:
            fails += 1
        else:
            # Output because we have failed too many times in a row
            argmax_X = points[np.argmax(values)]
            return argmax_X
            
    # Output
    argmax_X = points[np.argmax(values)]
    return argmax_X

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