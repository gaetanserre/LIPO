import torch

def Uniform(X, transform=None):
    '''
    This function generates a random point in the feasible region X. We assume that X is a subset of R^n 
    described by the inequalities X = {x in R^n | a_i <= x_i <= b_i, i = 0, ..., m-1} where a_i, b_i are given
    such that X[i,j] = [a_i, b_i] for i = 0, ..., m-1 and j = 0, 1.
    For simplicity, we assume that X C Rectangle given by an infinite norm (i.e. X = {x in R^n | -M <= x_i <= M, i = 1, ..., n}).
    X: feasible region (numpy array)
    transform: transformation to apply to the point (function). Default is None.
    '''

    theta = torch.rand(X.shape[0]) * (X[:,1] - X[:,0]) + X[:,0]
    return transform(theta) if transform is not None else theta


def AdaLIPO():
  pass