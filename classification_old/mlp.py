#
# Created in 2023 by Gaëtan Serré
#

import torch.nn as nn
import torch
import numpy as np

class MLP(nn.Module):
  def __init__(self, input_dim, hidden_dim):
    super(MLP, self).__init__()
    
    layers = []
    hidden_dim.insert(0, input_dim)
    for i in range(len(hidden_dim)-1):
      if i == 0:
        layers.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
      else:
        layers.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
      layers.append(nn.Tanh())
    layers.append(nn.Linear(hidden_dim[-1], 1))

    self.layers = nn.Sequential(*layers)

    """ for param in self.parameters():
      param.requires_grad = False """

  def forward(self, x):
    x /= np.sqrt(x.shape[0])

    x = self.layers(x)
    return nn.Sigmoid()(x)
  
  def get_num_params(self):
    shapes = 0
    for param in self.parameters():
      shapes += torch.numel(param)
    return shapes
  
  def evaluate(self, x, theta):
    offset = 0
    for param in self.parameters():
      nb_params = torch.numel(param)
      param.data = theta[offset:offset+nb_params].view(param.shape)
      offset += nb_params

    return self.forward(x)

