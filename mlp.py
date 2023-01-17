#
# Created in 2023 by Gaëtan Serré
#

import torch.nn as nn

class MLP(nn.Module):
  def __init__(self, input_dim, hidden_dim):
    super(MLP, self).__init__()
    
    layers = []
    for i in range(len(hidden_dim)):
      if i == 0:
        layers.append(nn.Linear(input_dim, hidden_dim[i]))
      else:
        layers.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))
      layers.append(nn.Sigmoid())
    layers.append(nn.Linear(hidden_dim[-1], 1))

    self.layers = nn.Sequential(*layers)

    for param in self.parameters():
      param.requires_grad = False

  def forward(self, x):
    x = self.layers(x)
    return nn.Sigmoid()(x)
  
  def get_shapes(self):
    shapes = []
    for param in self.parameters():
      shapes.append(param.shape)
    return shapes
  
  def evaluate(self, x, theta):
    for param, t in zip(self.parameters(), theta):
      param.data = t
    return self.forward(x)

