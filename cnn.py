#
# Created in 2023 by Gaëtan Serré
#

import torch.nn as nn
import torch
import numpy as np

class CNN(nn.Module):
  def __init__(self, input_chan, hidden_dim):
    super(CNN, self).__init__()
    
    self.conv1 = nn.Conv2d(input_chan, 32, 3, 1)
    self.conv2 = nn.Conv2d(32, 64, 3, 1)
    self.conv3 = nn.Conv2d(64, 64, 3, 1)

    self.fc = nn.Linear(53824, 1)

  def forward(self, x):
    #x /= np.sqrt(x.shape[0])

    x = self.conv1(x)
    x = nn.ReLU()(x)
    x = self.conv2(x)
    x = nn.ReLU()(x)
    x = self.conv3(x)
    x = nn.ReLU()(x)
    x = nn.MaxPool2d(2)(x)

    x = self.fc(nn.Flatten()(x))
    return nn.Sigmoid()(x)
  
  def get_shapes(self):
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

