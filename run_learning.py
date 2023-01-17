#
# Created in 2023 by Gaëtan Serré
#

import torch
import torchvision
import argparse

from random_search import random_search
from LIPO import LIPO
from mlp import MLP

def cli():
  args = argparse.ArgumentParser()
  args.add_argument("--image-size", type=int, default=128)
  args.add_argument("--hidden-dim", type=int, nargs="+", default=[128, 128])
  args.add_argument("--batch-size", type=int, default=32)
  args.add_argument("--data", type=str, default="data")
  return args.parse_args()

def LipCrossEntropyObj(input, target):
  def lip_log(x):
    return torch.log(torch.clamp(x, min=1e-1))

  return torch.mean(target * lip_log(input) + (1-target) * lip_log(1-input))
  

if __name__ == "__main__":
  args = cli()

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("Device:", device)

  # Load dog and cat dataset
  transform = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Resize((args.image_size, args.image_size)),
  ])

  dataset = torchvision.datasets.ImageFolder(root=args.data, transform=transform)
  print("Dataset size:", len(dataset))

  # Split dataset into train and test
  train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [int(len(dataset)*0.05),
    len(dataset)-int(len(dataset)*0.05)]
  )

  # Create dataloader
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

  # Create model
  model = MLP(3*args.image_size**2, args.hidden_dim).to(device)

  # Black-box function to optimize
  def evaluate(theta, loader=train_loader):
    """
    Function that computes the objective on the data given a set of parameters.
    We want to optimize this function.
    `theta`: the set of parameters
    """
    objective = 0
    # Evaluate objective on each batch
    for data, label in loader:
      data       = data.view(data.shape[0], -1).to(device)
      label      = label.to(device)
      output     = model.evaluate(data, theta)
      objective += LipCrossEntropyObj(output, label).item()
    
    # Average objective over batches
    objective /= len(train_loader)    
    return objective

  # Run random search
  #best = random_search(evaluate, model.get_shapes(), device, n_iter=10)
  #print("Best objective:", best[0])

  # Run LIPO
  import numpy as np

  k = np.sqrt(128**2*3) * np.sqrt(128)**2 * 0.5**3 * 10

  [((128**2*3, 128), -1, 1), (128, 128), (128, 1)]

  X = [(shape, -1, 1) for shape in model.get_shapes()]
  print(X)
  
  best = LIPO(n=10, k=k, X=X, f=evaluate, M=1, device=device)
  print("Best objective:", best[0])

  # Evaluate best model on test set
  #test_obj = evaluate(best[1], loader=test_loader)
  #print("Test objective:", test_obj)