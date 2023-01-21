import numpy as np
import torch
import torchvision
import argparse

from random_search import random_search
from LIPO import LIPO
from mlp import MLP

def cli():
  args = argparse.ArgumentParser()
  args.add_argument("--image-size", type=int, default=16)
  args.add_argument("--hidden-dim", type=int, nargs="+", default=[])
  args.add_argument("--batch-size", type=int, default=32)
  args.add_argument("--data", type=str, default="data_geometric")
  return args.parse_args()

def LipCrossEntropyObj(input, target):
  def lip_log(x):
    return torch.log(torch.clamp(x, min=1e-2))

  return torch.mean(target * lip_log(input) + (1-target) * lip_log(1-input))

def hinge_obj(input, target):
  input = input * 2 - 1
  target = target * 2 - 1
  return -torch.mean(torch.clamp(1 - input * target, min=0))

def zero_one_obj(input, target):
  return -torch.mean(torch.abs(input - target))

if __name__ == "__main__":
  args = cli()

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("Device:", device)

  # Load dog and cat dataset
  transform = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Resize((args.image_size, args.image_size)),
      torchvision.transforms.Grayscale(num_output_channels=1),
  ])

  dataset = torchvision.datasets.ImageFolder(root=args.data, transform=transform)
  print("Dataset size:", len(dataset))

  # Split dataset into train and test
  train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [int(len(dataset) * 0.8),
    len(dataset) - int(len(dataset) * 0.8)]
  )

  # Create dataloader
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

  # Create model
  model = MLP(args.image_size**2, args.hidden_dim).to(device)

  # Black-box function to optimize
  def evaluate(theta, obj_f, loader=train_loader):
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
      output     = model.evaluate(data, theta.to(device))
      objective += obj_f(output, label).item()
    
    # Average objective over batches
    objective /= len(train_loader)    
    return objective
  
  def accuracy(loader, theta=None):
    """
    Function that computes the accuracy on the data given a set of parameters.
    `theta`: the set of parameters
    """
    correct = 0
    total = 0
    ones = 0
    zeros = 0
    # Evaluate objective on each batch
    for data, label in loader:
      data       = data.view(data.shape[0], -1).to(device)
      label      = label.to(device)
      output     = model.evaluate(data, theta) if theta is not None else model(data)
      predicted  = torch.round(output).view(-1)

      for pred in predicted:
        if pred == 1: ones+=1
        else: zeros+=1

      total += label.size(0)
      correct += (predicted == label).sum().item()
    
    print("circles: ", ones)
    print("rectangles: ", zeros)
    
    # Average objective over batches
    return correct / total

  # Run random search
  """ f = lambda theta: evaluate(theta, obj_f=LipCrossEntropyObj)
  best = random_search(f, model.get_num_params(), device, n_iter=100)
  print("Best objective random:", -best[0])
  print("Accuracy on train set:", accuracy(train_loader, best[1])) """

  # Run LIPO
  k = 1 #1 * np.sqrt(128) * np.sqrt(128) * 0.5**3 * 100
  X = torch.ones((model.get_num_params(), 2))
  X[:, 0] = -1
  transformation = None #lambda w: w / np.sqrt(w.shape[0])
  """ print(f"1/k = {1/k}")

  f = lambda theta: evaluate(theta, obj_f=LipCrossEntropyObj)
  best = LIPO(f=f, n=1000, k=k, X=X, transform=transformation)
  print("Best objective LIPO (cross):", -best[0].item())
  print("Accuracy on train set:", accuracy(train_loader, best[1].to(device))) """

  k = 0.07 #1 * np.sqrt(128) * np.sqrt(128) * 0.5**3 * 1
  print(f"1/k = {1/k}")
  f = lambda theta: evaluate(theta, obj_f=hinge_obj)
  best = LIPO(f=f, n=10000, k=k, X=X, transform=transformation)
  print("Best objective LIPO (hinge):", -best[0].item())
  print("Accuracy on train set:", accuracy(train_loader, best[1].to(device)))


  def CrossEntropyLoss(input, target):
    return -torch.mean(target * torch.log(input) + (1-target) * torch.log(1-input))

  # Gradient descent
  """ model = MLP(args.image_size**2, args.hidden_dim).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
  for i in range(100):
    l = 0
    for data, label in train_loader:
      data  = data.view(data.shape[0], -1).to(device)
      label = label.to(device)
      output = model(data)
      loss = CrossEntropyLoss(output, label.view(-1, 1).float())
      l += loss.item()
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    print("Epoch:", i, "Loss:", l / len(train_loader))
  print("Accuracy on train set:", accuracy(train_loader)) """



  # Evaluate best model on test set
  #test_obj = evaluate(best[1], loader=test_loader)
  #print("Test objective:", test_obj)