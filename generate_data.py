import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def generate_data(n_rec, n_cir, im_size, root):
  """
  Generates a dataset of images containing random circles and rectangles.
  `n_rec`: number of rectangles (int)
  `n_cir`: number of circles (int)
  `im_size`: size of the image (int)
  `root`: root directory to save the images (str)
  """

  # Create directory to save images
  if not os.path.exists(root):
    os.makedirs(root)
  if not os.path.exists(root + "/rectangle"):
    os.makedirs(root + "/rectangle")
  if not os.path.exists(root + "/circle"):
    os.makedirs(root + "/circle")

  # Draw rectangles
  for i in range(n_rec):
    # Create a black image
    img = np.zeros((im_size, im_size, 3), dtype = "uint8")

    # Draw a random rectangle
    x, y = np.random.randint(0, im_size-1, 2)
    wx = np.random.randint(1, im_size-x)
    wy = np.random.randint(1, im_size-y)

    # Creating random rectangle
    rectangle = cv2.rectangle(img, (x, y), (x+wx, y+wy), (255, 255, 255), 1)

    # Save image
    cv2.imwrite(root + "/rectangle/" + str(i) + ".png", rectangle)
  
  # Draw circles
  for i in range(n_cir):
    # Create a black image
    img = np.zeros((im_size, im_size, 3), dtype = "uint8")
    
    # Draw a random circle
    x, y = np.random.randint(1, im_size-1, 2)
    r = np.random.randint(1, min([x, (im_size-1-x), y, (im_size-1-y)])+1)

    # Creating random circle
    circle = cv2.circle(img, (x, y), r, (255, 255, 255), 1)

    # Save image
    cv2.imwrite(root + "/circle/" + str(i) + ".png", circle)

if __name__ == "__main__":
  generate_data(100, 100, 16, "data_geometric")