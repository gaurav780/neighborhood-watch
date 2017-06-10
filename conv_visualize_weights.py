from math import sqrt, ceil
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def preprocess(img, size=224):
    transform = T.Compose([
        T.Scale(size),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN,
                    std=IMAGENET_STD),
        T.Lambda(lambda x: x[None]),
    ])
    return transform(img)



def visualize_grid(Xs, ubound=255.0, padding=1):
    """
    Reshape a 4D tensor of image data to a grid for easy visualization.

    Inputs:
    - Xs: Data of shape (N, H, W, C)
    - ubound: Output grid will have values scaled to the range [0, ubound]
    - padding: The number of blank pixels between elements of the grid
    """
    (N, C, H, W) = Xs.shape
    print Xs.shape
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    print grid.shape
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
		print img
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                # grid[y0:y1, x0:x1] = Xs[next_idx]
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    # grid_max = np.max(grid)
    # grid_min = np.min(grid)
    # grid = ubound * (grid - grid_min) / (grid_max - grid_min)
    return grid

modelpath='layer4-1000imgtest.pytorch'

model = torch.load(modelpath)
im = 'images_1000/val/low/37.791331_-122.215025_60.000000.png'
img = Image.open(im)

img.load()
'''print(model.layer4)
print '\n'
conv1 = list(model.layer4[1].conv1.parameters())[0]
print('conv1 size')
conv1_array = ((conv1.data).cpu().numpy())'''
img_final = preprocess(img)
res = model.conv1.forward(img_final)
grid = visualize_grid(conv1_array) #CHANGE TO CONVLAYER WE WANT
plt.imshow(grid.astype('uint8'))
plt.axis('off')
plt.gcf().set_size_inches(5, 5)
plt.show()
