import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import resnet #import BasicBlock
import torchvision.models as models
from matplotlib import pyplot as plt
import numpy as np

def plot_kernels(tensor, num_cols=6):
    
    print tensor.shape
    tensor = tensor.reshape((512,3,3,512))
    tensor = tensor[:,:,:,:3]
    print tensor.shape
    if not tensor.ndim==4:
        raise Exception("assumes a 4D tensor")
    if not tensor.shape[-1]==3:
        raise Exception("last dim needs to be 3 to plot")
    num_kernels = tensor.shape[0]
    num_rows = 1+ num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    for i in range(tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        ax1.imshow(tensor[i])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()




model = torch.load("layer4-1000imgtest.pytorch")
weights = model.double()
body_model = [i for i in weights.children()]
layer = body_model[-3][1].conv2
avpool = body_model[-2]
print avpool.shape
tensor = layer.weight.data.cpu().numpy()
print tensor.shape
plot_kernels(tensor)
