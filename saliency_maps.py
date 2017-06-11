import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import os

import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import resnet #import BasicBlock
import torchvision.models as models
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

model = torch.load("layer4-1000imgtest.pytorch")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make sure the model is in "test" mode
    model.eval()
    
    # Wrap the input tensors in Variables
    X_var = Variable(X, requires_grad=True)
    y_var = Variable(y)
    saliency = None
    ##############################################################################
    # TODO: Implement this function. Perform a forward and backward pass through #
    # the model to compute the gradient of the correct class score with respect  #
    # to each input image. You first want to compute the loss over the correct   #
    # scores, and then compute the gradients with a backward pass.               #
    ##############################################################################
    scores = model.cpu()(X_var)
    correct_scores = scores.gather(1, y_var.view(-1, 1)).squeeze()
    sum_score = sum(correct_scores)
    sum_score.backward()
    saliency = X_var.grad.data.abs().max(dim=1)[0].squeeze()
    ##############################################################################
    #                             END OF YOUR CODE                               #
    ##############################################################################
    return sum(sum(saliency))

def show_saliency_maps(X, y):
    # Convert X and y from numpy arrays to Torch Tensors
    X_tensor = X
    #X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0)
    y_tensor = torch.LongTensor(y)

    # Compute saliency maps for images in X
    saliency = compute_saliency_maps(X_tensor, y_tensor, model)
    return saliency
    # Convert the saliency map from Torch Tensor to numpy array and show images
    # and saliency maps together.
    saliency = np.array([saliency.numpy()])
    print saliency.shape
    print X.size()
    N = X.size()[0]
    for i in range(N):
        print (Variable(X[i]).data).cpu().numpy().shape
        plt.subplot(2, 1, 1)
        #plt.imshow((Variable(X[i]).data).cpu().numpy().reshape((224,224,3)))
        plt.imshow(np.asarray(deprocess(X)))
        plt.axis('off')
        plt.title('low')
        plt.subplot(2, N, N + i + 1)
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(12, 5)
    plt.show()


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

def deprocess(img, should_rescale=True):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=(1.0 / np.array(IMAGENET_STD))),
        T.Normalize(mean=-np.array(IMAGENET_MEAN), std=[1, 1, 1]),
        T.Lambda(rescale) if should_rescale else T.Lambda(lambda x: x),
        T.ToPILImage(),
    ])
    return transform(img)

def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled
    
def blur_image(X, sigma=1):
    X_np = X.cpu().clone().numpy()
    X_np = gaussian_filter1d(X_np, sigma, axis=2)
    X_np = gaussian_filter1d(X_np, sigma, axis=3)
    X.copy_(torch.Tensor(X_np).type_as(X))
    return X

def getKey(item):
  return item[1]

#data = np.asarray(img,dtype="int32")
low_files = os.listdir('images_50000/val/low/')
best_imgs = []
best_val = 0
for f in low_files:
  img = Image.open("images_50000/val/low/"+f)
  img.load()
  X= preprocess(img)
  y = np.array([2])
  val = show_saliency_maps(X, y)
  best_imgs.append((f,val))

s= sorted(best_imgs,key=getKey,reverse=True)
print s[:10]
