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
"""
Example PyTorch script for finetuning a ResNet model on your own data.
For this example we will use a tiny dataset of images from the COCO dataset.
We have chosen eight types of animals (bear, bird, cat, dog, giraffe, horse,
sheep, and zebra); for each of these categories we have selected 100 training
images and 25 validation images from the COCO dataset. You can download and
unpack the data (176 MB) by running:
wget cs231n.stanford.edu/coco-animals.zip
unzip coco-animals.zip
rm coco-animals.zip
The training data is stored on disk; each category has its own folder on disk
and the images for that category are stored as .jpg files in the category folder.
In other words, the directory structure looks something like this:
coco-animals/
  train/
    bear/
      COCO_train2014_000000005785.jpg
      COCO_train2014_000000015870.jpg
      [...]
    bird/
    cat/
    dog/
    giraffe/
    horse/
    sheep/
    zebra/
  val/
    bear/
    bird/
    cat/
    dog/
    giraffe/
    horse/
    sheep/
    zebra/
"""

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='coco-animals/train')
parser.add_argument('--val_dir', default='coco-animals/val')
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_epochs1', default=10, type=int)
parser.add_argument('--num_epochs2', default=10, type=int)
parser.add_argument('--use_gpu', action='store_true')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         return out

def plot_kernels(tensor, num_cols=6):
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

def main(args):
  # Figure out the datatype we will use; this will determine whether we run on
  # CPU or on GPU. Run on GPU by adding the command-line flag --use_gpu
  dtype = torch.FloatTensor
  if args.use_gpu:
    dtype = torch.cuda.FloatTensor

  # Use the torchvision.transforms package to set up a transformation to use
  # for our images at training time. The train-time transform will incorporate
  # data augmentation and preprocessing. At training time we will perform the
  # following preprocessing on our images:
  # (1) Resize the image so its smaller side is 256 pixels long
  # (2) Take a random 224 x 224 crop to the scaled image
  # (3) Horizontally flip the image with probability 1/2
  # (4) Convert the image from a PIL Image to a Torch Tensor
  # (5) Normalize the image using the mean and variance of each color channel
  #     computed on the ImageNet dataset.
  train_transform = T.Compose([
    T.Scale(256),
    T.RandomSizedCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),            
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
  ])
  
  # You load data in PyTorch by first constructing a Dataset object which
  # knows how to load individual data points (images and labels) and apply a
  # transform. The Dataset object is then wrapped in a DataLoader, which iterates
  # over the Dataset to construct minibatches. The num_workers flag to the
  # DataLoader constructor is the number of background threads to use for loading
  # data; this allows dataloading to happen off the main thread. You can see the
  # definition for the base Dataset class here:
  # https://github.com/pytorch/pytorch/blob/master/torch/utils/data/dataset.py
  #
  # and you can see the definition for the DataLoader class here:
  # https://github.com/pytorch/pytorch/blob/master/torch/utils/data/dataloader.py#L262
  #
  # The torchvision package provides an ImageFolder Dataset class which knows
  # how to read images off disk, where the image from each category are stored
  # in a subdirectory.
  #
  # You can read more about the ImageFolder class here:
  # https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
  train_dset = ImageFolder(args.train_dir, transform=train_transform)
  train_loader = DataLoader(train_dset,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    shuffle=True)

  # Set up a transform to use for validation data at test-time. For validation
  # images we will simply resize so the smaller edge has 224 pixels, then take
  # a 224 x 224 center crop. We will then construct an ImageFolder Dataset object
  # for the validation data, and a DataLoader for the validation set.
  val_transform = T.Compose([
    T.Scale(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
  ])
  val_dset = ImageFolder(args.val_dir, transform=val_transform)
  val_loader = DataLoader(val_dset,
                  batch_size=args.batch_size,
                  num_workers=args.num_workers)

  # Now that we have set up the data, it's time to set up the model.
  # For this example we will finetune a ResNet-18 model which has been
  # pretrained on ImageNet. We will first reinitialize the last layer of the
  # model, and train only the last layer for a few epochs. We will then finetune
  # the entire model on our dataset for a few more epochs.

  # First load the pretrained ResNet-18 model; this will download the model
  # weights from the web the first time you run it.
  model = torchvision.models.resnet18(pretrained=True)
  print model.layer4#.shape

  # Reinitialize the last layer of the model. Each pretrained model has a
  # slightly different structure, but from the ResNet class definition
  # we see that the final fully-connected layer is stored in model.fc:
  # https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L111
  num_classes = len(train_dset.classes)
  model.fc = nn.Linear(model.fc.in_features, num_classes)
  ## arg3 = num basic blocks 
  model.inplanes = 256
  model.layer4 = model._make_layer(resnet.BasicBlock, 512, 2, stride=2) 
  print "--------------"
  print model.layer4

  # = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
  #                             bias=False)

  # Cast the model to the correct datatype, and create a loss function for
  # training the model.
  model.type(dtype)
  loss_fn = nn.CrossEntropyLoss().type(dtype)

  # First we want to train only the reinitialized last layer for a few epochs.
  # During this phase we do not need to compute gradients with respect to the
  # other weights of the model, so we set the requires_grad flag to False for
  # all model parameters, then set requires_grad=True for the parameters in the
  # last layer only.
  for param in model.parameters():
    param.requires_grad = False
  for param in model.fc.parameters():
    param.requires_grad = True
  for param in model.layer4.parameters():
    param.requires_grad = True

  # Construct an Optimizer object for updating the last layer only.
  optimizer = torch.optim.Adam(
    [{'params': model.fc.parameters()}, 
    {'params': model.layer4.parameters()}], lr=1e-3)
    #model.fc.parameters(), model.layer3.parameters(), 
  #optimizerl3 = torch.optim.Adam(model.layer3.parameters(), lr=1e-3)


  # Update only the last layer for a few epochs.
  for epoch in range(args.num_epochs1):
    # Run an epoch over the training data.
    print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs1))
    run_epoch(model, loss_fn, train_loader, optimizer, dtype)

    # Check accuracy on the train and val sets.
    train_acc = check_accuracy(model, train_loader, dtype)
    val_acc = check_accuracy(model, val_loader, dtype)
    print('Train accuracy: ', train_acc)
    print('Val accuracy: ', val_acc)
    print()

  # Now we want to finetune the entire model for a few epochs. To do thise we
  # will need to compute gradients with respect to all model parameters, so
  # we flag all parameters as requiring gradients.
  for param in model.parameters():
    param.requires_grad = True
  
  # Construct a new Optimizer that will update all model parameters. Note the
  # small learning rate.
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

  # Train the entire model for a few more epochs, checking accuracy on the
  # train and validation sets after each epoch.
  for epoch in range(args.num_epochs2):
    print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs2))
    run_epoch(model, loss_fn, train_loader, optimizer, dtype)

    train_acc = check_accuracy(model, train_loader, dtype)
    val_acc = check_accuracy(model, val_loader, dtype)
    print('Train accuracy: ', train_acc)
    print('Val accuracy: ', val_acc)
    print()

  torch.save(model, './layer4-1000imgtest.pytorch')
  weights = model.double()
  body_model = [i for i in weights.children()][0]
  layer1 = body_model[0]
  tensor = layer1.weight.data.numpy()

  plot_kernels(tensor)


def run_epoch(model, loss_fn, loader, optimizer, dtype):
  """
  Train the model for one epoch.
  """
  # Set the model to training mode
  model.train()
  for x, y in loader:
    # The DataLoader produces Torch Tensors, so we need to cast them to the
    # correct datatype and wrap them in Variables.
    #
    # Note that the labels should be a torch.LongTensor on CPU and a
    # torch.cuda.LongTensor on GPU; to accomplish this we first cast to dtype
    # (either torch.FloatTensor or torch.cuda.FloatTensor) and then cast to
    # long; this ensures that y has the correct type in both cases.
    x_var = Variable(x.type(dtype))
    y_var = Variable(y.type(dtype).long())

    # Run the model forward to compute scores and loss.
    scores = model(x_var)
    loss = loss_fn(scores, y_var)

    # Run the model backward and take a step using the optimizer.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def check_accuracy(model, loader, dtype):
  """
  Check the accuracy of the model.
  """
  # Set the model to eval mode
  model.eval()
  num_correct, num_samples = 0, 0
  for x, y in loader:
    # Cast the image data to the correct type and wrap it in a Variable. At
    # test-time when we do not need to compute gradients, marking the Variable
    # as volatile can reduce memory usage and slightly improve speed.
    x_var = Variable(x.type(dtype), volatile=True)

    # Run the model forward, and compare the argmax score with the ground-truth
    # category.
    scores = model(x_var)
    _, preds = scores.data.cpu().max(1)
    num_correct += (preds == y).sum()
    num_samples += x.size(0)

  # Return the fraction of datapoints that were correctly classified.
  acc = float(num_correct) / num_samples
  return acc


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
