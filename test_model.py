import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder


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
FILENAME = "misclassified-test.txt"

def main(args):
  # Figure out the datatype we will use; this will determine whether we run on
  # CPU or on GPU. Run on GPU by adding the command-line flag --use_gpu
  dtype = torch.FloatTensor
  if args.use_gpu:
    dtype = torch.cuda.FloatTensor
  val_transform = T.Compose([
    T.Scale(224),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
  ])
  val_dset = ImageFolder(args.val_dir, transform=val_transform)
  val_loader = DataLoader(val_dset,
                  batch_size=args.batch_size,
                  num_workers=args.num_workers, shuffle=False)

  # Now that we have set up the data, it's time to set up the model.
  # For this example we will finetune a ResNet-18 model which has been
  # pretrained on ImageNet. We will first reinitialize the last layer of the
  # model, and train only the last layer for a few epochs. We will then finetune
  # the entire model on our dataset for a few more epochs.

  # First load the pretrained ResNet-18 model; this will download the model
  # weights from the web the first time you run it.
  model = torch.load('layer3-4_50k-6-10.pytorch')

  val_acc = check_accuracy(model, val_dset, val_loader, dtype, True, FILENAME)
  print('Val accuracy: ', val_acc)
  print()


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


def check_accuracy(model, folder, loader, dtype, write_bool, filename):
  """
  Check the accuracy of the model.
  """
  # Set the model to eval mode
  f = None
  if write_bool:
    f = open(filename, "w+")
  model.eval()
  num_correct, num_samples = 0, 0
  wrong_set = None
  batch_num = 0
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
    mask = (preds == y)
    mins, indices = torch.min(mask,0)
    for i in range(x.size(0)):
      path, true_class = folder.imgs[batch_num * args.batch_size + i]
      pred_class =  preds[i][0]
      if write_bool:
        f.write("%s\t%d\t%d\n" % (path, true_class, pred_class))
    num_samples += x.size(0)
    batch_num += 1
  # Return the fraction of datapoints that were correctly classified.
  acc = float(num_correct) / num_samples
  if write_bool:
    f.close()
  return acc


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)
