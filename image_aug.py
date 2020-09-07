##
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms
from dataset import *

##
data_dir = "./datasets"

transform_raw = transforms.Compose([])
dataset_train_raw = dataset(data_dir=os.path.join(data_dir, "train.csv"))

# transform = transforms.Compose([Noise(), ToPILImage(), RandomAffine(30), RandomCrop(), ToNumpy()])
# dataset_train = dataset(data_dir=os.path.join(data_dir, "train.csv"), transform=transform)

transform = transforms.Compose([ToPILImage(), RandomPerspective(p=1, scale=0.5), ToNumpy()])
dataset_train = dataset(data_dir=os.path.join(data_dir, "train.csv"), transform=transform)

##
data_raw = dataset_train_raw.__getitem__(1)
img_raw = data_raw['img']
img_raw = img_raw.squeeze()

data = dataset_train.__getitem__(1)
img = data['img']
img = img.squeeze()

##
plt.subplot(121)
plt.imshow(img_raw, cmap='gray')
plt.title("raw")

plt.subplot(122)
plt.imshow(img, cmap='gray')
plt.title("trans")
plt.show()
##

