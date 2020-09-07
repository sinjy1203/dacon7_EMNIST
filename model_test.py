## 패키지
import os
import numpy as np
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms

from model import *
from dataset import *
from util import *

## parameter
batch_size = 32

noise_std = 0.07
affine_degree = 30
rotation_degree = 30
perspective_scale = 0.5

aug_p = 0.5

data_dir = "./저장파일/datasets"
ckpt_dir = "./checkpoint"
log_dir = "./log"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## dataloader
transform_train = transforms.Compose([Noise(noise_std, p=aug_p), ToPILImage(), RandomPerspective(p=aug_p, scale=perspective_scale),
                                      RandomRotation(rotation_degree, p=aug_p), RandomAffine(affine_degree, p=aug_p),
                                      RandomCrop(p=aug_p), ToNumpy(), ToTensor()])
dataset_train = dataset(data_dir=os.path.join(data_dir, "train1.csv"), transform=transform_train)
loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)

## 테스트용
data = dataset_train.__getitem__(0)
img = data['img']
letter = data['letter']

##
img = img.reshape(-1, 1, 28, 28)
letter = letter.reshape(-1, 26)

##
net = Net5()
net.eval()
output = net(img, letter)