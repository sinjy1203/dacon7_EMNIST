## 패키지
import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms

from model import *
from util import *
from dataset import *

## parameter
batch_size = 128
lr = 1e-3

data_dir = "./data/test.csv"
ckpt_dir = "./checkpoint"
submission_dir = "./data/submission.csv"
result_dir = "./result"

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
submission = pd.read_csv('./data/submission.csv')

##
transform_test = transforms.Compose([ToTensor_test()])
dataset_test = dataset_test(data_dir=data_dir, transform=transform_test)
loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)

##
fn_pred = lambda x: np.argmax(x, axis=-1)
fn_acc = lambda output, label: np.mean(1.0 * (output == label))

##
net = Net5(p=0).to(device)
optim = torch.optim.Adam(params=net.parameters(), lr=lr)
fn_loss = nn.CrossEntropyLoss().to(device)

batch_test = np.ceil(len(dataset_test) / batch_size).astype(np.int)

##
net, optim = load(ckpt_dir, net, optim)
y_pred = []

with torch.no_grad():
    net.eval()

    for batch, data in enumerate(loader_test, 1):
        img, letter = data['img'].to(device), data['letter'].to(device)

        output = net(img, letter.float())
        output = fn_pred(output)
        y_pred += [output]

        print("TEST : %03d / %03d 진행중.." % (batch, batch_test))

submission.digit = torch.cat(y_pred).detach().cpu().numpy()
submission.to_csv(os.path.join(result_dir, "my_submission.csv"), index=False)