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
net1 = Net3().to(device)
optim = torch.optim.Adam(params=net1.parameters(), lr=lr)
net2 = Net3().to(device)
optim = torch.optim.Adam(params=net2.parameters(), lr=lr)
net3 = Net3().to(device)
optim = torch.optim.Adam(params=net3.parameters(), lr=lr)
net4 = Net3().to(device)
optim = torch.optim.Adam(params=net4.parameters(), lr=lr)
net5 = Net3().to(device)
optim = torch.optim.Adam(params=net5.parameters(), lr=lr)

fn_loss = nn.CrossEntropyLoss().to(device)

batch_test = np.ceil(len(dataset_test) / batch_size).astype(np.int)

##
ckpt_n = os.listdir(ckpt_dir)
ckpt_dir_lst = [os.path.join(ckpt_dir, name) for name in ckpt_n]
net1, optim1 = load_ensemble(ckpt_dir_lst[0], net1, optim)
net2, optim2 = load_ensemble(ckpt_dir_lst[1], net2, optim)
net3, optim3 = load_ensemble(ckpt_dir_lst[2], net3, optim)
net4, optim4 = load_ensemble(ckpt_dir_lst[3], net4, optim)
net5, optim5 = load_ensemble(ckpt_dir_lst[4], net5, optim)
y_pred = []

##

with torch.no_grad():
    net1.eval()
    net2.eval()
    net3.eval()
    net4.eval()
    net5.eval()

    for batch, data in enumerate(loader_test, 1):
        img, letter = data['img'].to(device), data['letter'].to(device)

        output1 = net1(img, letter.float())
        output2 = net2(img, letter.float())
        output3 = net3(img, letter.float())
        output4 = net4(img, letter.float())
        output5 = net5(img, letter.float())
        output = output1 + output2 + output3 + output4 + output5
        output = fn_pred(output)
        y_pred += [output]

        print("TEST : %03d / %03d 진행중.." % (batch, batch_test))

submission.digit = torch.cat(y_pred).detach().cpu().numpy()
submission.to_csv(os.path.join(result_dir, "my_submission.csv"), index=False)