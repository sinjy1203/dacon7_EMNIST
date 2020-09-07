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

## parser 생성하기
parser = argparse.ArgumentParser(description="train the dacon7", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--save_epoch_st", default=100, type=int, dest="save_epoch_st")
parser.add_argument("--save_epoch_la", default=200, type=int, dest="save_epoch_la")
parser.add_argument("--save_epoch_offset", default=10, type=int, dest="save_epoch_offset")

parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
parser.add_argument("--batch_size", default=64, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=200, type=int, dest="num_epoch")
parser.add_argument("--model", default="Net3", type= str, dest="model")
parser.add_argument("--dropout", default=0.5, type=float, dest="dropout")

parser.add_argument("--noise_std", default=0.07, type=float, dest="noise_std")
parser.add_argument("--affine_degree", default=30, type=float, dest="affine_degree")
parser.add_argument("--rotation_degree", default=30, type=float, dest="rotation_degree")
parser.add_argument("--perspective_scale", default=0.5, type=float, dest="perspective_scale")

parser.add_argument("--aug_p", default=0.3, type=float, dest="aug_p")

args = parser.parse_args()

## parameter
save_epoch_st = args.save_epoch_st
save_epoch_la = args.save_epoch_la
save_epoch_offset = args.save_epoch_offset

num_epoch = args.num_epoch
batch_size = args.batch_size
lr = args.lr
dropout = args.dropout

noise_std = args.noise_std
affine_degree = args.affine_degree
rotation_degree = args.rotation_degree
perspective_scale = args.perspective_scale

aug_p = args.aug_p

model = args.model

data_dir = "./drive/My Drive/Colab Notebooks/training_dacon7/datasets"
ckpt_dir = "./drive/My Drive/Colab Notebooks/training_dacon7/checkpoint"
log_dir = "./drive/My Drive/Colab Notebooks/training_dacon7/log"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

## dataloader
transform_train = transforms.Compose([Noise(noise_std, p=aug_p), ToPILImage(), RandomPerspective(p=aug_p, scale=perspective_scale),
                                      RandomRotation(rotation_degree, p=aug_p), RandomAffine(affine_degree, p=aug_p),
                                      RandomCrop(p=aug_p), ToNumpy(), ToTensor()])
dataset_train = dataset(data_dir=os.path.join(data_dir, "train.csv"), transform=transform_train)
loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)

transform_val = transforms.Compose([ToTensor()])
dataset_val = dataset(data_dir=os.path.join(data_dir, "val.csv"), transform=transform_val)
loader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False)

##
batch_train = np.ceil(len(dataset_train) / batch_size).astype(np.int)
batch_val = np.ceil(len(dataset_val) / batch_size).astype(np.int)

##
fn_pred = lambda x: np.argmax(x, axis=-1)
fn_acc = lambda output, label: np.mean(1.0 * (output == label))

##
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, "train"))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, "val"))

##
if model == "Net3": net = Net3().to(device)
if model == "Net4": net = Net4().to(device)
if model == "Net5": net = Net5(p=dropout).to(device)
optim = torch.optim.Adam(params=net.parameters(), lr=lr)
fn_loss = nn.CrossEntropyLoss().to(device)

##
for epoch in range(1, num_epoch+1):
    net.train()
    train_loss_arr = []
    train_pred_arr = []
    train_label_arr = []

    val_loss_arr = []
    val_pred_arr = []
    val_label_arr = []

    for batch, data in enumerate(loader_train, 0):
        img, letter, label = data['img'].to(device), data['letter'].to(device), data['label'].to(device)

        output = net(img, letter.float())

        loss = fn_loss(output, label)

        optim.zero_grad()
        loss.backward()
        optim.step()

        train_loss_arr += [loss.item()]
        train_pred_arr += [x for x in fn_pred(output.detach().to("cpu").numpy())]
        train_label_arr += [y for y in label.detach().to("cpu").numpy()]

    with torch.no_grad():
        net.eval()

        for batch, data in enumerate(loader_val, 0):
            img, letter, label = data['img'].to(device), data['letter'].to(device), data['label'].to(device)

            output = net(img, letter.float())
            loss = fn_loss(output, label)

            val_loss_arr += [loss.item()]
            val_pred_arr += [x for x in fn_pred(output.detach().to("cpu").numpy())]
            val_label_arr += [y for y in label.detach().to("cpu").numpy()]


    train_pred_arr, train_label_arr = np.array(train_pred_arr), np.array(train_label_arr)
    train_acc = fn_acc(train_pred_arr, train_label_arr)

    val_pred_arr, val_label_arr = np.array(val_pred_arr), np.array(val_label_arr)
    val_acc = fn_acc(val_pred_arr, val_label_arr)

    writer_train.add_scalar("loss", np.mean(train_loss_arr), epoch)
    writer_val.add_scalar("loss", np.mean(val_loss_arr), epoch)

    writer_train.add_scalar("acc", np.mean(train_acc), epoch)
    writer_val.add_scalar("acc", np.mean(val_acc), epoch)

    print("TRAIN: epoch %04d / %04d | loss %.4f | acc %.4f  || VAL: loss %.4f | acc %.4f" %
          (epoch, num_epoch, np.mean(train_loss_arr), train_acc, np.mean(val_loss_arr), val_acc))

    if epoch >= save_epoch_st and epoch <= save_epoch_la and epoch % save_epoch_offset == 0:
        save2(ckpt_dir, net, optim, val_acc, epoch)

writer_train.close()
writer_val.close()