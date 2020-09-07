import os

import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save(ckpt_dir, net, optim, val_acc):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()}, os.path.join(ckpt_dir, "dacon7_%.3f.pth" % val_acc))

def save2(ckpt_dir, net, optim, val_acc, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()}, os.path.join(ckpt_dir, "dacon7_%d_%.3f.pth" % (epoch, val_acc)))

def load(ckpt_dir, net, optim):
    ckpt_lst = os.listdir(ckpt_dir)
    model_n = ckpt_lst[-1]

    model = torch.load(os.path.join(ckpt_dir, model_n), map_location=device)

    net.load_state_dict(model['net'])
    optim.load_state_dict(model['optim'])

    return net, optim

def load_ensemble(ckpt_dir, net, optim):
    model = torch.load(os.path.join(ckpt_dir), map_location=device)

    net.load_state_dict(model['net'])
    optim.load_state_dict(model['optim'])

    return net, optim