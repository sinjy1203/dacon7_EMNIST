##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

data = pd.read_csv("data/train.csv")

##
idx = np.arange(2048)
np.random.shuffle(idx)

##
num_train = 1700
data_dir = "./datasets"
train_dir = os.path.join(data_dir, "train.csv")
val_dir = os.path.join(data_dir, "val.csv")

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

##
train_idx = idx[:1700]
val_idx = idx[1700:]

train_lst = []
val_lst = []
for idx in train_idx:
    train_lst += [data[idx : idx + 1]]

train_data = pd.concat(train_lst)
train_data.to_csv(train_dir, mode='w')

##
for idx in val_idx:
    val_lst += [data[idx : idx + 1]]

val_data = pd.concat(val_lst)
val_data.to_csv(val_dir, mode='w')

##
train = pd.read_csv(train_dir)
print(train.shape)

##
val = pd.read_csv(val_dir)
print(val.iloc[:, 4:].values)