##
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import transforms

##
class dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        data = pd.read_csv(data_dir)
        self.digit = data['digit'].values
        self.img = data.iloc[:, 4:].values.reshape(-1, 28, 28, 1).astype(np.int)
        self.letter = pd.get_dummies(data.letter).values.reshape(-1, 26)

    def __len__(self):
        return len(self.digit)

    def __getitem__(self, idx):
        img = self.img[idx]
        label = self.digit[idx]
        letter = self.letter[idx]

        img = img / 255.0

        data = {"img": img, "letter": letter, "label": label}

        if self.transform:
            data = self.transform(data)

        return data


class dataset_test(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        data = pd.read_csv(data_dir)
        self.img = data.iloc[:, 2:].values.reshape(-1, 28, 28, 1).astype(np.int)
        self.letter = pd.get_dummies(data.letter).values.reshape(-1, 26)

    def __len__(self):
        return self.letter.shape[0]

    def __getitem__(self, idx):
        img = self.img[idx]
        letter = self.letter[idx]

        img = img / 255.0

        data = {"img": img, "letter": letter}

        if self.transform:
            data = self.transform(data)

        return data


class ToTensor_test(object):
    def __call__(self, data):
        img = data["img"]
        letter = data['letter']

        # img = img.reshape(-1, 28, 28, 1)

        img = img.transpose((2, 0, 1)).astype(np.float32)

        letter = letter.astype(np.float32)

        try:
            label = data['label']
            data = {'img': torch.from_numpy(img), 'letter': torch.from_numpy(letter), 'label': label}
        except:
            data = {'img': torch.from_numpy(img), 'letter': torch.from_numpy(letter)}

        return data


class ToTensor(object):
    def __call__(self, data):
        img = data["img"]
        letter = data['letter']

        img = img.reshape(28, 28, 1)

        img = img.transpose((2, 0, 1)).astype(np.float32)

        letter = letter.astype(np.float32)

        try:
            label = data['label']
            data = {'img': torch.from_numpy(img), 'letter': torch.from_numpy(letter), 'label': label}
        except:
            data = {img: torch.from_numpy(img), 'letter': torch.from_numpy(letter)}

        return data

class ToPILImage(object):
    def __init__(self):
        self.topilimage = transforms.ToPILImage()

    def __call__(self, data):
        img = data["img"].astype(np.float32)
        label = data['label']
        letter = data['letter']

        img = self.topilimage(img)
        data = {'img': img, 'letter': letter, 'label': label}

        return data

class ToNumpy(object):
    def __call__(self, data):
        img = data["img"]
        label = data['label']
        letter = data['letter']

        img = np.array(img)
        data = {'img': img, 'letter': letter, 'label': label}

        return data

class RandomAffine(object):
    def __init__(self, degrees, p):
        self.randomaffine = transforms.RandomAffine(degrees)
        self.p = p

    def __call__(self, data):
        img = data["img"]
        label = data['label']
        letter = data['letter']

        if np.random.rand() > (1 - self.p):
            img = self.randomaffine(img)

        data = {'img': img, 'letter': letter, 'label': label}

        return data

class RandomRotation(object):
    def __init__(self, degree, p):
        self.randomrotation = transforms.RandomRotation(degrees=degree, expand=False)
        self.p = p

    def __call__(self, data):
        img = data["img"]
        label = data['label']
        letter = data['letter']

        if np.random.rand() > (1-self.p):
            img = self.randomrotation(img)
        data = {'img': img, 'letter': letter, 'label': label}

        return data

class RandomCrop(object):
    def __init__(self, p):
        self.randomcrop = transforms.RandomCrop((24, 24))
        self.resize = transforms.Resize((28, 28))
        self.p = p

    def __call__(self, data):
        img = data["img"]
        label = data['label']
        letter = data['letter']

        if np.random.rand() > (1-self.p):
            img = self.randomcrop(img)
            img = self.resize(img)
        data = {'img': img, 'letter': letter, 'label': label}

        return data

class Noise(object):
    def __init__(self, std, p):
        self.std = std
        self.p = p

    def __call__(self, data):
        img = data["img"]
        label = data['label']
        letter = data['letter']

        if np.random.rand() > (1-self.p):
            noise = np.random.randn(28, 28, 1) * self.std
            img = img + noise
            img = np.clip(img, 0, 1)
        data = {'img': img, 'letter': letter, 'label': label}

        return data

class RandomPerspective(object):
    def __init__(self, p, scale):
        self.randomperspective = transforms.RandomPerspective(p=p, distortion_scale=scale)

    def __call__(self, data):
        img = data["img"]
        label = data['label']
        letter = data['letter']

        img = self.randomperspective(img)

        data = {'img': img, 'letter': letter, 'label': label}

        return data