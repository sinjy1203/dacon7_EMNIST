##
import os
import numpy as np

import torch
import torch.nn as nn

##
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1, bias=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU()

        self.fc1 = nn.Linear(in_features=980, out_features=100, bias=True)
        self.relu1_fc1 = nn.ReLU()
        self.drop1_fc1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(in_features=100, out_features=10, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu2(x)

        x = x.view(-1, 980)

        x = self.fc1(x)
        x = self.relu1_fc1(x)
        x = self.drop1_fc1(x)

        x = self.fc2(x)

        return x


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            cbr_lst = []
            cbr_lst += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=stride, padding=padding, bias=bias)]
            cbr_lst += [nn.BatchNorm2d(num_features=out_channels)]
            cbr_lst += [nn.ReLU()]

            cbr = nn.Sequential(*cbr_lst)
            return cbr

        def LBR(in_features, out_features, bias=True):
            lbr_lst = []
            lbr_lst += [nn.Linear(in_features=in_features, out_features=out_features, bias=bias)]
            lbr_lst += [nn.BatchNorm1d(num_features=out_features)]
            lbr_lst += [nn.ReLU()]

            lbr = nn.Sequential(*lbr_lst)
            return lbr

        self.conv1_1 = CBR2d(in_channels=1, out_channels=32)
        self.conv1_2 = CBR2d(in_channels=32, out_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = CBR2d(in_channels=64, out_channels=128)
        self.conv2_2 = CBR2d(in_channels=128, out_channels=256)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.lbr1 = LBR(in_features=12544, out_features=1000)

        self.lbr2 = LBR(in_features=1000, out_features=200)

        self.lbr2_letter = LBR(in_features=26, out_features=100)

        self.fc3 = nn.Linear(in_features=300, out_features=10, bias=True)


    def forward(self, x, letter):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)

        x = x.view(-1, 12544)

        x = self.lbr1(x)

        x = self.lbr2(x)

        letter = self.lbr2_letter(letter)

        y = torch.cat((x, letter), dim=1)
        y = self.fc3(y)

        return y


class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1_1 = nn.BatchNorm2d(num_features=32)
        self.relu1_1 = nn.ReLU()

        self.conv1_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1_2 = nn.BatchNorm2d(num_features=64)
        self.relu1_2 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2_1 = nn.BatchNorm2d(num_features=128)
        self.relu2_1 = nn.ReLU()

        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2_2 = nn.BatchNorm2d(num_features=256)
        self.relu2_2 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.dropout1 = nn.Dropout2d(p=0.5)

        self.fc1 = nn.Linear(in_features=12544, out_features=1000, bias=True)
        self.fc1_bn = nn.BatchNorm1d(num_features=1000)
        self.fc1_relu = nn.ReLU()

        self.dropout2 = nn.Dropout2d(p=0.5)

        self.fc2 = nn.Linear(in_features=1000, out_features=200, bias=True)
        self.fc2_bn = nn.BatchNorm1d(num_features=200)
        self.fc2_relu = nn.ReLU()

        self.fc2_letter = nn.Linear(in_features=26, out_features=100, bias=True)
        self.fc2_letter_bn = nn.BatchNorm1d(num_features=100)
        self.fc2_letter_relu = nn.ReLU()

        self.dropout3 = nn.Dropout2d(p=0.5)

        self.fc3 = nn.Linear(in_features=300, out_features=10, bias=True)

        nn.init.kaiming_normal_(self.conv1_1.weight)
        nn.init.kaiming_normal_(self.conv1_2.weight)
        nn.init.kaiming_normal_(self.conv2_1.weight)
        nn.init.kaiming_normal_(self.conv2_2.weight)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc2_letter.weight)
        nn.init.kaiming_normal_(self.fc3.weight)



    def forward(self, x, letter):
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu1_1(x)

        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)

        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.relu2_1(x)

        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu2_2(x)

        x = self.pool2(x)

        x = x.view(-1, 12544)

        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = self.fc1_relu(x)

        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = self.fc2_relu(x)

        letter = self.fc2_letter(letter)
        letter = self.fc2_letter_bn(letter)
        letter = self.fc2_letter_relu(letter)

        y = torch.cat((x, letter), dim=1)
        y = self.dropout3(y)
        y = self.fc3(y)

        return y

class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1_1 = nn.BatchNorm2d(num_features=32)
        self.relu1_1 = nn.ReLU()

        self.conv1_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1_2 = nn.BatchNorm2d(num_features=64)
        self.relu1_2 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2_1 = nn.BatchNorm2d(num_features=128)
        self.relu2_1 = nn.ReLU()

        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2_2 = nn.BatchNorm2d(num_features=256)
        self.relu2_2 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.dropout1 = nn.Dropout2d(p=0.5)

        self.fc1 = nn.Linear(in_features=12544, out_features=1000, bias=True)
        self.fc1_bn = nn.BatchNorm1d(num_features=1000)
        self.fc1_relu = nn.ReLU()

        self.dropout2 = nn.Dropout2d(p=0.5)

        self.fc2 = nn.Linear(in_features=1000, out_features=100, bias=True)
        self.fc2_bn = nn.BatchNorm1d(num_features=100)
        self.fc2_relu = nn.ReLU()

        self.dropout3 = nn.Dropout2d(p=0.5)

        self.fc3 = nn.Linear(in_features=100, out_features=10, bias=True)

        nn.init.kaiming_normal_(self.conv1_1.weight)
        nn.init.kaiming_normal_(self.conv1_2.weight)
        nn.init.kaiming_normal_(self.conv2_1.weight)
        nn.init.kaiming_normal_(self.conv2_2.weight)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)



    def forward(self, x, letter):
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu1_1(x)

        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)

        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.relu2_1(x)

        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu2_2(x)

        x = self.pool2(x)

        x = x.view(-1, 12544)

        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = self.fc1_relu(x)

        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = self.fc2_relu(x)

        x = self.dropout3(x)
        x = self.fc3(x)

        return x


class Net5(nn.Module):
    def __init__(self, p):
        super(Net5, self).__init__()
        self.p = p

        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1_1 = nn.BatchNorm2d(num_features=32)
        self.relu1_1 = nn.ReLU()

        self.conv1_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1_2 = nn.BatchNorm2d(num_features=64)
        self.relu1_2 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # output.shape = 64 14 14

        #

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn2_1 = nn.BatchNorm2d(num_features=32)
        self.relu2_1 = nn.ReLU()

        self.conv2_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2_2 = nn.BatchNorm2d(num_features=32)
        self.relu2_2 = nn.ReLU()

        self.conv2_3 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2_3 = nn.BatchNorm2d(num_features=128)

        self.conv2_skip = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn2_skip = nn.BatchNorm2d(num_features=128)

        # add(self.bn2_2, self.bn2_skip)
        self.relu2_3 = nn.ReLU()
        # output.shape = 128 7 7

        #

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3_1 = nn.BatchNorm2d(num_features=64)
        self.relu3_1 = nn.ReLU()

        self.conv3_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3_2 = nn.BatchNorm2d(num_features=64)
        self.relu3_2 = nn.ReLU()

        self.conv3_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3_3 = nn.BatchNorm2d(num_features=128)

        # add(self.bn3_2, self.bn3_skip)
        self.relu3_3 = nn.ReLU()
        # output.shape = 128 7 7

        #

        self.conv4_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn4_1 = nn.BatchNorm2d(num_features=64)
        self.relu4_1 = nn.ReLU()

        self.conv4_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4_2 = nn.BatchNorm2d(num_features=64)
        self.relu4_2 = nn.ReLU()

        self.conv4_3 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4_3 = nn.BatchNorm2d(num_features=256)

        self.conv4_skip = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn4_skip = nn.BatchNorm2d(num_features=256)

        # add(self.bn3_2, self.bn3_skip)
        self.relu4_3 = nn.ReLU()
        # output.shape = 256 4 4

        #

        self.conv5_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn5_1 = nn.BatchNorm2d(num_features=128)
        self.relu5_1 = nn.ReLU()

        self.conv5_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn5_2 = nn.BatchNorm2d(num_features=128)
        self.relu5_2 = nn.ReLU()

        self.conv5_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn5_3 = nn.BatchNorm2d(num_features=256)

        # add(self.bn3_2, self.bn3_skip)
        self.relu5_3 = nn.ReLU()
        # output.shape = 256 4 4

        #

        self.conv6_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn6_1 = nn.BatchNorm2d(num_features=128)
        self.relu6_1 = nn.ReLU()

        self.conv6_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn6_2 = nn.BatchNorm2d(num_features=128)
        self.relu6_2 = nn.ReLU()

        self.conv6_3 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn6_3 = nn.BatchNorm2d(num_features=512)

        self.conv6_skip = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn6_skip = nn.BatchNorm2d(num_features=512)

        # add(self.bn3_2, self.bn3_skip)
        self.relu6_3 = nn.ReLU()
        # output.shape = 512 2 2

        #

        self.conv7_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn7_1 = nn.BatchNorm2d(num_features=256)
        self.relu7_1 = nn.ReLU()

        self.conv7_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn7_2 = nn.BatchNorm2d(num_features=256)
        self.relu7_2 = nn.ReLU()

        self.conv7_3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn7_3 = nn.BatchNorm2d(num_features=512)

        # add(self.bn3_2, self.bn3_skip)
        self.relu7_3 = nn.ReLU()
        # output.shape = 512 2 2

        #

        self.avgpool = nn.AvgPool2d(kernel_size=2)
        # output.shape = 512

        self.fc1_letter = nn.Linear(in_features=26, out_features=64, bias=True)
        self.fc1_letter_bn = nn.BatchNorm1d(num_features=64)
        self.fc1_letter_relu = nn.ReLU()

        self.dropout1 = nn.Dropout2d(p=self.p)

        self.fc2 = nn.Linear(in_features=576, out_features=10, bias=True)

        nn.init.kaiming_normal_(self.conv1_1.weight)
        nn.init.kaiming_normal_(self.conv1_2.weight)
        nn.init.kaiming_normal_(self.conv2_1.weight)
        nn.init.kaiming_normal_(self.conv2_2.weight)
        nn.init.kaiming_normal_(self.conv2_3.weight)
        nn.init.kaiming_normal_(self.conv2_skip.weight)
        nn.init.kaiming_normal_(self.conv3_1.weight)
        nn.init.kaiming_normal_(self.conv3_2.weight)
        nn.init.kaiming_normal_(self.conv3_3.weight)
        nn.init.kaiming_normal_(self.conv4_1.weight)
        nn.init.kaiming_normal_(self.conv4_2.weight)
        nn.init.kaiming_normal_(self.conv4_3.weight)
        nn.init.kaiming_normal_(self.conv4_skip.weight)
        nn.init.kaiming_normal_(self.conv5_1.weight)
        nn.init.kaiming_normal_(self.conv5_2.weight)
        nn.init.kaiming_normal_(self.conv5_3.weight)
        nn.init.kaiming_normal_(self.conv6_1.weight)
        nn.init.kaiming_normal_(self.conv6_2.weight)
        nn.init.kaiming_normal_(self.conv6_3.weight)
        nn.init.kaiming_normal_(self.conv6_skip.weight)
        nn.init.kaiming_normal_(self.conv7_1.weight)
        nn.init.kaiming_normal_(self.conv7_2.weight)
        nn.init.kaiming_normal_(self.conv7_3.weight)
        nn.init.kaiming_normal_(self.fc1_letter.weight)
        nn.init.kaiming_normal_(self.fc2.weight)



    def forward(self, x, letter):
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu1_1(x)

        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)

        x = self.pool1(x)

        #

        x1 = self.conv2_1(x)
        x1 = self.bn2_1(x1)
        x1 = self.relu2_1(x1)

        x1 = self.conv2_2(x1)
        x1 = self.bn2_2(x1)
        x1 = self.relu2_2(x1)

        x1 = self.conv2_3(x1)
        x1 = self.bn2_3(x1)

        x2 = self.conv2_skip(x)
        x2 = self.bn2_skip(x2)

        x = x1 + x2
        x = self.relu2_2(x)

        #

        x1 = self.conv3_1(x)
        x1 = self.bn3_1(x1)
        x1 = self.relu3_1(x1)

        x1 = self.conv3_2(x1)
        x1 = self.bn3_2(x1)
        x1 = self.relu3_2(x1)

        x1 = self.conv3_3(x1)
        x1 = self.bn3_3(x1)

        x2 = x

        x = x1 + x2
        x = self.relu3_3(x)

        #

        x1 = self.conv4_1(x)
        x1 = self.bn4_1(x1)
        x1 = self.relu4_1(x1)

        x1 = self.conv4_2(x1)
        x1 = self.bn4_2(x1)
        x1 = self.relu4_2(x1)

        x1 = self.conv4_3(x1)
        x1 = self.bn4_3(x1)

        x2 = self.conv4_skip(x)
        x2 = self.bn4_skip(x2)

        x = x1 + x2
        x = self.relu4_3(x)

        #

        x1 = self.conv5_1(x)
        x1 = self.bn5_1(x1)
        x1 = self.relu5_1(x1)

        x1 = self.conv5_2(x1)
        x1 = self.bn5_2(x1)
        x1 = self.relu5_2(x1)

        x1 = self.conv5_3(x1)
        x1 = self.bn5_3(x1)

        x2 = x

        x = x1 + x2
        x = self.relu5_3(x)

        #

        x1 = self.conv6_1(x)
        x1 = self.bn6_1(x1)
        x1 = self.relu6_1(x1)

        x1 = self.conv6_2(x1)
        x1 = self.bn6_2(x1)
        x1 = self.relu6_2(x1)

        x1 = self.conv6_3(x1)
        x1 = self.bn6_3(x1)

        x2 = self.conv6_skip(x)
        x2 = self.bn6_skip(x2)

        x = x1 + x2
        x = self.relu6_3(x)

        #

        x1 = self.conv7_1(x)
        x1 = self.bn7_1(x1)
        x1 = self.relu7_1(x1)

        x1 = self.conv7_2(x1)
        x1 = self.bn7_2(x1)
        x1 = self.relu7_2(x1)

        x1 = self.conv7_3(x1)
        x1 = self.bn7_3(x1)

        x2 = x

        x = x1 + x2
        x = self.relu5_3(x)

        #

        x = self.avgpool(x)
        x = x.view(-1, 512)

        letter = self.fc1_letter(letter)
        letter = self.fc1_letter_bn(letter)
        letter = self.fc1_letter_relu(letter)

        y = torch.cat((x, letter), dim=1)
        y = self.dropout1(y)
        y = self.fc2(y)

        return y