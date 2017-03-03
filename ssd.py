import math
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from multibox import MultiBox
from encoder import DataEncoder


class L2Norm2d(nn.Module):
    '''L2Norm layer across all channels.'''
    def __init__(self, scale):
        super(L2Norm2d, self).__init__()
        self.scale = scale

    def forward(self, x):
        '''out = scale * x / sqrt(\sum x_i^2)'''
        norm = x.pow(2).sum(1).sqrt()  # [N,1,H,W]
        out = self.scale * x / (norm.expand_as(x) + 1e-5)  # [N,C,H,W]
        return out


class SSD300(nn.Module):
    input_size = 300
    feature_map_sizes = [38, 19, 10, 5, 3, 1]
    aspect_ratios = [(2,),(2,3),(2,3),(2,3),(2,),(2,)]

    def __init__(self):
        super(SSD300, self).__init__()
        # Data encoder.
        self.data_encoder = DataEncoder(self.input_size,　self.feature_map_sizes,　self.aspect_ratios)

        # Model.
        self.base = self.VGG16()
        self.norm4 = L2Norm2d(20)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5_3 = nn.BatchNorm2d(512)

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(1024)

        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.bn7 = nn.BatchNorm2d(1024)

        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1)
        self.bn8_1 = nn.BatchNorm2d(256)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2)
        self.bn8_2 = nn.BatchNorm2d(512)

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1)
        self.bn9_1 = nn.BatchNorm2d(128)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)
        self.bn9_2 = nn.BatchNorm2d(256)

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.bn10_1 = nn.BatchNorm2d(128)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3)
        self.bn10_2 = nn.BatchNorm2d(256)

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.bn11_1 = nn.BatchNorm2d(128)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3)
        self.bn11_2 = nn.BatchNorm2d(256)

        # multibox
        self.multibox = MultiBox()

    def forward(self, x):
        hs = []  # cache intermediate layers outputs

        h = self.base(x)
        h = self.norm4(h)
        hs.append(h)  # conv4_3

        h = F.max_pool2d(h, kernel_size=2, stride=2, ceil_mode=True)

        h = self.conv5_1(h)
        h = F.relu(self.bn5_1(h))
        h = self.conv5_2(h)
        h = F.relu(self.bn5_2(h))
        h = self.conv5_3(h)
        h = F.relu(self.bn5_3(h))
        h = F.max_pool2d(h, kernel_size=3, padding=1, stride=1, ceil_mode=True)

        h = self.conv6(h)
        h = F.relu(self.bn6(h))

        h = self.conv7(h)
        h = F.relu(self.bn7(h))
        hs.append(h)  # conv7

        h = self.conv8_1(h)
        h = F.relu(self.bn8_1(h))
        h = self.conv8_2(h)
        h = F.relu(self.bn8_2(h))
        hs.append(h)  # conv8_2

        h = self.conv9_1(h)
        h = F.relu(self.bn9_1(h))
        h = self.conv9_2(h)
        h = F.relu(self.bn9_2(h))
        hs.append(h)  # conv9_2

        h = self.conv10_1(h)
        h = F.relu(self.bn10_1(h))
        h = self.conv10_2(h)
        h = F.relu(self.bn10_2(h))
        hs.append(h)  # conv10_2

        h = self.conv11_1(h)
        h = F.relu(self.bn11_1(h))
        h = self.conv11_2(h)
        h = F.relu(self.bn11_2(h))
        hs.append(h)  # conv11_2

        # print('\nintermediate feature map sizes:')
        # for x in hs:
        #     print(x.size())

        loc_preds, conf_preds = self.multibox(hs)
        return loc_preds, conf_preds

    def VGG16(self):
        '''VGG16 layers.'''
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(True)]
                in_channels = x
        return nn.Sequential(*layers)
