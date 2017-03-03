from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import itertools

from ssd import *
from datagen import *
from utils import progress_bar, get_mean_and_std
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch SSD Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()

# Data
print('==> Preparing data..')
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.mul(255)), transforms.Normalize((104, 117, 123), (1,1,1))])
#transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4494,0.4236,0.3882), (0.2408,0.2342,0.2365))])
dataset = ListDataset(root='/search/liukuang/data/VOC2007/JPEGImages', list_file='./voc_data/index.txt', transform=transform)

# Model
net = SSD300()
if use_cuda:
    #net = torch.nn.DataParallel(net, device_ids=[0])
    net.cuda()
    #cudnn.benchmark = True

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
#optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0

    # load a batch
    batch_size = 16
    for batch_idx in range(len(dataset) / batch_size):
        images, boxes, labels = dataset.load(batch_size)
        images = images.cuda()
        optimizer.zero_grad()

        images = Variable(images)
        loc_preds, conf_preds = net(images)

        loc_targets = torch.Tensor(batch_size, 8732, 4)
        conf_targets = torch.LongTensor(batch_size, 8732)
        for i in range(batch_size):
            loc_target, conf_target = net.data_encoder.encode(boxes[i], labels[i])
            loc_targets[i] = loc_target
            conf_targets[i] = conf_target

        loc_targets = Variable(loc_targets.cuda())
        conf_targets = Variable(conf_targets.cuda())
        loss = net.multibox.loss(loc_preds, loc_targets, conf_preds, conf_targets)
        loss.backward()
        optimizer.step()

        print(loss.data[0])
        #train_loss += loss.data[0]
        #print(train_loss/(batch_idx+1))

for epoch in range(200):
    train(epoch)
