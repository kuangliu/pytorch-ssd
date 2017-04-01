'''Load image/class/box from a annotation file.

The annotation file is organized as:
    image_name #obj xmin ymin xmax ymax class_index ..
'''
from __future__ import print_function

import os
import sys
import os.path

import random
import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from encoder import DataEncoder


class ListDataset(data.Dataset):
    image_size = 300

    def __init__(self, root, list_file, transform):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str) path to index file.
          transform: ([transforms]) image transforms.
        '''
        self.root = root
        self.transform = transform

        self.fnames = []
        self.boxes = []
        self.labels = []

        self.data_encoder = DataEncoder()

        with open(list_file) as f:
            lines = f.readlines()
            self.num_samples = len(lines)

        for line in lines:
            splited = line.strip().split()
            self.fnames.append(splited[0])

            num_objs = int(splited[1])
            box = []
            label = []
            for j in range(num_objs):
                xmin = splited[2+5*j]
                ymin = splited[3+5*j]
                xmax = splited[4+5*j]
                ymax = splited[5+5*j]
                c = splited[6+5*j]
                box.append([float(xmin),float(ymin),float(xmax),float(ymax)])
                label.append(int(c))
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        im = Image.open(os.path.join(self.root, fname))
        # RGB to BGR for pretrained VGG16 model.
        im = np.array(im)
        im = im[:,:,::-1]
        im = Image.fromarray(im)
        im = im.resize((self.image_size,self.image_size))
        im = self.transform(im)
        # Encode loc & conf targets.
        loc_target, conf_target = self.data_encoder.encode(self.boxes[idx], self.labels[idx])
        return im, loc_target, conf_target

    def __len__(self):
        return self.num_samples
