'''Load image/class/box from a annotation file.

The annotation file is organized as:
image_name #obj x y w h class_index ...

e.g.
1.jpg 1 312 143 193 178 0
2.jpg 2 388 119 193 162 0 656 415 243 183 0
...

'''

from __future__ import print_function

import os
import sys
import os.path

import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image


class ListDataset:
    num_samples = 4
    image_size = 300

    def __init__(self, root, list_file):
        self.root = root

        self.fnames = []
        self.boxes = []
        self.labels = []

        f = open(list_file)
        for i in range(self.num_samples):
            line = f.readline()
            splited = line.strip().split()
            self.fnames.append(splited[0])

            num_objs = int(splited[1])
            box = []
            label = []
            for j in range(num_objs):
                x = splited[2+5*j]
                y = splited[3+5*j]
                w = splited[4+5*j]
                h = splited[5+5*j]
                c = splited[6+5*j]
                box.append([int(x),int(y),int(w),int(h)])
                label.append(int(c))
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))

    def load(self, batch_size):
        assert batch_size <= self.num_samples, 'load batch_size is too large'
        images = []
        boxes = []
        labels = []
        indices = torch.randperm(self.num_samples)[:batch_size]
        for index in indices:
            fname = self.fnames[index]
            im = Image.open(os.path.join(self.root, fname))

            # transform box (with original image sizes)
            box = self.as_ssd_box(im, self.boxes[index])
            boxes.append(box)

            # resize image
            im = im.resize((self.image_size,self.image_size))
            im = transforms.ToTensor()(im)  # PIL image -> tensor
            images.append(im.unsqueeze(0))  # [C,H,W] -> [1,C,H,W]

            # add labels
            labels.append(self.labels[index])
        return torch.cat(images,0), boxes, labels

    def as_ssd_box(self, im, box):
        '''Transform absolute box (x,y,w,h) to SSD relative box (cx,cy,w,h).

        Args:
          im: (PIL image) image sample.
          box: (tensor) image object bounding boxes, sized [N,4].

        Return:
          (tensor) box after transform, sized [N,4].
        '''
        imh, imw = im.size
        cx = (box[:,0] + box[:,2] / 2.0) / imw
        cy = (box[:,1] + box[:,3] / 2.0) / imh
        w = box[:,2] / imw
        h = box[:,3] / imh
        return torch.cat([cx,cy,w,h], 1)

    def __len__(self):
        return self.num_samples



# dataset = ListDataset(root='./fake_data/', list_file='./fake_data/index.txt')
# images, boxes, labels = dataset.load(4)
# print(images.size())
# print(boxes)
# print(labels)
# print(len(dataset))
