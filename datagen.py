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


class ListDataset(data.Dataset):
    image_size = 300
    max_obj_per_image = 10

    def __init__(self, root, list_file):
        self.root = root

        self.fnames = []
        self.boxes = []
        self.labels = []

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
                box.append([int(xmin),int(ymin),int(xmax),int(ymax)])
                label.append(int(c) + 1)  # background label is 0
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))

    def __getitem__(self, index):
        '''To take advantage of the PyTorch multi-thread dataloader, we use a
        little trick. We concat multiple object bboxes and class labels together
        to create a fix sized long vector, with some -1s at the end.

        e.g.
        image1: [box1,box2,box3...,c1,c2,c3...,-1,-1,-1..]
        '''
        fname = self.fnames[index]
        im = Image.open(os.path.join(self.root, fname))
        im = im.resize((self.image_size,self.image_size))
        im = transforms.ToTensor()(im)

        box = self.as_ssd_box(im, self.boxes[index], 'XYXY')  # [#obj,4]
        target = torch.Tensor()
        pass
        return im

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
            box = self.as_ssd_box(im, self.boxes[index], 'XYXY')
            boxes.append(box)

            # resize image
            im = im.resize((self.image_size,self.image_size))
            im = transforms.ToTensor()(im)  # PIL image -> tensor
            images.append(im.unsqueeze(0))  # [C,H,W] -> [1,C,H,W]

            # add labels
            labels.append(self.labels[index])
        return torch.cat(images,0), boxes, labels

    def as_ssd_box(self, im, box, arrange):
        '''Transform absolute box (x,y,w,h) or (x,y,x,y) to SSD relative box (cx,cy,w,h).

        Args:
          im: (PIL image) image sample.
          box: (tensor) image object bounding boxes, sized [N,4].
          arrange: (str) 'XYWH' or 'XYXY' .

        Return:
          (tensor) box after transform, sized [N,4].
        '''
        imh, imw = im.size
        if arrange == 'XYWH':
            cx = (box[:,0] + box[:,2] / 2.0) / imw
            cy = (box[:,1] + box[:,3] / 2.0) / imh
            w = box[:,2] / imw
            h = box[:,3] / imh
        elif arrange == 'XYXY':
            cx = (box[:,0] + box[:,2]) / 2.0 / imw
            cy = (box[:,1] + box[:,3]) / 2.0 / imh
            w = (box[:,2] - box[:,0]) / imw
            h = (box[:,3] - box[:,1]) / imh
        return torch.cat([cx,cy,w,h], 1)

    def __len__(self):
        return self.num_samples


# dataset = ListDataset(root='./fake_data/', list_file='./fake_data/index.txt')
# loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)


# dataset = ListDataset(root='/mnt/hgfs/D/download/PASCAL VOC/VOC2007/JPEGImages', list_file='./voc_data/index.txt')
# images, boxes, labels = dataset.load(10)
# print(images.size())
# print(boxes)
# print(labels)
# print(len(dataset))
