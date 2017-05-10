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

from encoder import DataEncoder
from PIL import Image, ImageOps


class ListDataset(data.Dataset):
    img_size = 300

    def __init__(self, root, list_file, train, transform):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str) path to index file.
          train: (boolean) train or test.
          transform: ([transforms]) image transforms.
        '''
        self.root = root
        self.train = train
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
            for i in range(num_objs):
                xmin = splited[2+5*i]
                ymin = splited[3+5*i]
                xmax = splited[4+5*i]
                ymax = splited[5+5*i]
                c = splited[6+5*i]
                box.append([float(xmin),float(ymin),float(xmax),float(ymax)])
                label.append(int(c))
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))

    def __getitem__(self, idx):
        '''Load a image, and encode its bbox locations and class labels.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          loc_target: (tensor) location targets, sized [8732,4].
          conf_target: (tensor) label targets, sized [8732,].
        '''
        # Load image and bbox locations.
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.root, fname))
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx]

        # Data augmentation while training.
        if self.train:
            img, boxes = self.random_flip(img, boxes)
            img, boxes, labels = self.random_crop(img, boxes, labels)

        # Scale bbox locaitons to [0,1].
        w,h = img.size
        boxes /= torch.Tensor([w,h,w,h]).expand_as(boxes)

        img = img.resize((self.img_size,self.img_size))
        img = self.transform(img)

        # Encode loc & conf targets.
        loc_target, conf_target = self.data_encoder.encode(boxes, labels)
        return img, loc_target, conf_target

    def random_flip(self, img, boxes):
        '''Randomly flip the image and adjust the bbox locations.

        For bbox (xmin, ymin, xmax, ymax), the flipped bbox is:
        (w-xmax, ymin, w-xmin, ymax).

        Args:
          img: (PIL.Image) image.
          boxes: (tensor) bbox locations, sized [#obj, 4].

        Returns:
          img: (PIL.Image) randomly flipped image.
          boxes: (tensor) randomly flipped bbox locations, sized [#obj, 4].
        '''
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            w = img.width
            xmin = w - boxes[:,2]
            xmax = w - boxes[:,0]
            boxes[:,0] = xmin
            boxes[:,2] = xmax
        return img, boxes

    def random_crop(self, img, boxes, labels):
        '''Randomly crop the image and adjust the bbox locations.

        For more details, see 'Chapter2.2: Data augmentation' of the paper.

        Args:
          img: (PIL.Image) image.
          boxes: (tensor) bbox locations, sized [#obj, 4].
          labels: (tensor) bbox labels, sized [#obj,].

        Returns:
          img: (PIL.Image) cropped image.
          selected_boxes: (tensor) selected bbox locations.
          labels: (tensor) selected bbox labels.
        '''
        imw, imh = img.size
        while True:
            min_iou = random.choice([None, 0.1, 0.3, 0.5, 0.7, 0.9])
            if min_iou is None:
                return img, boxes, labels

            for _ in range(100):
                w = random.randrange(int(0.1*imw), imw)
                h = random.randrange(int(0.1*imh), imh)

                if h > 2*w or w > 2*h:
                    continue

                x = random.randrange(imw - w)
                y = random.randrange(imh - h)
                roi = torch.Tensor([[x, y, x+w, y+h]])

                center = (boxes[:,:2] + boxes[:,2:]) / 2  # [N,2]
                roi2 = roi.expand(len(center), 4)  # [N,4]
                mask = (center > roi2[:,:2]) & (center < roi2[:,2:])  # [N,2]
                mask = mask[:,0] & mask[:,1]  #[N,]
                if not mask.any():
                    continue

                selected_boxes = boxes.index_select(0, mask.nonzero().squeeze(1))

                iou = self.data_encoder.iou(selected_boxes, roi)
                if iou.min() < min_iou:
                    continue

                img = img.crop((x, y, x+w, y+h))
                selected_boxes[:,0].add_(-x).clamp_(min=0, max=w)
                selected_boxes[:,1].add_(-y).clamp_(min=0, max=h)
                selected_boxes[:,2].add_(-x).clamp_(min=0, max=w)
                selected_boxes[:,3].add_(-y).clamp_(min=0, max=h)
                return img, selected_boxes, labels[mask]

    def __len__(self):
        return self.num_samples
