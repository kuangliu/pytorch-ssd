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

        # Augmentation.
        img, boxes = self.random_flip(img, boxes)
        img, boxes = self.random_crop(img, boxes)

        # Scale bbox locaitons to [0,1].
        w,h = img.size
        boxes /= torch.Tensor([w,h,w,h]).expand_as(boxes)

        # RGB to BGR for pretrained VGG16 model.
        img = np.array(img)
        img = img[:,:,::-1]
        img = Image.fromarray(img)

        img = img.resize((self.img_size,self.img_size))
        img = self.transform(img)

        # Encode loc & conf targets.
        loc_target, conf_target = self.data_encoder.encode(boxes, self.labels[idx])
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
            xmin = boxes[:,0]
            xmax = boxes[:,2]
            xmin, xmax = w-xmax, w-xmin
        return img, boxes

    def random_crop(self, img, boxes, padding=4):
        '''Randomly crop the image and adjust the bbox locations.

        Args:
          img: (PIL.Image) image.
          boxes: (tensor) bbox locations, sized [#obj, 4].
          padding: (int) number of padding pixels.

        Returns:
          img: (PIL.Image) randomly cropped image.
          boxes: (tensor) randomly cropped bbox locations, sized [#obj, 4].
        '''
        w, h = img.size
        img = ImageOps.expand(img, border=padding, fill=0)
        x1 = random.randint(0, 2*padding)
        y1 = random.randint(0, 2*padding)
        img = img.crop((x1, y1, x1 + w, y1 + h))
        boxes[:,0] += padding - x1
        boxes[:,1] += padding - y1
        boxes[:,2] += padding - x1
        boxes[:,3] += padding - y1
        return img, boxes

    def __len__(self):
        return self.num_samples
