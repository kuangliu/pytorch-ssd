import math
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from multibox import MultiBox


class SSD300(nn.Module):
    input_size = 300

    def __init__(self):
        super(SSD300, self).__init__()
        self.base = self.VGG16()

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

        # Prepare default boxes.
        feature_map_sizes = [38, 19, 10, 5, 3, 1]
        aspect_ratios = [(2,),(2,3),(2,3),(2,3),(2,),(2,)]
        self.default_boxes = self._get_default_boxes(feature_map_sizes, aspect_ratios)
        print('default_boxes size:')
        print(self.default_boxes.size())

    def _get_default_boxes(self, feature_map_sizes, aspect_ratios):
        '''Compute default box sizes with scale and aspect transform.

        Args:
          feature_map_sizes: (list) size of each intermediate layer output.
          aspect_ratios: (list) aspect ratios.

        Returns:
            (tensor) of default boxes, sized [8732,4].
        '''
        input_size = self.input_size

        num_layers = len(feature_map_sizes)
        min_ratio = 20
        max_ratio = 90
        step = (max_ratio-min_ratio) / (num_layers-2)

        min_sizes = [input_size * 0.1]
        max_sizes = [input_size * 0.2]
        for ratio in range(min_ratio, max_ratio+1, step):
            min_sizes.append(input_size * ratio / 100.)
            max_sizes.append(input_size * (ratio+step) / 100.)

        boxes = []
        for i in range(num_layers):
            fmsize = feature_map_sizes[i]
            for h,w in itertools.product(range(fmsize), repeat=2):
                cx = (w + 0.5) / fmsize
                cy = (h + 0.5) / fmsize

                box_size = min_sizes[i] / input_size
                boxes.append((cx, cy, box_size, box_size))

                for ar in aspect_ratios[i]:
                    boxes.append((cx, cy, box_size * math.sqrt(ar), box_size / math.sqrt(ar)))
                    boxes.append((cx, cy, box_size / math.sqrt(ar), box_size * math.sqrt(ar)))

                box_size = math.sqrt(min_sizes[i] * max_sizes[i]) / input_size
                boxes.append((cx, cy, box_size, box_size))
        return torch.Tensor(boxes)

    def forward(self, x):
        hs = []  # cache intermediate layers outputs

        h = self.base(x)
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

        print('\nintermediate feature map sizes:')
        for x in hs:
            print(x.size())

        loc_preds, conf_preds = self.multibox(hs)
        return loc_preds, conf_preds

    def encode(self, boxes, classes, threshold=0.5):
        '''Transform target bounding boxes and class labels to SSD boxes and classes.

        Match each object box to all the default boxes, pick the ones with the
        Jaccard-Index > 0.5:
            Jaccard(A,B) = AB / (A+B-AB)

        Args:
          boxes: (tensor) object bounding boxes (x1,y1,x2,y2) of a image, sized [#obj, 4].
          classes: (tensor) object class labels of a image, sized [#obj,].
          threshold: (float) Jaccard index threshold

        Returns:
          boxes: (tensor) bounding boxes, sized [#obj, 8732, 4].
          classes: (tensor) class labels, sized [8732,]
        '''
        default_boxes = self.default_boxes
        num_default_boxes = default_boxes.size(0)
        num_objs = boxes.size(0)

        lt = default_boxes[:,:2] - default_boxes[:,2:]/2  # [8732,2]
        lt = torch.max(
            lt.unsqueeze(0).expand(num_objs, num_default_boxes, 2),          # [8732,2] -> [1,8732,2] -> [#obj,8732,2]
            boxes[:,:2].unsqueeze(1).expand(num_objs, num_default_boxes, 2)  # [#obj,2] -> [#obj,1,2] -> [#obj,8732,2]
        )

        rb = default_boxes[:,:2] + default_boxes[:,2:]/2  # [8732,2]
        rb = torch.min(
            rb.unsqueeze(0).expand(num_objs, num_default_boxes, 2),          # [8732,2] -> [1,8732,2] -> [#obj,8732,2]
            boxes[:,2:].unsqueeze(1).expand(num_objs, num_default_boxes, 2)  # [#obj,2] -> [#obj,1,2] -> [#obj,8732,2]
        )

        wh = rb - lt  # [#obj,8732,2]
        wh[wh<0] = 0  # clip at 0
        inter_areas = wh[:,:,0] * wh[:,:,1]  # [#obj,8732]

        default_box_areas = default_boxes[:,2] * default_boxes[:,3]        # [8732,]
        obj_box_areas = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])  # [#obj,]

        default_box_areas = default_box_areas.unsqueeze(0).expand_as(inter_areas)  # [8732,] -> [1,8732] -> [#obj,8732]
        obj_box_areas = obj_box_areas.unsqueeze(1).expand_as(inter_areas)          # [#obj,] -> [#obj,1] -> [#obj,8732]

        iou = inter_areas / (default_box_areas + obj_box_areas - inter_areas)  # [#obj,8732]

        iou, max_idx = iou.max(0)  # [1,8732]
        max_idx.squeeze_(0)        # [8732,]
        iou.squeeze_(0)            # [8732,]

        boxes = boxes[max_idx]     # [8732,4]
        # (x1,y1,x2,y2) -> (cx,cy,w,h)
        variances = [0.1, 0.2]
        xy = (boxes[:,:2] + boxes[:,2:]) / 2 - default_boxes[:,:2]  # [8732,2]
        xy /= (variances[0] * default_boxes[:,2:])
        wh = (boxes[:,2:] - boxes[:,:2]) / default_boxes[:,2:]      # [8732,2]
        wh = torch.log(wh) / variances[1]
        loc = torch.cat([xy, wh], 1)  # [8732,4]

        conf = 1 + classes[max_idx]   # [8732,], background class = 0
        conf[iou<threshold] = 0       # background
        return loc, conf

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


def test_ssd():
    net = SSD300()
    x = torch.Tensor(1,3,300,300)
    loc_preds, conf_preds = net(Variable(x))
    print('\nloc_preds:')
    print(loc_preds.size())
    print('\nconf_preds:')
    print(conf_preds.size())

    boxes = torch.Tensor([[0,0,0.4,0.4], [0.2,0.2,0.8,0.8]])  # x y x y  [nobj,4]
    classes = torch.LongTensor([0,1])
    loc, conf = net.encode(boxes, classes)
    print('\nencoded data:')
    print(loc.size())
    print(conf.size())

test_ssd()
