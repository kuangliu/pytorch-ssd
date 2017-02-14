import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class MultiBox(nn.Module):
    num_classes = 21
    num_anchors = [4,6,6,6,4,4]
    # output channels of layer [conv4_3, conv7, conv8_2, conv9_2, conv10_2, avgpool]
    in_channels = [512,1024,512,256,256,256]

    def __init__(self):
        super(MultiBox, self).__init__()
        # Add prediction heads.
        self.loc_layers = []
        self.conf_layers = []
        for i,n in zip(self.in_channels, self.num_anchors):
            self.loc_layers.append(nn.Conv2d(i, n*4, kernel_size=3, padding=1))
            self.conf_layers.append(nn.Conv2d(i, n*self.num_classes, kernel_size=3, padding=1))

    def forward(self, xs):
        '''
        Args:
          xs: (list) of tensor containing intermediate layer outputs.

        Returns:
          loc_preds: (tensor) predicted locations, sized [N,8732,4].
          conf_preds: (tensor) predicted class confidences, sized [N,8732,21].
        '''
        loc_preds = []
        conf_preds = []
        for i,x in enumerate(xs):
            loc_pred = self.loc_layers[i](x)
            N,C,H,W = loc_pred.size()
            loc_pred.permute(0,2,3,1)  # Permute: [N,C,H,W] -> [N,H,W,C]
            loc_pred = loc_pred.resize(N, C*H*W//4, 4)  # Resize: [N,H,W,C] -> [N,H*W*n,4]  while n = C/4.
            loc_preds.append(loc_pred)

            conf_pred = self.conf_layers[i](x)
            N,C,H,W = conf_pred.size()
            conf_pred.permute(0,2,3,1)  # Permute: [N,C,H,W] -> [N,H,W,C]
            conf_pred = conf_pred.resize(N, C*H*W//self.num_classes, self.num_classes)  # Resize: [N,H,W,C] -> [N,H*W*n,21]  while n = C/21.
            conf_preds.append(conf_pred)

        loc_preds = torch.cat(loc_preds, 1)
        conf_preds = torch.cat(conf_preds, 1)
        print('\nloc_preds size:')
        print(loc_preds.size())
        print('\nconf_preds size:')
        print(conf_preds.size())
        return loc_preds, conf_preds
