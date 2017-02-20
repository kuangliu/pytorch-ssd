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

    def cross_entropy_loss(x,y):
        '''Cross entropy loss w/o averaging across all examples.

        Args:
          x: (tensor Variable) sized [N,D].
          y: (tensor Variable) sized [N,].

        Return:
          (tensor Variable) Cross entroy loss, sized [N,].
        '''
        xmax = x.max().data[0]
        log_sum_exp = torch.log(torch.sum(torch.exp(x-xmax), 1)) + xmax
        return log_sum_exp - x.gather(1, y.view(-1,1))

    def test_cross_entropy_loss():
        a = Variable(torch.randn(10,4))
        b = Variable(torch.ones(10).long())
        loss = cross_entropy_loss(a,b)
        print(loss.mean())
        print(F.cross_entropy(a,b))

    def loss(loc_preds, loc_targets, conf_preds, conf_targets):
        '''Compute loss between (loc_preds, loc_targets) and (conf_preds, conf_targets).

        Args:
          loc_preds: (tensor) predicted locations, sized [#samples, 8732, 4].
          loc_targets: (tensor) encoded target locations, sized [#samples, 8732, 4].
          conf_preds: (tensor) predicted class confidences, sized [#samples, 8732, #classes].
          conf_targets: (tensor) encoded target classes, sized [#samples, 8732].

        loss:
          (Variable) SmoothL1Loss(loc_preds, loc_targets) + SoftmaxLoss(conf_preds, conf_targets).
        '''
        batch_size, num_boxes, _ = loc_preds.size()

        pos = conf_targets>0  # [N,8732]
        pos_mask = pos.unsqueeze(2).expand_as(loc_preds)  # [N,8732,4]
        loc_preds_pos = loc_preds[pos_mask].view(-1,4)  # [4390,4]
        loc_targets_pos = loc_targets[pos_mask].view(-1,4)  # [4390,4]

        loc_loss = F.smooth_l1_loss(loc_preds_pos, loc_targets_pos, size_average=False)
        conf_loss = F.cross_entropy(conf_preds.view(-1,self.num_classes), conf_targets.view(-1), size_average=False)
        return loc_loss, conf_loss
