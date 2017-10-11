import torch
import torch.nn as nn
import torch.nn.functional as F

from .convnets import factory
from .utils import FC

import pdb

class Baseline_v1(nn.Module):

    def __init__(self, opts):
        super(Baseline_v1, self).__init__()
        self.features, dim_h = factory(opts['conv_arch'], dilation=opts.get('dilation', 1))
        self.classifier = FC(dim_h, 80, relu=False)
        self.opts = opts
        self.pooling = self.opts['pooling']


    def forward(self, imgs):
        imgs = self.features(imgs)
        if self.pooling:
            imgs = F.avg_pool2d(imgs, (imgs.size(2), imgs.size(3)))
            imgs = imgs.view(imgs.size(0), -1)
        if 'dropout' in self.opts.keys():
        	imgs = F.dropout(imgs, self.opts['dropout'], training=self.training)
        pred = self.classifier(imgs)
        return pred
