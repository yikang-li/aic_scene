import torch
import torch.nn as nn
import torch.nn.functional as F

from .convnets import ResNet
from .utils import FC


class Baseline_v1(nn.Module):

    def __init__(self, opts):
        super(Baseline_v1, self).__init__()
        self.features = ResNet(opts['conv_arch'], pooling=opts['pooling'])
        self.classifier = FC(2048, 80, relu=False)
        self.features.set_trainable(True)
        self.opts = opts

    def forward(self, imgs):
        imgs = self.features(imgs)
        if 'dropout' in opts.keys():
        	imgs = F.dropout(imgs, opts['dropout'], training=self.training)
        pred = self.classifier(imgs)
        return pred


class Baseline_v2(Baseline_v1):

    def __init__(self, opts):
        super(Baseline_v2, self).__init__(opts)
        self.fc6 = FC(2048, 2048, relu=False)

    def forward(self, imgs):
        imgs = F.relu(self.features(imgs))
        if 'dropout' in opts.keys():
        	imgs = F.dropout(imgs, opts['dropout'], training=self.training)
        imgs = self.fc6(imgs)
        if 'dropout' in opts.keys():
        	imgs = F.dropout(imgs, opts['dropout'], training=self.training)
        pred = self.classifier(imgs)
        return pred