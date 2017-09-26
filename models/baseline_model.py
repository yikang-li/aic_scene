import torch
import torch.nn as nn

from .convnets import ResNet
from .utils import FC

class Baseline(nn.Module):

    def __init__(self, opts):
        super(Baseline, self).__init__()
        self.features = ResNet(opts['conv_arch'], pooling=opts['pooling'])
        self.classifier = FC(1024, 80, relu=False)
        self.features.set_trainable(True)

    def forward(self, imgs):
        imgs = self.features(imgs)
        pred = self.classifier(imgs)
        return pred