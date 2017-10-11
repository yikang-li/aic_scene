import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as pytorch_models
from .utils import set_trainable, set_trainable_param

import pdb

import sys
sys.path.append('external/pretrained_models')
import pretrainedmodels as torch7_models

pytorch_resnet_names = sorted(name for name in pytorch_models.__dict__
    if name.islower()
    and name.startswith("resnet")
    and callable(pytorch_models.__dict__[name])) #     

torch7_resnet_names = sorted(name for name in torch7_models.__dict__
    if name.islower()
    and callable(torch7_models.__dict__[name]))

model_names = pytorch_resnet_names + torch7_resnet_names


def factory(arch, dilation=1):

    if arch in pytorch_resnet_names:
        model = pytorch_models.__dict__[arch](pretrained=True)

    elif arch == 'densenet_places':
        if not os.access('./data/models/densenet161_places365.pth3.tar', os.R_OK):
            os.system('mkdir -p ./data/models/')
            os.system('wget https://www.dropbox.com/s/m8lraqezvu4hlax/densenet161_places365.pth3.tar?dl=0')
            os.system('mv densenet161_places365.pth3.tar ./data/models/')

        model = torch.load('./data/models/densenet161_places365.pth3.tar')

    elif arch == 'fbresnet152':
        model = torch7_models.__dict__[arch](num_classes=1000,
                                                    pretrained='imagenet')

    elif arch in torch7_resnet_names:
        model = torch7_models.__dict__[arch](num_classes=1000,
                                                    pretrained='imagenet')
    else:
        raise ValueError

    # As utilizing the pretrained_model on 224 image, 
    # when applied on 448 images, please set the corresponding [dilation]
    set_dilation(model, dilation)
    if 'resnet' in arch:
        return ResNet(model), model.fc.in_features
    elif arch == 'densenet_places':
        return model.features, model.classifier.in_features
    else:
        raise ValueError

def set_dilation(model, dilation=1):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.dilation = (dilation, dilation)
            m.padding = tuple([item * dilation for item in m.padding])

class ResNet(nn.Module):
    def __init__(self, convnet):
        super(ResNet, self).__init__()
        self.conv1 = convnet.conv1
        self.bn1 = convnet.bn1
        self.relu = convnet.relu
        self.maxpool = convnet.maxpool
        self.layer1 = convnet.layer1
        self.layer2 = convnet.layer2
        self.layer3 = convnet.layer3
        self.layer4 = convnet.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
