from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from torch.autograd import grad
import logging
from torch import ones, eye
#from sbi import utils as utils

logger = logging.getLogger(__name__)


import torch.nn as nn


def is_positive_int(x):
    return isinstance(x, int) and x > 0

def merge_leading_dims(x, num_dims):
    """Reshapes the tensor `x` such that the first `num_dims` dimensions are merged to
    one."""
    if not is_positive_int(num_dims):
        raise TypeError("Number of leading dims must be a positive integer.")
    if num_dims > x.dim():
        raise ValueError(
            "Number of leading dims can't be greater than total number of dims."
        )
    new_shape = torch.Size([-1]) + x.shape[num_dims:]
    return torch.reshape(x, new_shape)


def repeat_rows(x, num_reps):
    """Each row of tensor `x` is repeated `num_reps` times along leading dimension."""
    if not is_positive_int(num_reps):
        raise TypeError("Number of repetitions must be a positive integer.")
    shape = x.shape
    x = x.unsqueeze(1)
    x = x.expand(shape[0], num_reps, *shape[1:])
    return merge_leading_dims(x, num_dims=2)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, p_dropout=0):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        
        if (p_dropout < 0 or p_dropout > 1): raise ValueError('dropout rate of %f needs to be between 0 and 1' % (p_dropout))
        self.dropout_bool = p_dropout > 0
        if (self.dropout_bool): self.dropout = nn.Dropout2d(p=p_dropout)

    def forward(self, x):
        identity = x

        if self.dropout_bool: x = self.dropout(x) 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.dropout_bool: out = self.dropout(out) 
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, p_dropout=0):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
        if (p_dropout < 0 or p_dropout > 1): raise ValueError('dropout rate of %f needs to be between 0 and 1' % (p_dropout))
        self.dropout_bool = p_dropout > 0
        if (self.dropout_bool): self.dropout = nn.Dropout2d(p=p_dropout)

    def forward(self, x):
        identity = x

        if self.dropout_bool: x = self.dropout(x) 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.dropout_bool: out = self.dropout(out) 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.dropout_bool: out = self.dropout(out) 
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetEstimator(nn.Module):
    def __init__(self, n_aux=0, cfg=18, n_hidden=2048, n_out=128, input_mean=None, input_std=None, log_input=False, zero_init_residual=False, norm_layer=None, zero_bias=False):
        super(ResNetEstimator, self).__init__()

        self.input_mean = input_mean
        self.input_std = input_std
        self.log_input = log_input

        block, layers = self._load_cfg(cfg)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512 * block.expansion + n_aux, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if zero_bias:
                    try:
                        nn.init.constant_(m.bias, 0)
                    except AttributeError:
                        pass
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if zero_bias:
                    try:
                        nn.init.constant_(m.bias, 0)
                    except AttributeError:
                        pass

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x, x_aux=None):

        # Preprocessing
        h = self._preprocess(x)

        # ResNet
        h = self.conv1(h)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.maxpool(h)

        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)

        h = self.avgpool(h)
        h = h.view(h.size(0), -1)
        if x_aux is not None:
            h = torch.cat((h, x_aux), 1)
        h = self.fc1(h)
        h = self.relu(h)
        h = self.fc2(h)

        return h

    @staticmethod
    def _load_cfg(cfg):
        if cfg == 18:
            block = BasicBlock
            layers = [2, 2, 2, 2]
        elif cfg == 34:
            block = BasicBlock
            layers = [3, 4, 6, 3]
        elif cfg == 50:
            block = Bottleneck
            layers = [3, 4, 6, 3]
        elif cfg == 101:
            block = Bottleneck
            layers = [3, 4, 23, 3]
        elif cfg == 152:
            block = Bottleneck
            layers = [3, 8, 36, 3]
        else:
            raise ValueError("Unknown ResNet configuration {}, use 18, 34, 50, 101, or 152!".format(cfg))

        return block, layers

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _preprocess(self, x):
        if self.log_input:
            x = torch.log(1.0 + x)
        if self.input_mean is not None and self.input_std is not None:
            x = x - self.input_mean
            x = x / self.input_std
        x = x.unsqueeze(1)
        return x
    
    
class ResNetRatioEstimator(nn.Module):
    def __init__(self, n_aux=0, cfg=18, n_hidden=512, n_out=128, input_mean=None, input_std=None, log_input=False, zero_init_residual=False, norm_layer=None, zero_bias=False, p_dropout=0):
        super(ResNetRatioEstimator, self).__init__()

        self.input_mean = input_mean
        self.input_std = input_std
        self.log_input = log_input

        block, layers = self._load_cfg(cfg)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, p_dropout=p_dropout)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer, p_dropout=p_dropout)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer, p_dropout=p_dropout)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer, p_dropout=p_dropout)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512 * block.expansion + n_aux, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)
        self.sigmoid = nn.Sigmoid()
        
        if (p_dropout < 0 or p_dropout > 1): raise ValueError('dropout rate of %f needs to be between 0 and 1' % (p_dropout))
        self.dropout_bool = p_dropout > 0
        if (self.dropout_bool): 
            self.dropout2d = nn.Dropout2d(p=p_dropout)
            self.dropout = nn.Dropout(p=p_dropout) 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if zero_bias:
                    try:
                        nn.init.constant_(m.bias, 0)
                    except AttributeError:
                        pass
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if zero_bias:
                    try:
                        nn.init.constant_(m.bias, 0)
                    except AttributeError:
                        pass

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x, x_aux=None, train=True):
        # double the dataset into half original and half shuffled (marginal distribution) alternating 
        if (train):
            x, x_aux = self._classifier_data(x_aux, x) 
        
        # Preprocessing
        h = self._preprocess(x)

        # ResNet
        if (self.dropout_bool): h = self.dropout2d(h)
        h = self.conv1(h)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.maxpool(h)

        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)

        h = self.avgpool(h)
        h = h.view(h.size(0), -1)
        if x_aux is not None:
            h = torch.cat((h, x_aux), 1)
            
        if (self.dropout_bool): h = self.dropout(h)
        h = self.fc1(h)
        h = self.relu(h)
        
        if (self.dropout_bool): h = self.dropout(h)
        log_r = self.fc2(h)
        
        # Transform to outputs
        s = self.sigmoid(log_r)

        return s, log_r

    @staticmethod
    def _load_cfg(cfg):
        if cfg == 18:
            block = BasicBlock
            layers = [2, 2, 2, 2]
        elif cfg == 34:
            block = BasicBlock
            layers = [3, 4, 6, 3]
        elif cfg == 50:
            block = Bottleneck
            layers = [3, 4, 6, 3]
        elif cfg == 101:
            block = Bottleneck
            layers = [3, 4, 23, 3]
        elif cfg == 152:
            block = Bottleneck
            layers = [3, 8, 36, 3]
        else:
            raise ValueError("Unknown ResNet configuration {}, use 18, 34, 50, 101, or 152!".format(cfg))

        return block, layers

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None, p_dropout=0):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer, p_dropout=p_dropout))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer, p_dropout=p_dropout))

        return nn.Sequential(*layers)

    def _preprocess(self, x):
        if self.log_input:
            x = torch.log(1.0 + x)
        if self.input_mean is not None and self.input_std is not None:
            x = x - self.input_mean
            x = x / self.input_std
        x = x.unsqueeze(1)
        return x
    
    
    # this encodes the main modifications to standard resnet 
    def _classifier_data(self, theta, x):
        """Convert sets of (x, theta) pairs into alternating pairs from p(x,theta) and p(x)p(theta). The size 
        of dataset is doubled in the process. 
        """
        batch_size = theta.shape[0]
        repeated_x = repeat_rows(x, 2)

        probs = ones(batch_size, batch_size) * (1 - eye(batch_size)) / (batch_size - 1)

        choices = torch.multinomial(probs, num_samples=1, replacement=False)

        contrasting_theta = theta[choices]

        atomic_theta = torch.cat((theta[:, None, :], contrasting_theta), dim=1).reshape(batch_size*2, -1)

        return repeated_x, atomic_theta
    
    
