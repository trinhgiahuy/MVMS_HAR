import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

# SeparableConv2D block with kernel and bias initialization
class SeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SeparableConv2D, self).__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Initialize weights using He initialization and bias to constant 0.2
        # nn.init.kaiming_normal_(self.depthwise.weight, nonlinearity='relu')
        # nn.init.kaiming_normal_(self.pointwise.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.depthwise.weight)
        nn.init.xavier_uniform_(self.pointwise.weight)
        nn.init.constant_(self.depthwise.bias, 0.2)
        nn.init.constant_(self.pointwise.bias, 0.2)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# Residual Block for ResNet18
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super(ResNetBlock, self).__init__()
        self.downsample = downsample

        self.conv1 = SeparableConv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = SeparableConv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if downsample:
            self.downsample_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
            self.downsample_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample_conv(x)
            identity = self.downsample_bn(identity)

        out += identity
        out = F.relu(out)
        return out


# Full ResNet18-like Architecture using SeparableConv2D
class SingleStream(nn.Module):
    def __init__(self, num_classes=6):
        super(SingleStream, self).__init__()
        
        # Initial convolution layer
        self.conv1 = SeparableConv2D(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet blocks (two blocks for each layer group as in ResNet18)
        self.layer1 = self._make_layer(64, 64, num_blocks=2, downsample=False)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, downsample=True)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, downsample=True)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, downsample=True)

        # Global Average Pooling and Final FC layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, in_channels, out_channels, num_blocks, downsample):
        layers = []
        print(downsample)
        layers.append(ResNetBlock(in_channels, out_channels, stride=2 if downsample else 1, downsample=downsample))
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)
        return x

class ResNet18Separable(nn.Module):
    def __init__(self, name, num_classes=6, dtype=torch.float32):
        super(ResNet18Separable, self).__init__()
        self.name = "ResNet18Separable" + name

        # Single Stream for view 1
        self.stream1_ = SingleStream(num_classes=num_classes)

        # Single Stream for view 2
        self.stream2_ = SingleStream(num_classes=num_classes)
        self.fc = nn.Linear(512 * 2, num_classes)

    def forward(self, x):
        # print(f"Input shape: {x.shape}")  # Expecting [batch_size, 3, 2, 32, 256]

        # Split into two views
        view1 = x[:, :, 0, :, :]  # [batch_size, 3, 32, 256]
        view2 = x[:, :, 1, :, :]  # [batch_size, 3, 32, 256]
        out1 = self.stream1_(view1)
        out2 = self.stream2_(view2)

        out = torch.cat((out1, out2), dim=1)

        out = self.fc(out)

        return out

    def get_dir(self):
        path = 'models/' + self.name + '/'
        os.makedirs(path, exist_ok=True)
        return path