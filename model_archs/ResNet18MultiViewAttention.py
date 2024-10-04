import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# SeparableConv2D block for reducing computation
class SeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(SeparableConv2D, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Channel Attention Mechanism
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

# Spatial Attention Mechanism
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# Residual Block for ResNet18
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResNetBlock, self).__init__()
        self.downsample = downsample
        stride = 2 if downsample else 1
        
        self.conv1 = SeparableConv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = SeparableConv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        if self.downsample:
            self.downsample_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
            self.downsample_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            identity = self.downsample_conv(x)
            identity = self.downsample_bn(identity)

        out += identity
        out = self.relu(out)
        return out
        
class ResNet18MultiViewAttention(nn.Module):
    def __init__(self, name, num_classes=6, dtype=torch.float32):
        super(ResNet18MultiViewAttention, self).__init__()

        self.name = 'ResNet18MultiViewAttention' + name

        # First view stream
        self.conv1_view1 = SeparableConv2D(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1_view1 = nn.BatchNorm2d(64)
        self.layer1_view1 = self._make_layer(64, 64)
        self.layer2_view1 = self._make_layer(64, 128, downsample=True)
        self.layer3_view1 = self._make_layer(128, 256, downsample=True)
        self.layer4_view1 = self._make_layer(256, 512, downsample=True)

        # Second view stream
        self.conv1_view2 = SeparableConv2D(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1_view2 = nn.BatchNorm2d(64)
        self.layer1_view2 = self._make_layer(64, 64)
        self.layer2_view2 = self._make_layer(64, 128, downsample=True)
        self.layer3_view2 = self._make_layer(128, 256, downsample=True)
        self.layer4_view2 = self._make_layer(256, 512, downsample=True)

        # Attention Mechanisms for Each View
        self.channel_attention1 = ChannelAttention(512)
        self.spatial_attention1 = SpatialAttention()

        self.channel_attention2 = ChannelAttention(512)
        self.spatial_attention2 = SpatialAttention()

        # Global Average Pooling for fusion
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer
        self.fc = nn.Linear(512 * 2, num_classes)

    def _make_layer(self, in_channels, out_channels, downsample=False):
        return ResNetBlock(in_channels, out_channels, downsample)

    def forward(self, x):
        # Input: [batch_size, 3, 2, 32, 256] -> 3 heatmaps, 2 views
        view1 = x[:, :, 0, :, :]  # Extract view 1: [batch_size, 3, 32, 256]
        view2 = x[:, :, 1, :, :]  # Extract view 2: [batch_size, 3, 32, 256]

        # Process the first view
        out1 = self.conv1_view1(view1)
        out1 = self.bn1_view1(out1)
        out1 = self.layer1_view1(out1)
        out1 = self.layer2_view1(out1)
        out1 = self.layer3_view1(out1)
        out1 = self.layer4_view1(out1)
        out1 = self.channel_attention1(out1) * self.spatial_attention1(out1)

        # Process the second view
        out2 = self.conv1_view2(view2)
        out2 = self.bn1_view2(out2)
        out2 = self.layer1_view2(out2)
        out2 = self.layer2_view2(out2)
        out2 = self.layer3_view2(out2)
        out2 = self.layer4_view2(out2)
        out2 = self.channel_attention2(out2) * self.spatial_attention2(out2)

        # Global average pooling and concatenate both views
        out1 = self.global_avg_pool(out1)  # Result: [batch_size, 512, 1, 1]
        out2 = self.global_avg_pool(out2)  # Result: [batch_size, 512, 1, 1]

        # Flatten and concatenate
        out1 = torch.flatten(out1, 1)  # Shape: [batch_size, 512]
        out2 = torch.flatten(out2, 1)  # Shape: [batch_size, 512]

        # Concatenate both flattened outputs
        out = torch.cat((out1, out2), dim=1)  # Shape: [batch_size, 1024]

        # Final fully connected layer
        out = self.fc(out)

        return out

    def get_dir(self):
        path = 'models/' + self.name + '/'
        os.makedirs(path, exist_ok=True)
        return path
