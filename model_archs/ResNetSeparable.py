import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# SeparableConv2D block with kernel and bias initialization
class SeparableConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(SeparableConv2D, self).__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Initialize the kernel and bias
        nn.init.kaiming_normal_(self.depthwise.weight, nonlinearity='relu')  
        nn.init.kaiming_normal_(self.pointwise.weight, nonlinearity='relu')  
        nn.init.constant_(self.depthwise.bias, 0.2)
        nn.init.constant_(self.pointwise.bias, 0.2)

    def forward(self, x):
        # print(f"Input to SeparableConv2D: {x.shape}")
        x = self.depthwise(x)
        x = self.pointwise(x)
        # print(f"Output of SeparableConv2D: {x.shape}")
        return x

# Residual Block with SeparableConv2D
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super(ResNetBlock, self).__init__()
        self.downsample = downsample
        stride = 2 if downsample else 1
        
        self.conv1 = SeparableConv2D(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = SeparableConv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Adjust the identity path if downsampling or if input and output channels are different
        if downsample or in_channels != out_channels:
            self.downsample_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
            self.downsample_bn = nn.BatchNorm2d(out_channels)
        else:
            self.downsample_conv = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply downsample if necessary
        if self.downsample_conv is not None:
            identity = self.downsample_conv(x)
            identity = self.downsample_bn(identity)

        out += identity
        out = F.relu(out)

        return out

# ResNet-like network using SeparableConv2D for Multi-View Radar Data
class ResNetSeparable(nn.Module):
    def __init__(self, name, num_classes=6, dtype=torch.float32):
        super(ResNetSeparable, self).__init__()
        self.name = "ResNetSeparable" + name

        # First view stream
        self.conv1_view1 = SeparableConv2D(3, 32, kernel_size=7, stride=2, padding=3)
        self.bn1_view1 = nn.BatchNorm2d(32)
        self.layer1_view1 = self._make_layer(32, 64)
        self.layer2_view1 = self._make_layer(64, 128, downsample=True)
        self.layer3_view1 = self._make_layer(128, 256, downsample=True)
        self.layer4_view1 = self._make_layer(256, 512, downsample=True)

        # Second view stream
        self.conv1_view2 = SeparableConv2D(3, 32, kernel_size=7, stride=2, padding=3)
        self.bn1_view2 = nn.BatchNorm2d(32)
        self.layer1_view2 = self._make_layer(32, 64)
        self.layer2_view2 = self._make_layer(64, 128, downsample=True)
        self.layer3_view2 = self._make_layer(128, 256, downsample=True)
        self.layer4_view2 = self._make_layer(256, 512, downsample=True)

        # Global Average Pooling for both views
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer
        self.fc = nn.Linear(512 * 2, num_classes)

    def _make_layer(self, in_channels, out_channels, downsample=False):
        return ResNetBlock(in_channels, out_channels, downsample=downsample)

    def forward(self, x):
        # print(f"Input shape: {x.shape}")  # Expecting [batch_size, 3, 2, 32, 256]

        # Split into two views
        view1 = x[:, :, 0, :, :]  # [batch_size, 3, 32, 256]
        view2 = x[:, :, 1, :, :]  # [batch_size, 3, 32, 256]
        
        # Process view 1
        out1 = self.conv1_view1(view1)
        out1 = self.bn1_view1(out1)
        out1 = self.layer1_view1(out1)
        out1 = self.layer2_view1(out1)
        out1 = self.layer3_view1(out1)
        out1 = self.layer4_view1(out1)
        out1 = self.global_avg_pool(out1)
        out1 = torch.flatten(out1, 1)
        # print(f"After processing view1, shape: {out1.shape}")

        # Process view 2
        out2 = self.conv1_view2(view2)
        out2 = self.bn1_view2(out2)
        out2 = self.layer1_view2(out2)
        out2 = self.layer2_view2(out2)
        out2 = self.layer3_view2(out2)
        out2 = self.layer4_view2(out2)
        out2 = self.global_avg_pool(out2)
        out2 = torch.flatten(out2, 1)
        # print(f"After processing view2, shape: {out2.shape}")

        # Concatenate both views
        out = torch.cat((out1, out2), dim=1)
        # print(f"After concatenating both views, shape: {out.shape}")

        # Final fully connected layer
        out = self.fc(out)
        # print(f"Final output shape: {out.shape}")

        return out

    def get_dir(self):
        path = 'models/' + self.name + '/'
        os.makedirs(path, exist_ok=True)
        return path
