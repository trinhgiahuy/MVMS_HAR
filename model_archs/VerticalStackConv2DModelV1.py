import torch
import torch.nn as nn
import os

# Define the SE block to recalibrate channel-wise feature maps
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16, dtype=torch.float32):
        super(SEBlock, self).__init__()
        self.dtype = dtype
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1).to(dtype=self.dtype)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False).to(dtype=self.dtype)
        self.relu = nn.ReLU().to(dtype=self.dtype)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False).to(dtype=self.dtype)
        self.sigmoid = nn.Sigmoid().to(dtype=self.dtype)

    def forward(self, x):
        b, c, _, _ = x.size()
        se = self.global_avg_pool(x).view(b, c)
        se = self.relu(self.fc1(se))
        se = self.sigmoid(self.fc2(se)).view(b, c, 1, 1)
        return x * se


# Residual Block for skip connections
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dtype=torch.float32):
        super(ResidualBlock, self).__init__()
        self.dtype = dtype

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, bias=False).to(dtype=self.dtype)
        self.bn1 = nn.BatchNorm2d(out_channels).to(dtype=self.dtype)
        self.relu = nn.ReLU().to(dtype=self.dtype)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=False).to(dtype=self.dtype)
        self.bn2 = nn.BatchNorm2d(out_channels).to(dtype=self.dtype)

        # Squeeze-and-Excitation Block
        self.se = SEBlock(out_channels).to(dtype=self.dtype)

        # Downsampling layer (if dimensions of input and output don't match)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False).to(dtype=self.dtype),
            nn.BatchNorm2d(out_channels).to(dtype=self.dtype)
        ) if in_channels != out_channels or stride != 1 else None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply Squeeze-and-Excitation
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class VerticalStackConv2DModelV1(nn.Module):
    def __init__(self, name, num_classes=7, dtype=torch.float32):
        super(VerticalStackConv2DModelV1, self).__init__()

        self.name = 'VerticalStackConv2DModelV1' + name
        self.dtype = dtype

        # Initial Conv2D layer
        self.initial_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1).to(dtype=self.dtype)
        self.initial_bn = nn.BatchNorm2d(64).to(dtype=self.dtype)
        self.initial_relu = nn.ReLU().to(dtype=self.dtype)

        # Residual Blocks with SE blocks
        self.block1 = ResidualBlock(64, 64, stride=2).to(dtype=self.dtype) 
        self.block2 = ResidualBlock(64, 128, stride=2).to(dtype=self.dtype)
        self.block3 = ResidualBlock(128, 256, stride=2).to(dtype=self.dtype)
        self.block4 = ResidualBlock(256, 512, stride=2).to(dtype=self.dtype)

        # Additional Residual Block for more depth
        self.block5 = ResidualBlock(512, 1024, stride=2).to(dtype=self.dtype)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1)).to(dtype=self.dtype)

        # Fully Connected Layer with Higher Dropout
        self.fc1 = nn.Linear(1024, 1024).to(dtype=self.dtype)
        self.relu_fc1 = nn.ReLU().to(dtype=self.dtype)
        self.dropout = nn.Dropout(0.6).to(dtype=self.dtype)

        self.fc2 = nn.Linear(1024, num_classes).to(dtype=self.dtype)

    def forward(self, x):
        x = x.to(self.dtype)

        # Initial Conv2D Layer
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.initial_relu(x)

        # Forward through each Residual Block with SE
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        # Global Average Pooling to reduce the dimensions
        x = self.global_avg_pool(x)

        # Flatten and apply FC layers
        x = torch.flatten(x, 1)
        x = self.relu_fc1(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def get_dir(self):
        path = 'models/' + self.name + '/'
        os.makedirs(path, exist_ok=True)
        return path