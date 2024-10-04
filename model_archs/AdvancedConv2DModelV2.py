import torch
import torch.nn as nn
import os
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

# Define the advanced Conv2D architecture with added techniques
class AdvancedConv2DModelV2(nn.Module):
    def __init__(self, name, num_classes=7, dtype=torch.float32):
        super(AdvancedConv2DModelV2, self).__init__()

        self.name = 'AdvancedConv2DModelV2' + name
        self.dtype = dtype

        # Conv2D Block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1).to(dtype=self.dtype)
        self.bn1 = nn.BatchNorm2d(64).to(dtype=self.dtype)
        self.relu1 = nn.ReLU().to(dtype=self.dtype)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2).to(dtype=self.dtype)

        # Conv2D Block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1).to(dtype=self.dtype)
        self.bn2 = nn.BatchNorm2d(128).to(dtype=self.dtype)
        self.relu2 = nn.ReLU().to(dtype=self.dtype)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2).to(dtype=self.dtype)

        # Conv2D Block 3
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1).to(dtype=self.dtype)
        self.bn3 = nn.BatchNorm2d(256).to(dtype=self.dtype)
        self.relu3 = nn.ReLU().to(dtype=self.dtype)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2).to(dtype=self.dtype)

        # Conv2D Block 4
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1).to(dtype=self.dtype)
        self.bn4 = nn.BatchNorm2d(512).to(dtype=self.dtype)
        self.relu4 = nn.ReLU().to(dtype=self.dtype)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1)).to(dtype=self.dtype)

        # Fully Connected Layer
        self.fc1 = nn.Linear(512, 1024).to(dtype=self.dtype)
        self.relu_fc1 = nn.ReLU().to(dtype=self.dtype)
        self.dropout = nn.Dropout(0.6).to(dtype=self.dtype)  # Increased dropout for better regularization

        self.fc2 = nn.Linear(1024, num_classes).to(dtype=self.dtype)

    def forward(self, x):
        x = x.to(self.dtype)

        # Forward through each layer
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)

        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.maxpool2(x)

        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool3(x)

        x = self.relu4(self.bn4(self.conv4(x)))

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
