import torch
import torch.nn as nn
import os

class AdvancedConv2DModel(nn.Module):
    def __init__(self, name, num_classes=7, dtype=torch.float16):
        super(AdvancedConv2DModel, self).__init__()

        self.name = 'AdvancedConv2D_' + name
        self.dtype = dtype  # Set the desired data type

        # Conv2D Layer 1: input channels = 3 (RGB), output channels = 64
        self.conv2d_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1, padding=1).to(dtype=self.dtype)
        self.bn1 = nn.BatchNorm2d(64).to(dtype=self.dtype)
        self.relu1 = nn.ReLU().to(dtype=self.dtype)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)).to(dtype=self.dtype)

        # Conv2D Layer 2: input channels = 64, output channels = 128
        self.conv2d_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1).to(dtype=self.dtype)
        self.bn2 = nn.BatchNorm2d(128).to(dtype=self.dtype)
        self.relu2 = nn.ReLU().to(dtype=self.dtype)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)).to(dtype=self.dtype)

        # Conv2D Layer 3: input channels = 128, output channels = 256
        self.conv2d_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1, padding=1).to(dtype=self.dtype)
        self.bn3 = nn.BatchNorm2d(256).to(dtype=self.dtype)
        self.relu3 = nn.ReLU().to(dtype=self.dtype)
        self.maxpool_3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)).to(dtype=self.dtype)

        # Conv2D Layer 4: input channels = 256, output channels = 512
        self.conv2d_4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=1, padding=1).to(dtype=self.dtype)
        self.bn4 = nn.BatchNorm2d(512).to(dtype=self.dtype)
        self.relu4 = nn.ReLU().to(dtype=self.dtype)
        self.maxpool_4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)).to(dtype=self.dtype)

        # Flatten the output
        self.flatten = nn.Flatten().to(dtype=self.dtype)

        # Fully connected layers
        self.fc1 = nn.Linear(512 * 2 * 16, 1024).to(dtype=self.dtype)  # Adjust based on input size after conv layers
        self.relu_fc1 = nn.ReLU().to(dtype=self.dtype)
        self.dropout = nn.Dropout(0.5).to(dtype=self.dtype)

        self.fc2 = nn.Linear(1024, num_classes).to(dtype=self.dtype)

    def forward(self, x):
        x = x.to(self.dtype)

        # Pass through Conv2D and MaxPool layers
        x = self.conv2d_1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool_1(x)

        x = self.conv2d_2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool_2(x)

        x = self.conv2d_3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.maxpool_3(x)

        x = self.conv2d_4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.maxpool_4(x)

        # Flatten the feature maps
        x = self.flatten(x)

        # Fully connected layers
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def get_dir(self):
        path = 'models/' + self.name + '/'
        os.makedirs(path, exist_ok=True)
        return path
