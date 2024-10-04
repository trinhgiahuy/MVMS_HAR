import torch
import torch.nn as nn
import os

class AdvancedConv2DModel1(nn.Module):
    def __init__(self, name, num_classes=7, dtype=torch.float16):
        super(AdvancedConv2DModel1, self).__init__()

        self.name = 'AdvancedConv2DModel1' + name
        self.dtype = dtype  # Set the desired data type

        # Conv2D Layer 1
        self.conv2d_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.maxpool_1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Conv2D Layer 2
        self.conv2d_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.maxpool_2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Conv2D Layer 3
        self.conv2d_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()
        self.maxpool_3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Conv2D Layer 4
        self.conv2d_4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU()
        self.maxpool_4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers with BatchNorm and Dropout
        self.fc1 = nn.Linear(512 * 2 * 16, 2048)  # Increased capacity
        self.bn_fc1 = nn.BatchNorm1d(2048)
        self.relu_fc1 = nn.ReLU()
        self.dropout_fc1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(2048, 1024)
        self.bn_fc2 = nn.BatchNorm1d(1024)
        self.relu_fc2 = nn.ReLU()
        self.dropout_fc2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(1024, num_classes)

        # Apply weight initialization
        self._initialize_weights()

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
        x = self.bn_fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout_fc1(x)

        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = self.relu_fc2(x)
        x = self.dropout_fc2(x)

        x = self.fc3(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_dir(self):
        path = 'models/' + self.name + '/'
        os.makedirs(path, exist_ok=True)
        return path
