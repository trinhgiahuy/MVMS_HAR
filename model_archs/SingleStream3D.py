import torch
import torch.nn as nn
import os

class SingleStream3D(nn.Module):
    def __init__(self, name, num_classes=7, dtype=torch.float16):
        super(SingleStream3D, self).__init__()

        self.name = 'SingleStream3D' + name
        self.dtype = dtype  # Set the desired data type

        # Conv3D Layer 1
        self.conv3d_1 = nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)).to(dtype=self.dtype)
        self.bn1 = nn.BatchNorm3d(64).to(dtype=self.dtype)
        self.relu1 = nn.ReLU().to(dtype=self.dtype)
        # Adjust pooling kernel to prevent shrinking of spatial dimensions
        self.maxpool_1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)).to(dtype=self.dtype)

        # Conv3D Layer 2
        self.conv3d_2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)).to(dtype=self.dtype)
        self.bn2 = nn.BatchNorm3d(128).to(dtype=self.dtype)
        self.relu2 = nn.ReLU().to(dtype=self.dtype)
        self.maxpool_2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)).to(dtype=self.dtype)

        # Flattening
        self.flatten = nn.Flatten().to(dtype=self.dtype)

        # Fully connected layers (adjusted based on the output size from convolutions)
        self.fc1 = nn.Linear(128 * 1 * 8 * 64, 512).to(dtype=self.dtype)
        self.relu3 = nn.ReLU().to(dtype=self.dtype)
        self.dropout = nn.Dropout(0.5).to(dtype=self.dtype)

        self.fc2 = nn.Linear(512, num_classes).to(dtype=self.dtype)

    def forward(self, x):
        x = x.to(self.dtype)  # Convert the input to the correct dtype
        x = self.conv3d_1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool_1(x)

        x = self.conv3d_2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool_2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    
    def get_dir(self):
        path = 'models/' + self.name + '/'
        os.makedirs(path, exist_ok=True)
        return path
