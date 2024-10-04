import os
import torch
import torch.nn as nn
import shutil
from datetime import datetime

class CustomCNN4_dropout_multi(nn.Module):
    def __init__(self, name, num_classes=10, dtype=torch.float32):
        super(CustomCNN4_dropout_multi, self).__init__()

        self.name = 'CustomCNN4_dropout_multi' + name

        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4), padding=0)
        self.pool_small = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)

        # LogSoftmax (optional, not used directly here)
        self.logSoftmax = nn.LogSoftmax(dim=1)

        # Convolutional layers (using 2 input channels as per radar feature maps)
        self.conv11 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=5)  # 2 channels
        self.conv12 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)  # No stride

        self.conv21 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=5)  # 2 channels
        self.conv22 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)  # No stride

        self.conv31 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=5)  # 2 channels
        self.conv32 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)  # No stride

        # Global Max Pooling layer
        self.global_pooling = nn.AdaptiveMaxPool2d(1)

        # Fully connected layers (we'll calculate the input size dynamically)
        self.fc1 = None  # Placeholder for fc1, we will set this later after figuring out the input size
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, num_classes)

        # Dropout layers
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)

    def get_dir(self):
        path = 'models/' + self.name + '/'
        os.makedirs(path, exist_ok=True)
        return path

    def forward(self, features):
        features = features.to(torch.float)

        # Assuming features are of the shape [batch_size, 3, height, width]
        # where each radar feature (Azimuth, Elevation, Doppler) is a separate channel
        x1 = features[:, 0, :, :, :]  # Azimuth map
        x2 = features[:, 1, :, :, :]  # Elevation map
        x3 = features[:, 2, :, :, :]  # Doppler map

        # Forward pass through the first stream
        x1 = self.pool(x1)
        x1 = torch.relu(self.conv11(x1))
        x1 = self.pool(x1)
        x1 = torch.relu(self.conv12(x1))

        # Forward pass through the second stream
        x2 = self.pool(x2)
        x2 = torch.relu(self.conv21(x2))
        x2 = self.pool(x2)
        x2 = torch.relu(self.conv22(x2))

        # Forward pass through the third stream
        x3 = self.pool(x3)
        x3 = torch.relu(self.conv31(x3))
        x3 = self.pool(x3)
        x3 = torch.relu(self.conv32(x3))

        # Flatten the outputs
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x3 = x3.view(x3.size(0), -1)

        # Concatenate along the feature dimension
        x = torch.cat((x1, x2, x3), dim=1)

        # Print the shape of the flattened tensor to debug the size
      #   print(f"Flattened tensor shape: {x.shape}")

        # Dynamically initialize the fully connected layer if it hasn't been initialized yet
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.shape[1], 128).to(x.device)

        # Fully connected layers with dropout
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)

        return x
