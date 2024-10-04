import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Simple CNN Encoder
class SimpleEncoder(nn.Module):
    def __init__(self, in_channels):
        super(SimpleEncoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),  # Output: [batch, 32, 16, 128]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),           # Output: [batch, 64, 8, 64]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),          # Output: [batch, 128, 4, 32]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1))                                     # Output: [batch, 128, 1, 1]
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten to [batch_size, 128]
        return x

# Simplified Model for Multi-View, Multi-Modality Fusion
class SimpleFusionModel(nn.Module):
    # def __init__(self, num_classes=6):
    def __init__(self, name, num_classes=6, dtype=torch.float32):
        super(SimpleFusionModel, self).__init__()
        self.name = "SimpleFusionModel" + name
        self.modalities = ['Doppler', 'Azimuth', 'Elevation']
        self.num_modalities = len(self.modalities)

        # Modality-specific encoders with shared weights for views
        self.encoders = nn.ModuleDict()
        for modality in self.modalities:
            encoder = SimpleEncoder(in_channels=1)
            self.encoders[modality] = encoder

        # Fully connected layers after modality fusion
        self.fc = nn.Sequential(
            nn.Linear(128 * self.num_modalities, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x shape: [batch_size, 3, 2, 32, 256]
        modality_features = []

        for i, modality in enumerate(self.modalities):
            # Extract modality data
            modality_data = x[:, i, :, :, :]  # [batch_size, 2, 32, 256]

            # Split views
            view1 = modality_data[:, 0, :, :].unsqueeze(1)  # [batch_size, 1, 32, 256]
            view2 = modality_data[:, 1, :, :].unsqueeze(1)  # [batch_size, 1, 32, 256]

            # Process both views with the same encoder
            encoder = self.encoders[modality]
            feat_view1 = encoder(view1)  # [batch_size, 128]
            feat_view2 = encoder(view2)  # [batch_size, 128]

            # Fuse features from both views
            # Option 1: Element-wise addition
            fused_feat = feat_view1 + feat_view2  # [batch_size, 128]

            # Option 2: Concatenation followed by linear layer (uncomment if using)
            # fused_feat = torch.cat((feat_view1, feat_view2), dim=1)  # [batch_size, 256]
            # fused_feat = self.view_fusion_layers[modality](fused_feat)  # [batch_size, 128]

            modality_features.append(fused_feat)

        # Concatenate features from all modalities
        combined_features = torch.cat(modality_features, dim=1)  # [batch_size, 128 * num_modalities]

        # Pass through fully connected layers
        out = self.fc(combined_features)  # [batch_size, num_classes]

        return out

    def get_dir(self):
        path = 'models/' + self.name + '/'
        os.makedirs(path, exist_ok=True)
        return path
